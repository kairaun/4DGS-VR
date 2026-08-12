import os, sys, json, math, time, argparse
import numpy as np

LAB       = r"PATH/TO/ROOT"
CT_DIR    = f"{LAB}/CT ROOT"
MULTISEG_DIR = f"{LAB}/SEG ROOT"
ENVMAP    = f"{LAB}/table_mountain_2_puresky_1k.hdr"
OUT_DIR   = f"{LAB}/OUTPUT/DATA/DIR"

VALVE_COLOR = (0.98, 0.92, 0.55)
LAYERS = {
    "valve":       dict(labels=[2], colors={2: VALVE_COLOR},
                        smooth=1.5, gamma=1.8, min=0.0),
    "heart_valve": dict(labels=[1, 2], colors={1: None, 2: VALVE_COLOR}),
    "myocardium":  dict(labels=[1], colors={1: None}),
}
COLOR_DILATE = 4
LAYER_SEG_LABEL = {1: "心臟本體", 2: "主動脈瓣"}

N_VIEWS   = 120
IMG_SIZE  = 1600
SPP       = 4096
BOUNCES   = 4
ALBEDO    = 0.9
ENV_STRENGTH = 2.0
FOVY_DEG  = 50.0
FRAME_FILL = 0.88
DENSITY_SCALE = 3.2
CUTOFF    = 0.02
USE_HEART_MASK = True
MASK_SMOOTH_SIGMA = 3.0
MASK_CLOSE_ITER = 2
MASK_GAMMA = 2.2
VOL_SMOOTH_SIGMA = 0.7
EXPOSURE  = 0.55
SEED      = 42
N_INIT_POINTS = 100000

TF_POINTS = [
    (  -1024, 0.00, 0.00, 0.00, 0.000),
    (     35, 0.00, 0.00, 0.00, 0.000),
    (     75, 0.52, 0.04, 0.05, 0.070),
    (    120, 0.66, 0.06, 0.07, 0.300),
    (    175, 0.74, 0.09, 0.09, 0.500),
    (    260, 0.80, 0.13, 0.11, 0.640),
    (    380, 0.84, 0.15, 0.12, 0.820),
    (    480, 0.82, 0.12, 0.11, 0.930),
    (    580, 0.82, 0.13, 0.12, 0.985),
    (    680, 0.83, 0.15, 0.13, 0.998),
    (   2000, 0.84, 0.17, 0.15, 1.000),
]
LUT_N = 1024
HU_MIN, HU_MAX = -1024.0, 2000.0

MB = 8
TIMING_WARMUP = 3

import taichi as ti


def build_lut():
    hu = np.linspace(HU_MIN, HU_MAX, LUT_N)
    pts = np.array(TF_POINTS, dtype=np.float64)
    lut = np.zeros((LUT_N, 4), dtype=np.float32)
    for c in range(4):
        lut[:, c] = np.interp(hu, pts[:, 0], pts[:, c + 1])
    lut[lut[:, 3] < CUTOFF, 3] = 0.0
    return lut


def _clean_label(m):
    from scipy import ndimage as ndi
    lab, n = ndi.label(m)
    if n > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        m = lab == sizes.argmax()
    if MASK_CLOSE_ITER > 0:
        m = ndi.binary_closing(m, iterations=MASK_CLOSE_ITER)
        m = ndi.binary_fill_holes(m)
    return m


def _feather(m, smooth=None, gamma=None, fmin=0.0):
    from scipy import ndimage as ndi
    s = MASK_SMOOTH_SIGMA if smooth is None else smooth
    g = MASK_GAMMA if gamma is None else gamma
    f = m.astype(np.float32)
    if s > 0:
        f = ndi.gaussian_filter(f, s)
    f = np.clip(f, 0.0, 1.0) ** g
    if fmin > 0:
        f[f < fmin] = 0.0
    return f


def load_ct(idx):
    import nibabel as nib
    from scipy import ndimage as ndi
    ct = nib.load(os.path.join(CT_DIR, f"frame_{idx:02d}.nii.gz"))
    vol = np.asarray(ct.dataobj).astype(np.float32)
    if VOL_SMOOTH_SIGMA > 0:
        vol = ndi.gaussian_filter(vol, VOL_SMOOTH_SIGMA)
    return vol, ct.affine.astype(np.float64)


def load_frame(idx):
    vol, affine = load_ct(idx)
    mask = np.ones_like(vol)
    if USE_HEART_MASK:
        seg = load_multiseg(idx)
        union = np.zeros(vol.shape, bool)
        for m in seg.values():
            union |= m
        mask = _feather(_clean_label(union))
    return vol, mask, affine


def load_multiseg(idx):
    import nibabel as nib
    for name in (f"segmentation_frame_{idx:02d}_fold4.nii.gz",
                 f"frame_{idx:02d}.nii.gz",
                 f"segmentation_frame_{idx:02d}.nii.gz"):
        p = os.path.join(MULTISEG_DIR, name)
        if os.path.exists(p):
            seg = np.asarray(nib.load(p).dataobj)
            return {lab: _clean_label(seg == lab) for lab in LAYER_SEG_LABEL}
    raise FileNotFoundError(f"找不到 frame {idx} 的多標籤分割於 {MULTISEG_DIR}")


def build_layer_fields(vol, label_masks, layer_cfg, lut):
    idx = np.clip((vol - HU_MIN) / (HU_MAX - HU_MIN) * (LUT_N - 1), 0, LUT_N - 1).astype(np.int32)
    tf_op = lut[idx, 3]
    col = lut[idx, :3].astype(np.float32).copy()

    sig = np.zeros_like(vol, dtype=np.float32)
    colors = layer_cfg.get("colors", {})
    sm, gm = layer_cfg.get("smooth"), layer_cfg.get("gamma")
    fmin = layer_cfg.get("min", 0.0)
    for lb in layer_cfg["labels"]:
        f = _feather(label_masks[lb], sm, gm, fmin)
        sig = np.maximum(sig, tf_op * f)
        if colors.get(lb) is not None:
            from scipy import ndimage as ndi
            region = f > 0
            if COLOR_DILATE > 0:
                region = ndi.binary_dilation(region, iterations=COLOR_DILATE)
            col[region] = np.array(colors[lb], np.float32)
    return sig.astype(np.float32), col


def load_envmap(path):
    img = None
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if img is not None:
            img = img[:, :, ::-1].astype(np.float32)
    except Exception:
        pass
    if img is None:
        import imageio.v3 as iio
        img = iio.imread(path).astype(np.float32)
        if img.dtype == np.uint8 or img.max() > 300:
            img /= 255.0
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    lum = float((img * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2).mean())
    img = img / max(lum, 1e-6)
    return np.ascontiguousarray(img)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)], 1)


def split_of(i):
    m = i % 8
    return "test" if m == 0 else ("val" if m == 4 else "train")


def build_sigma_and_majorant(vol, mask, lut):
    idx = np.clip((vol - HU_MIN) / (HU_MAX - HU_MIN) * (LUT_N - 1), 0, LUT_N - 1).astype(np.int32)
    sig = (lut[idx, 3] * mask).astype(np.float32)
    col = lut[idx, :3].astype(np.float32)
    return sig, col, majorant_of(sig)


def majorant_of(sig):
    s = sig
    for ax in range(3):
        s = np.maximum(np.maximum(s, np.roll(s, 1, ax)), np.roll(s, -1, ax))
    pad = [(0, -(-n // MB) * MB - n) for n in sig.shape]
    s = np.pad(s, pad, mode="edge")
    mnx, mny, mnz = [n // MB for n in s.shape]
    return s.reshape(mnx, MB, mny, MB, mnz, MB).max(axis=(1, 3, 5)).astype(np.float32)


def write_ply(path, xyz, rgb255):
    n = xyz.shape[0]
    d = np.zeros((n, 9), np.float32)
    d[:, :3] = xyz; d[:, 6:9] = rgb255
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property float nx\nproperty float ny\nproperty float nz\n"
           "property float red\nproperty float green\nproperty float blue\n"
           "end_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(d.astype("<f4").tobytes())


ti.init(arch=ti.cuda, default_fp=ti.f32, random_seed=SEED)

SIG = None; COL = None; MAJ = None
ENV = None; ENV_CDF_M = None; ENV_CDF_C = None
BB_LO = None; BB_HI = None
NX = NY = NZ = 0; MNX = MNY = MNZ = 0; EH = EW = 0


def alloc(vol_shape, env_shape):
    global SIG, COL, MAJ, ENV, ENV_CDF_M, ENV_CDF_C, BB_LO, BB_HI
    global NX, NY, NZ, MNX, MNY, MNZ, EH, EW
    NX, NY, NZ = vol_shape
    MNX, MNY, MNZ = [-(-n // MB) for n in (NX, NY, NZ)]
    EH, EW = env_shape
    BB_LO = ti.Vector.field(3, ti.f32, shape=())
    BB_HI = ti.Vector.field(3, ti.f32, shape=())
    SIG = ti.field(ti.f32, shape=(NX, NY, NZ))
    COL = ti.Vector.field(3, ti.f32, shape=(NX, NY, NZ))
    MAJ = ti.field(ti.f32, shape=(MNX, MNY, MNZ))
    ENV = ti.Vector.field(3, ti.f32, shape=(EH, EW))
    ENV_CDF_M = ti.field(ti.f32, shape=EH)
    ENV_CDF_C = ti.field(ti.f32, shape=(EH, EW))


@ti.func
def sample_col(p):
    res = ti.Vector([0.0, 0.0, 0.0])
    if 0 <= p[0] <= NX - 1 and 0 <= p[1] <= NY - 1 and 0 <= p[2] <= NZ - 1:
        ix = ti.math.clamp(ti.cast(ti.floor(p[0]), ti.i32), 0, NX - 2)
        iy = ti.math.clamp(ti.cast(ti.floor(p[1]), ti.i32), 0, NY - 2)
        iz = ti.math.clamp(ti.cast(ti.floor(p[2]), ti.i32), 0, NZ - 2)
        fx = p[0] - ix; fy = p[1] - iy; fz = p[2] - iz
        for dx, dy, dz in ti.static([(u, v, w) for u in range(2)
                                     for v in range(2) for w in range(2)]):
            wt = ((1.0 - fx) if dx == 0 else fx) * \
                 ((1.0 - fy) if dy == 0 else fy) * \
                 ((1.0 - fz) if dz == 0 else fz)
            res += wt * COL[ix + dx, iy + dy, iz + dz]
    return res


@ti.func
def sigma_at(p):
    res = 0.0
    if 0 <= p[0] <= NX - 1 and 0 <= p[1] <= NY - 1 and 0 <= p[2] <= NZ - 1:
        ix = ti.math.clamp(ti.cast(ti.floor(p[0]), ti.i32), 0, NX - 2)
        iy = ti.math.clamp(ti.cast(ti.floor(p[1]), ti.i32), 0, NY - 2)
        iz = ti.math.clamp(ti.cast(ti.floor(p[2]), ti.i32), 0, NZ - 2)
        fx = p[0] - ix; fy = p[1] - iy; fz = p[2] - iz
        c = 0.0
        for dx, dy, dz in ti.static([(u, v, w) for u in range(2)
                                     for v in range(2) for w in range(2)]):
            wt = ((1.0 - fx) if dx == 0 else fx) * \
                 ((1.0 - fy) if dy == 0 else fy) * \
                 ((1.0 - fz) if dz == 0 else fz)
            c += wt * SIG[ix + dx, iy + dy, iz + dz]
        res = c * DENSITY_SCALE
    return res


@ti.func
def cell_exit_t(o, d, cx, cy, cz):
    res = 1e30
    for k in ti.static(range(3)):
        c = cx if k == 0 else (cy if k == 1 else cz)
        lo = ti.cast(c * MB, ti.f32)
        if ti.abs(d[k]) > 1e-8:
            b = (lo + MB) if d[k] > 0 else lo
            res = ti.min(res, (b - o[k]) / d[k])
    return res


@ti.func
def maj_at(p):
    cx = ti.math.clamp(ti.cast(ti.floor(p[0] / MB), ti.i32), 0, MNX - 1)
    cy = ti.math.clamp(ti.cast(ti.floor(p[1] / MB), ti.i32), 0, MNY - 1)
    cz = ti.math.clamp(ti.cast(ti.floor(p[2] / MB), ti.i32), 0, MNZ - 1)
    return ti.Vector([MAJ[cx, cy, cz] * DENSITY_SCALE,
                      ti.cast(cx, ti.f32), ti.cast(cy, ti.f32), ti.cast(cz, ti.f32)])


@ti.func
def track_collision(o, d, tmin, tmax):
    res = -1.0
    t = tmin
    done = 0
    guard = 0
    while done == 0 and t < tmax and guard < 4096:
        guard += 1
        p = o + (t + 1e-4) * d
        m4 = maj_at(p); sb = m4[0]
        t_out = ti.min(cell_exit_t(o, d, ti.cast(m4[1], ti.i32),
                                   ti.cast(m4[2], ti.i32),
                                   ti.cast(m4[3], ti.i32)), tmax)
        if sb > 1e-8:
            tt = t
            while True:
                tt -= ti.log(1.0 - ti.random()) / sb
                if tt >= t_out:
                    break
                if ti.random() * sb < sigma_at(o + tt * d):
                    res = tt
                    done = 1
                    break
        if done == 0:
            t = ti.max(t_out, t + 1e-3) + 1e-4
    return res


@ti.func
def track_transmittance(o, d):
    tr = 1.0
    tb = aabb(o, d)
    tmin = ti.max(tb[0], 0.0); tmax = tb[1]
    t = tmin
    guard = 0
    if tmax > tmin:
        while t < tmax and tr > 1e-4 and guard < 4096:
            guard += 1
            p = o + (t + 1e-4) * d
            m4 = maj_at(p); sb = m4[0]
            t_out = ti.min(cell_exit_t(o, d, ti.cast(m4[1], ti.i32),
                                       ti.cast(m4[2], ti.i32),
                                       ti.cast(m4[3], ti.i32)), tmax)
            if sb > 1e-8:
                tt = t
                while True:
                    tt -= ti.log(1.0 - ti.random()) / sb
                    if tt >= t_out:
                        break
                    tr *= 1.0 - sigma_at(o + tt * d) / sb
                    if tr < 1e-4:
                        break
            t = ti.max(t_out, t + 1e-3) + 1e-4
    return tr


@ti.func
def env_eval(d):
    u = (ti.atan2(d[1], d[0]) + math.pi) / (2.0 * math.pi)
    v = ti.acos(ti.math.clamp(d[2], -1.0, 1.0)) / math.pi
    x = ti.math.clamp(int(u * EW), 0, EW - 1)
    y = ti.math.clamp(int(v * EH), 0, EH - 1)
    return ENV[y, x] * ENV_STRENGTH


@ti.func
def env_sample():
    r1 = ti.random(); r2 = ti.random()
    lo, hi = 0, EH - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if ENV_CDF_M[mid] < r1: lo = mid + 1
        else: hi = mid
    y = lo
    lo2, hi2 = 0, EW - 1
    while lo2 < hi2:
        mid = (lo2 + hi2) // 2
        if ENV_CDF_C[y, mid] < r2: lo2 = mid + 1
        else: hi2 = mid
    x = lo2
    v = (y + 0.5) / EH; u = (x + 0.5) / EW
    theta = v * math.pi; phi = u * 2.0 * math.pi - math.pi
    st = ti.sin(theta)
    d = ti.Vector([st * ti.cos(phi), st * ti.sin(phi), ti.cos(theta)])
    pm = ENV_CDF_M[y] - (ENV_CDF_M[y - 1] if y > 0 else 0.0)
    pc = ENV_CDF_C[y, x] - (ENV_CDF_C[y, x - 1] if x > 0 else 0.0)
    pdf = pm * pc * (EH * EW) / (2.0 * math.pi * math.pi * ti.max(st, 1e-4))
    e = ENV[y, x] * ENV_STRENGTH
    return ti.Vector([d[0], d[1], d[2], e[0], e[1], e[2], ti.max(pdf, 1e-6)])


@ti.func
def aabb(o, d):
    dx = d[0] if ti.abs(d[0]) > 1e-8 else 1e-8
    dy = d[1] if ti.abs(d[1]) > 1e-8 else 1e-8
    dz = d[2] if ti.abs(d[2]) > 1e-8 else 1e-8
    inv = ti.Vector([1.0 / dx, 1.0 / dy, 1.0 / dz])
    t0 = (BB_LO[None] - o) * inv
    t1 = (BB_HI[None] - o) * inv
    tn = ti.min(t0, t1); tf_ = ti.max(t0, t1)
    return ti.Vector([ti.max(ti.max(tn[0], tn[1]), tn[2]),
                      ti.min(ti.min(tf_[0], tf_[1]), tf_[2])])


@ti.func
def sample_sphere():
    z = 1.0 - 2.0 * ti.random()
    r = ti.sqrt(ti.max(0.0, 1.0 - z * z))
    p = 2.0 * math.pi * ti.random()
    return ti.Vector([r * ti.cos(p), r * ti.sin(p), z])


@ti.kernel
def render(out: ti.types.ndarray(), cam_o: ti.types.vector(3, ti.f32),
           cam_x: ti.types.vector(3, ti.f32), cam_y: ti.types.vector(3, ti.f32),
           cam_z: ti.types.vector(3, ti.f32), tan_half: ti.f32,
           res: ti.i32, spp: ti.i32):
    for px, py in ti.ndrange(res, res):
        acc = ti.Vector([0.0, 0.0, 0.0]); acc_a = 0.0
        for _ in range(spp):
            sx = (2.0 * (px + ti.random()) / res - 1.0) * tan_half
            sy = (1.0 - 2.0 * (py + ti.random()) / res) * tan_half
            d = (cam_x * sx + cam_y * sy - cam_z).normalized()
            o = ti.Vector([cam_o[0], cam_o[1], cam_o[2]])

            L = ti.Vector([0.0, 0.0, 0.0]); thr = ti.Vector([1.0, 1.0, 1.0])
            hit_any = 0.0
            for _b in range(BOUNCES):
                tb = aabb(o, d)
                if tb[1] <= ti.max(tb[0], 0.0):
                    break
                t = track_collision(o, d, ti.max(tb[0], 0.0), tb[1])
                if t < 0.0:
                    break
                p = o + t * d
                hit_any = 1.0
                thr *= sample_col(p) * ALBEDO

                s7 = env_sample()
                ld = ti.Vector([s7[0], s7[1], s7[2]])
                le = ti.Vector([s7[3], s7[4], s7[5]])
                tr = track_transmittance(p, ld)
                L += thr * le * tr * (1.0 / (4.0 * math.pi)) / s7[6]

                o = p
                d = sample_sphere()
                if ti.random() > 0.9:
                    break
                thr /= 0.9
            acc += L; acc_a += hit_any
        c = acc / spp
        c = ti.math.clamp(c * EXPOSURE, 0.0, 1e4)
        c = (c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14)
        c = ti.math.clamp(c, 0.0, 1.0) ** (1.0 / 2.2)
        al = acc_a / spp
        inv = 1.0 / ti.max(al, 0.02)
        s = c * inv if al > 0.02 else ti.Vector([0.0, 0.0, 0.0])
        s = ti.math.clamp(s, 0.0, 1.0)
        out[py, px, 0] = s[0]; out[py, px, 1] = s[1]; out[py, px, 2] = s[2]
        out[py, px, 3] = al


def main():
    global DENSITY_SCALE, EXPOSURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--spp", type=int, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--views", type=int, default=N_VIEWS)
    ap.add_argument("--out", type=str, default=OUT_DIR)
    ap.add_argument("--density", type=float, default=DENSITY_SCALE)
    ap.add_argument("--exposure", type=float, default=EXPOSURE)
    ap.add_argument("--layered", action="store_true")
    ap.add_argument("--only_layers", nargs="+", default=None, choices=list(LAYERS.keys()))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    a = ap.parse_args()
    DENSITY_SCALE = a.density
    EXPOSURE = a.exposure
    spp = a.spp if a.spp is not None else (512 if a.preview else SPP)
    res = a.size if a.size is not None else (512 if a.preview else IMG_SIZE)
    out_dir = a.out + ("_preview" if a.preview else "")
    os.makedirs(out_dir, exist_ok=True)

    lut = build_lut()
    env = load_envmap(ENVMAP)
    v0, affine = load_ct(0)
    alloc(v0.shape, env.shape[:2])
    ENV.from_numpy(env)

    lum = env.mean(2) * np.sin((np.arange(env.shape[0]) + 0.5) / env.shape[0] * np.pi)[:, None]
    cond = np.cumsum(lum, 1); rowsum = cond[:, -1:].copy()
    cond /= np.maximum(rowsum, 1e-12)
    marg = np.cumsum(rowsum[:, 0]); marg /= max(marg[-1], 1e-12)
    ENV_CDF_C.from_numpy(cond.astype(np.float32))
    ENV_CDF_M.from_numpy(marg.astype(np.float32))
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    fovy = math.radians(FOVY_DEG)
    fovx = 2 * math.atan(math.tan(fovy / 2))

    if a.layered:
        seg0 = load_multiseg(0)
        vis0 = np.zeros(v0.shape, bool)
        for m in seg0.values():
            vis0 |= m
    else:
        _, m0, _ = load_frame(0)
        idxl = np.clip(((v0 - HU_MIN) / (HU_MAX - HU_MIN) * (LUT_N - 1)), 0, LUT_N - 1).astype(int)
        vis0 = (lut[idxl, 3] * m0) > CUTOFF
    idx = np.nonzero(vis0)
    lo_v = np.array([idx[0].min(), idx[1].min(), idx[2].min()], float)
    hi_v = np.array([idx[0].max(), idx[1].max(), idx[2].max()], float)
    BB_LO[None] = (lo_v - 2).clip(0).astype(np.float32)
    BB_HI[None] = np.minimum(hi_v + 2, np.array(v0.shape) - 1).astype(np.float32)
    center_v = (lo_v + hi_v) / 2.0
    r_mm = float(np.linalg.norm((hi_v - lo_v) * spacing) / 2.0)
    dist_mm = r_mm / math.sin(FRAME_FILL * fovy / 2.0)
    scale = 1.0 / r_mm
    dist_v = dist_mm / float(spacing.mean())

    dirs = fibonacci_sphere(a.views)
    n_time = len([f for f in os.listdir(CT_DIR) if f.endswith(".nii.gz")])
    view_list = range(0, a.views, max(1, a.views // 3)) if a.preview else range(a.views)
    time_list = [0] if a.preview else range(n_time)
    layers = list(LAYERS.keys()) if a.layered else [None]
    if a.layered and a.only_layers:
        layers = [L for L in layers if L in a.only_layers]

    def cam_basis(v):
        d = dirs[v]
        up = np.array([0., 1., 0.]) if abs(d[2]) > 0.98 else np.array([0., 0., 1.])
        cz = d / np.linalg.norm(d)
        cx = np.cross(up, cz); cx /= np.linalg.norm(cx)
        cy = np.cross(cz, cx)
        return cx, cy, cz, center_v + cz * dist_v

    from PIL import Image
    buf = np.zeros((res, res, 4), np.float32)
    t_all = time.time()
    render_ms = []
    n_warm = 0

    want_splits = set(a.splits)
    if want_splits != {"train", "val", "test"}:
        print(f"[note] 只渲染 splits={sorted(want_splits)}")

    for layer in layers:
        ldir = os.path.join(out_dir, layer) if layer else out_dir
        for sp in want_splits:
            os.makedirs(os.path.join(ldir, sp), exist_ok=True)
        frames = {k: [] for k in ["train", "val", "test"]}

        for t in time_list:
            vol, _ = load_ct(t)
            if layer:
                seg = load_multiseg(t)
                sig, col = build_layer_fields(vol, seg, LAYERS[layer], lut)
            else:
                _, msk, _ = load_frame(t)
                sig, col, _ = build_sigma_and_majorant(vol, msk, lut)
            SIG.from_numpy(np.ascontiguousarray(sig))
            COL.from_numpy(np.ascontiguousarray(col))
            MAJ.from_numpy(np.ascontiguousarray(majorant_of(sig)))
            tval = t / (n_time - 1) if n_time > 1 else 0.0
            tag = f"[{layer or 'heart'} t={t}]"
            print(f"{tag} sig>0 {float((sig>CUTOFF).mean())*100:.2f}%")
            for v in view_list:
                if split_of(v) not in want_splits:
                    continue
                cx, cy, cz, cam_o = cam_basis(v)
                args_ = (buf, cam_o.astype(np.float32), cx.astype(np.float32),
                         cy.astype(np.float32), cz.astype(np.float32),
                         math.tan(fovy / 2.0), res, spp)
                if n_warm < TIMING_WARMUP:
                    render(*args_); ti.sync(); n_warm += 1
                t0 = time.perf_counter()
                render(*args_); ti.sync()
                ms = (time.perf_counter() - t0) * 1000.0
                render_ms.append(ms)
                img = (np.clip(buf, 0, 1) * 255).astype(np.uint8)
                sp = split_of(v); name = f"r_{t:03d}_{v:03d}"
                Image.fromarray(img, "RGBA").save(os.path.join(ldir, sp, name + ".png"))
                c2w = np.eye(4)
                c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = cx, cy, cz
                c2w[:3, 3] = (cam_o - center_v) * spacing.mean() * scale
                frames[sp].append({"file_path": f"./{sp}/{name}", "rotation": 0.0,
                                   "time": tval, "transform_matrix": c2w.tolist()})

        for sp in want_splits:
            with open(os.path.join(ldir, f"transforms_{sp}.json"), "w") as f:
                json.dump({"camera_angle_x": fovx, "frames": frames[sp]}, f, indent=2)

        if not a.preview and want_splits == {"train", "val", "test"}:
            vol0, _ = load_ct(0)
            if layer:
                sig0, col0 = build_layer_fields(vol0, load_multiseg(0), LAYERS[layer], lut)
            else:
                _, msk0, _ = load_frame(0)
                sig0, col0, _ = build_sigma_and_majorant(vol0, msk0, lut)
            pidx = np.nonzero(sig0 > CUTOFF)
            pts = np.stack(pidx, 1).astype(np.float64)
            op = sig0[pidx]
            k = min(N_INIT_POINTS, len(pts))
            if k > 0:
                w = np.log(np.clip(op, 1e-6, None) ** 0.5) - np.log(-np.log(np.random.rand(len(op)) + 1e-12) + 1e-12)
                sel = np.argpartition(-w, k - 1)[:k]
                xyz = (pts[sel] + np.random.uniform(-.5, .5, (k, 3)) - center_v) * spacing.mean() * scale
                write_ply(os.path.join(ldir, "fused.ply"), xyz.astype(np.float32),
                          (col0[pidx[0][sel], pidx[1][sel], pidx[2][sel]] * 255).astype(np.float32))

    rm = np.array(render_ms)
    print("\n" + "=" * 62)
    print(f"渲染時間  {rm.mean():.1f} ± {rm.std():.1f} ms/張"
          f"   (中位 {np.median(rm):.1f}, 最小 {rm.min():.1f}, 最大 {rm.max():.1f})")
    print(f"設定      {res}x{res} px, {spp} spp, {BOUNCES} bounces, {len(rm)} 張"
          f"{'  x '+str(len(layers))+' 層' if a.layered else ''}")
    print(f"吞吐量    {res*res*spp/ (rm.mean()/1000) / 1e6:.0f} M sample-paths/s")
    print("=" * 62)
    with open(os.path.join(out_dir, "render_times.csv"), "w") as f:
        f.write("index,ms\n")
        for i, v in enumerate(rm):
            f.write(f"{i},{v:.3f}\n")

    json.dump({"n_timesteps": n_time, "n_views": a.views, "spp": spp, "bounces": BOUNCES,
               "layered": a.layered, "layers": layers if a.layered else None,
               "render_ms_mean": float(rm.mean()), "render_ms_std": float(rm.std()),
               "render_ms_median": float(np.median(rm)),
               "macrocell_size": MB, "n_images": int(len(rm)),
               "albedo": ALBEDO, "env_strength": ENV_STRENGTH, "envmap": ENVMAP,
               "fovy_deg": FOVY_DEG, "camera_angle_x_rad": fovx, "image_size": res,
               "object_radius_mm": r_mm, "camera_distance_mm": dist_mm,
               "world_to_normalized_scale": scale, "density_scale": DENSITY_SCALE},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    print(f"\n完成，耗時 {(time.time()-t_all)/60:.1f} 分鐘 -> {out_dir}")


if __name__ == "__main__":
    main()
