import os, sys, glob, json, argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loss_utils import ssim as ssim_fn
from utils.image_utils import psnr as psnr_fn


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).cuda()


def alpha_mask_source(model_dir):
    import re
    src = None
    cfg = os.path.join(model_dir, "cfg_args")
    if os.path.exists(cfg):
        t = open(cfg).read()
        m = re.search(r"source_path='([^']+)'", t)
        if m:
            src = m.group(1)
    if not src:
        for cand in glob.glob(os.path.join("data", "dnerf", "*", os.path.basename(model_dir.rstrip("/\\")))):
            src = cand
    if not src or not os.path.isdir(os.path.join(src, "test")):
        return None
    j = os.path.join(src, "transforms_test.json")
    if not os.path.exists(j):
        return None
    order = [f["file_path"].split("/")[-1] for f in json.load(open(j))["frames"]]
    return [os.path.join(src, "test", nm + ".png") for nm in order]


def load_alpha(p, size):
    im = Image.open(p)
    if im.mode != "RGBA":
        return None
    a = np.asarray(im, np.float32)[..., 3] / 255.0
    t = torch.from_numpy(a)[None, None].cuda()
    t = torch.nn.functional.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return (t[0, 0] > 0.05)


def image_metrics(model_dir, it, lpips_fn):
    gt = sorted(glob.glob(os.path.join(model_dir, "**", f"test/ours_{it}/gt/*.png"), recursive=True))
    rd = sorted(glob.glob(os.path.join(model_dir, "**", f"test/ours_{it}/renders/*.png"), recursive=True))
    if not gt or len(gt) != len(rd):
        return None
    mask_src = alpha_mask_source(model_dir)
    P, S, L, Pf, Sf, Lf, Frac = [], [], [], [], [], [], []
    for i, (a, b) in enumerate(zip(gt, rd)):
        A, B = load(a), load(b)
        H = A.shape[1]
        P.append(psnr_fn(A, B).mean().item())
        S.append(ssim_fn(A[None], B[None]).item())
        if lpips_fn is not None:
            L.append(lpips_fn(A[None], B[None], normalize=True).mean().item())
        m = None
        if mask_src and i < len(mask_src) and os.path.exists(mask_src[i]):
            m = load_alpha(mask_src[i], H)
        if m is None:
            m = A.max(0).values > (2 / 255.0)
        Frac.append(float(m.float().mean().item()))
        if m.any():
            e = ((A - B) ** 2).mean(0)[m].mean().item()
            Pf.append(10 * np.log10(1.0 / max(e, 1e-12)))
            box = torch.where(m.any(1))[0], torch.where(m.any(0))[0]
            r0, r1, c0, c1 = box[0].min(), box[0].max() + 1, box[1].min(), box[1].max() + 1
            Ac, Bc = A[:, r0:r1, c0:c1], B[:, r0:r1, c0:c1]
            Sf.append(ssim_fn(Ac[None], Bc[None]).item())
            if lpips_fn is not None:
                Lf.append(lpips_fn(Ac[None], Bc[None], normalize=True).mean().item())
    return dict(n_views=len(gt), psnr=float(np.mean(P)), ssim=float(np.mean(S)),
                lpips=float(np.mean(L)) if L else None,
                psnr_fg=float(np.mean(Pf)), ssim_fg=float(np.mean(Sf)),
                lpips_fg=float(np.mean(Lf)) if Lf else None,
                fg_frac=float(np.mean(Frac)),
                mask=("alpha" if mask_src else "brightness"))


def geometry_metrics(model_dir, it):
    from plyfile import PlyData
    plys = sorted(glob.glob(os.path.join(model_dir, "**", "gaussian_pertimestamp", "*.ply"), recursive=True))
    src = "gaussian_pertimestamp (變形後全序列)"
    if not plys:
        plys = sorted(glob.glob(os.path.join(model_dir, "**", f"point_cloud/iteration_{it}/*.ply"), recursive=True))
        src = "canonical point_cloud"
    if not plys:
        return None
    ratios, counts = [], []
    for p in plys:
        v = PlyData.read(p)["vertex"]
        s = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
        ratios.append(s.max(1) / np.maximum(s.min(1), 1e-12))
        counts.append(len(s))
    r = np.concatenate(ratios)
    size_mb = sum(os.path.getsize(p) for p in plys) / 1024 ** 2
    return dict(source=src, n_frames=len(plys), n_gaussians=int(np.mean(counts)),
                ratio_median=float(np.median(r)),
                frac_gt5=float((r > 5).mean()), frac_gt10=float((r > 10).mean()),
                ratio_max=float(r.max()), total_ply_mb=float(size_mb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--iteration", type=int, default=20000)
    ap.add_argument("--out", type=str, default="eval_all_results.json")
    a = ap.parse_args()

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    except Exception as e:
        print(f"[warn] LPIPS 不可用（{e}），略過")
        lpips_fn = None

    res = {}
    for m in a.models:
        name = os.path.basename(m.rstrip("/\\"))
        print(f"\n=== {name}")
        im = image_metrics(m, a.iteration, lpips_fn)
        gm = geometry_metrics(m, a.iteration)
        if im is None:
            print("  [!] 找不到 test/ours_%d/{gt,renders}，請先跑 render_heart.py" % a.iteration)
        else:
            print("  n=%d  遮罩=%s  前景佔比 %.1f%%" % (im["n_views"], im["mask"], im["fg_frac"] * 100))
            print("  全圖  : PSNR %.2f  SSIM %.4f  LPIPS %s" %
                  (im["psnr"], im["ssim"], ("%.4f" % im["lpips"]) if im["lpips"] is not None else "n/a"))
            print("  前景  : PSNR %.2f  SSIM %.4f  LPIPS %s" %
                  (im["psnr_fg"], im["ssim_fg"], ("%.4f" % im["lpips_fg"]) if im["lpips_fg"] is not None else "n/a"))
        if gm is None:
            print("  [!] 找不到 .ply")
        else:
            print("  高斯 %d  軸比中位 %.2f  >5 %.2f%%  >10 %.2f%%  max %.1f  PLY %.0f MB  (%s)" %
                  (gm["n_gaussians"], gm["ratio_median"], gm["frac_gt5"] * 100,
                   gm["frac_gt10"] * 100, gm["ratio_max"], gm["total_ply_mb"], gm["source"]))
        res[name] = {"image": im, "geometry": gm}

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n\n| Model | fg% | PSNR-fg | SSIM-fg | LPIPS-fg | PSNR-full | #Gauss | ratio med | ratio max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in res.items():
        i, g = v["image"], v["geometry"]
        if not i or not g:
            continue
        print("| %s | %.1f%% | %.2f | %.4f | %s | %.2f | %d | %.2f | %.1f |" %
              (k, i["fg_frac"] * 100, i["psnr_fg"], i["ssim_fg"],
               ("%.4f" % i["lpips_fg"]) if i["lpips_fg"] is not None else "-",
               i["psnr"], g["n_gaussians"], g["ratio_median"], g["ratio_max"]))
    print(f"\n已寫入 {a.out}")


if __name__ == "__main__":
    main()
