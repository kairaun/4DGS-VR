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


def image_metrics(model_dir, it, lpips_fn):
    gt = sorted(glob.glob(os.path.join(model_dir, "**", f"test/ours_{it}/gt/*.png"), recursive=True))
    rd = sorted(glob.glob(os.path.join(model_dir, "**", f"test/ours_{it}/renders/*.png"), recursive=True))
    if not gt or len(gt) != len(rd):
        return None
    P, S, L, Pf = [], [], [], []
    for a, b in zip(gt, rd):
        A, B = load(a), load(b)
        P.append(psnr_fn(A, B).mean().item())
        S.append(ssim_fn(A[None], B[None]).item())
        if lpips_fn is not None:
            L.append(lpips_fn(A[None], B[None], normalize=True).mean().item())
        m = A.max(0).values > (2 / 255.0)
        if m.any():
            e = ((A - B) ** 2).mean(0)[m].mean().item()
            Pf.append(10 * np.log10(1.0 / max(e, 1e-12)))
    return dict(n_views=len(gt), psnr=float(np.mean(P)), ssim=float(np.mean(S)),
                lpips=float(np.mean(L)) if L else None, psnr_fg=float(np.mean(Pf)))


def geometry_metrics(model_dir, it):
    from plyfile import PlyData
    plys = sorted(glob.glob(os.path.join(model_dir, "**", "gaussian_pertimestamp", "*.ply"), recursive=True))
    src = "gaussian_pertimestamp"
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
            print("  n=%d  PSNR %.2f  SSIM %.4f  LPIPS %s  PSNR-fg %.2f" %
                  (im["n_views"], im["psnr"], im["ssim"],
                   ("%.4f" % im["lpips"]) if im["lpips"] is not None else "n/a", im["psnr_fg"]))
        if gm is None:
            print("  [!] 找不到 .ply")
        else:
            print("  高斯 %d  軸比中位 %.2f  >5 %.2f%%  >10 %.2f%%  max %.1f  PLY %.0f MB  (%s)" %
                  (gm["n_gaussians"], gm["ratio_median"], gm["frac_gt5"] * 100,
                   gm["frac_gt10"] * 100, gm["ratio_max"], gm["total_ply_mb"], gm["source"]))
        res[name] = {"image": im, "geometry": gm}

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n\n| Model | PSNR | SSIM | LPIPS | PSNR-fg | #Gauss | ratio med | >10 | ratio max |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in res.items():
        i, g = v["image"], v["geometry"]
        if not i or not g:
            continue
        print("| %s | %.2f | %.4f | %s | %.2f | %d | %.2f | %.2f%% | %.1f |" %
              (k, i["psnr"], i["ssim"],
               ("%.4f" % i["lpips"]) if i["lpips"] is not None else "-",
               i["psnr_fg"], g["n_gaussians"], g["ratio_median"],
               g["frac_gt10"] * 100, g["ratio_max"]))
    print(f"\n已寫入 {a.out}")


if __name__ == "__main__":
    main()
