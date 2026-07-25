import os, sys, json, glob, argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loss_utils import ssim as ssim_fn
from utils.image_utils import psnr as psnr_fn


def load_dir(d):
    fs = sorted(glob.glob(os.path.join(d, "*.png")))
    if not fs:
        fs = sorted(glob.glob(os.path.join(d, "*.jpg")))
    return fs


def to_tensor(p, size=None):
    im = Image.open(p)
    if im.mode == "RGBA":
        a = np.asarray(im, np.float32) / 255.0
        rgb = a[..., :3] * a[..., 3:4]
    else:
        rgb = np.asarray(im.convert("RGB"), np.float32) / 255.0
    if size is not None and rgb.shape[:2] != size:
        rgb = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8))
                         .resize((size[1], size[0]), Image.LANCZOS), np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--out", default="table3_results.json")
    ap.add_argument("--eval_size", type=int, default=800)
    ap.add_argument("--times", nargs="*", default=[])
    a = ap.parse_args()

    gt_files = load_dir(a.gt)
    if not gt_files:
        sys.exit(f"[X] GT 資料夾沒有影像：{a.gt}")
    gt0 = to_tensor(gt_files[0])
    H, W = gt0.shape[1], gt0.shape[2]
    EVAL = (a.eval_size, a.eval_size) if a.eval_size > 0 else None

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    except Exception as e:
        lpips_fn = None

    times = {}
    for t in a.times:
        k, v = t.split("=", 1)
        times[k.strip()] = v.strip()

    results = {}
    for spec in a.methods:
        name, d = spec.split("=", 1)
        name, d = name.strip(), d.strip()
        fs = load_dir(d)
        if not fs:
            continue
        if len(fs) != len(gt_files):
            print(f"[!] {name}: 張數 {len(fs)} != GT {len(gt_files)}，"
                  f"改以最小張數 {min(len(fs),len(gt_files))} 比較")
        n = min(len(fs), len(gt_files))
        P, S, L = [], [], []
        for i in range(n):
            g = to_tensor(gt_files[i], size=EVAL)
            r = to_tensor(fs[i], size=(g.shape[1], g.shape[2]))
            P.append(psnr_fn(g, r).mean().item())
            S.append(ssim_fn(g[None], r[None]).item())
            if lpips_fn is not None:
                L.append(lpips_fn(g[None], r[None], normalize=True).mean().item())
        results[name] = dict(
            n=n,
            psnr_mean=float(np.mean(P)), psnr_std=float(np.std(P, ddof=1)),
            ssim_mean=float(np.mean(S)), ssim_std=float(np.std(S, ddof=1)),
            lpips_mean=float(np.mean(L)) if L else None,
            lpips_std=float(np.std(L, ddof=1)) if L else None,
            time=times.get(name))
        print(f"[{name}] n={n}  PSNR {np.mean(P):.3f}±{np.std(P,ddof=1):.3f}  "
              f"SSIM {np.mean(S):.3f}±{np.std(S,ddof=1):.3f}"
              + (f"  LPIPS {np.mean(L):.3f}±{np.std(L,ddof=1):.3f}" if L else ""))

    json.dump({"gt": a.gt, "gt_n": len(gt_files), "resolution": [W, H],
               "results": results}, open(a.out, "w"), indent=2)

    names = list(results.keys())
    print("| | " + " | ".join(names) + " |")
    print("|---|" + "---:|" * len(names))
    if any(results[k]["time"] for k in names):
        print("| Time (ms) | " + " | ".join(results[k]["time"] or "-" for k in names) + " |")
    for key, lab, fmt in [("psnr", "PSNR ↑", "{:.3f} ± {:.3f}"),
                          ("ssim", "SSIM ↑", "{:.3f} ± {:.3f}"),
                          ("lpips", "LPIPS ↓", "{:.3f} ± {:.3f}")]:
        cells = []
        for k in names:
            m, s = results[k][key + "_mean"], results[k][key + "_std"]
            cells.append(fmt.format(m, s) if m is not None else "-")
        print(f"| {lab} | " + " | ".join(cells) + " |")

    print("\n" + "=" * 70 + "\nLaTeX\n" + "=" * 70)
    print("\\begin{tabular}{l" + "r" * len(names) + "}")
    print("\\toprule\n & " + " & ".join(names) + " \\\\\n\\midrule")
    if any(results[k]["time"] for k in names):
        print("Time (ms) & " + " & ".join(results[k]["time"] or "-" for k in names) + " \\\\")
    for key, lab in [("psnr", "PSNR $\\uparrow$"), ("ssim", "SSIM $\\uparrow$"),
                     ("lpips", "LPIPS $\\downarrow$")]:
        cells = []
        for k in names:
            m, s = results[k][key + "_mean"], results[k][key + "_std"]
            cells.append(f"${m:.3f} \\pm {s:.3f}$" if m is not None else "-")
        print(f"{lab} & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")
    print(f"\n已寫入 {a.out}")

if __name__ == "__main__":
    main()
