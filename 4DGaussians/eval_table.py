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


def alpha_of(p, size):
    im = Image.open(p)
    if im.mode != "RGBA":
        return None
    al = np.asarray(im, np.float32)[..., 3] / 255.0
    if size is not None and al.shape != size:
        al = np.asarray(Image.fromarray((al * 255).astype(np.uint8))
                        .resize((size[1], size[0]), Image.LANCZOS), np.float32) / 255.0
    return (torch.from_numpy(np.ascontiguousarray(al)).cuda() > 0.05)


def crop_box(mask):
    rows = torch.where(mask.any(1))[0]
    cols = torch.where(mask.any(0))[0]
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT 影像資料夾（4096 spp path trace 的 test）")
    ap.add_argument("--methods", nargs="+", required=True,
                    help='多個 "名稱=資料夾"，順序即表格欄位順序')
    ap.add_argument("--out", default="table3_results.json")
    ap.add_argument("--eval_size", type=int, default=800,
                    help="所有影像先統一縮到此邊長再比較。預設 800 = 4DGS 的原生輸出"
                         "解析度（dnerf loader 會把訓練影像硬縮成 800x800），"
                         "把 1600 的 GT/對照組降下來才是公平比較；若上採樣我們的 800 "
                         "輸出去比 1600，等於要求它產生它從未學過的高頻。0 = 用 GT 原尺寸。")
    ap.add_argument("--times", nargs="*", default=[],
                    help='可選，多個 "名稱=毫秒" 或 "名稱=mean±std"，填入 Time 列')
    ap.add_argument("--foreground", action="store_true",
                    help="只在 GT alpha>0.05 的前景遮罩內計算指標（小結構主表用，避免空背景稀釋）")
    a = ap.parse_args()

    gt_files = load_dir(a.gt)
    if not gt_files:
        sys.exit(f"[X] GT 資料夾沒有影像：{a.gt}")
    gt0 = to_tensor(gt_files[0])
    H, W = gt0.shape[1], gt0.shape[2]
    EVAL = (a.eval_size, a.eval_size) if a.eval_size > 0 else None
    print(f"[GT] {a.gt}  {len(gt_files)} 張  原始 {W}x{H}"
          + (f"  -> 全部統一縮至 {a.eval_size}x{a.eval_size} 比較" if EVAL else "")
          + ("  [前景遮罩內計算]" if a.foreground else ""))

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").cuda().eval()
    except Exception as e:
        print(f"[warn] LPIPS 不可用（{e}）")
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
            print(f"[!] {name}: 找不到影像 {d}，跳過")
            continue
        if len(fs) != len(gt_files):
            print(f"[!] {name}: 張數 {len(fs)} != GT {len(gt_files)}，"
                  f"改以最小張數 {min(len(fs),len(gt_files))} 比較")
        n = min(len(fs), len(gt_files))
        P, S, L, Frac = [], [], [], []
        for i in range(n):
            g = to_tensor(gt_files[i], size=EVAL)
            r = to_tensor(fs[i], size=(g.shape[1], g.shape[2]))
            if a.foreground:
                m = alpha_of(gt_files[i], (g.shape[1], g.shape[2]))
                if m is None or not m.any():
                    continue
                Frac.append(float(m.float().mean().item()))
                e = ((g - r) ** 2).mean(0)[m].mean().item()
                P.append(10 * np.log10(1.0 / max(e, 1e-12)))
                r0, r1, c0, c1 = crop_box(m)
                gc, rc = g[:, r0:r1, c0:c1], r[:, r0:r1, c0:c1]
                S.append(ssim_fn(gc[None], rc[None]).item())
                if lpips_fn is not None:
                    L.append(lpips_fn(gc[None], rc[None], normalize=True).mean().item())
            else:
                P.append(psnr_fn(g, r).mean().item())
                S.append(ssim_fn(g[None], r[None]).item())
                if lpips_fn is not None:
                    L.append(lpips_fn(g[None], r[None], normalize=True).mean().item())
        results[name] = dict(
            n=len(P), fg_frac=float(np.mean(Frac)) if Frac else None,
            psnr_mean=float(np.mean(P)), psnr_std=float(np.std(P, ddof=1)),
            ssim_mean=float(np.mean(S)), ssim_std=float(np.std(S, ddof=1)),
            lpips_mean=float(np.mean(L)) if L else None,
            lpips_std=float(np.std(L, ddof=1)) if L else None,
            time=times.get(name))
        fgtxt = f"  前景佔比 {np.mean(Frac)*100:.1f}%" if Frac else ""
        print(f"[{name}] n={len(P)}  PSNR {np.mean(P):.3f}±{np.std(P,ddof=1):.3f}  "
              f"SSIM {np.mean(S):.3f}±{np.std(S,ddof=1):.3f}"
              + (f"  LPIPS {np.mean(L):.3f}±{np.std(L,ddof=1):.3f}" if L else "") + fgtxt)

    json.dump({"gt": a.gt, "gt_n": len(gt_files), "resolution": [W, H],
               "results": results}, open(a.out, "w"), indent=2)

    names = list(results.keys())
    print("\n\n" + "=" * 70 + "\n貼進論文用（markdown）\n" + "=" * 70)
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
