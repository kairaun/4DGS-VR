import os, sys, argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, ModelHiddenParams, get_combined_args


def main():
    p = argparse.ArgumentParser()
    lp, hp = ModelParams(p, sentinel=True), ModelHiddenParams(p)
    p.add_argument("--iteration", type=int, default=20000)
    p.add_argument("--configs", type=str, default="")
    p.add_argument("--scale_mm", type=float, default=None,
                   help="物體外接球半徑(mm)，用來把正規化位移換算成 mm（見 meta.json）")
    p.add_argument("--n_times", type=int, default=11)
    a = get_combined_args(p)
    if a.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        a = merge_hparams(a, mmcv.Config.fromfile(a.configs))
    ds, hyper = lp.extract(a), hp.extract(a)

    pcd = os.path.join(a.model_path, f"point_cloud/iteration_{a.iteration}")
    g = GaussianModel(ds.sh_degree, hyper)
    g.load_ply(os.path.join(pcd, "point_cloud.ply"))
    g.load_model(pcd)
    g._deformation = g._deformation.cuda().eval()
    print(f"[1] 載入 {g.get_xyz.shape[0]:,} 個高斯 + 變形場 from {pcd}")
    print(f"    hexplane bounds = {hyper.bounds}")

    xyz = g.get_xyz.detach()
    sc, rot, op = g._scaling.detach(), g._rotation.detach(), g._opacity.detach()
    shs = g.get_features.detach()

    ref = None
    disp_all = []
    with torch.no_grad():
        for i in range(a.n_times):
            t = i / (a.n_times - 1)
            tt = torch.full((xyz.shape[0], 1), t, device="cuda")
            pt, sf, rf, of, sh = g._deformation(xyz, sc, rot, op, shs, tt)
            if ref is None:
                ref = pt
            d = (pt - ref).norm(dim=1)
            disp_all.append(d)
            mm = f" = {d.mean().item()*a.scale_mm:6.2f} mm" if a.scale_mm else ""
            print(f"    t={t:.2f}  位移 vs t=0：mean {d.mean().item():.5f}  "
                  f"p95 {torch.quantile(d,0.95).item():.5f}  max {d.max().item():.5f}{mm}")

    D = torch.stack(disp_all)
    per_g = D.max(0).values
    print("\n[2] 整個心動週期，每個高斯的最大位移：")
    for q in [0.5, 0.9, 0.99, 1.0]:
        v = torch.quantile(per_g, q).item() if q < 1 else per_g.max().item()
        mm = f" = {v*a.scale_mm:6.2f} mm" if a.scale_mm else ""
        print(f"    p{int(q*100):3d}  {v:.5f}{mm}")
    moving = (per_g > 0.005).float().mean().item()
    print(f"    位移 > 0.005 (正規化) 的高斯佔比：{moving*100:.1f}%")

    print("\n[3] 判讀：")
    med_mm = torch.quantile(per_g, 0.5).item() * (a.scale_mm or 1)
    if a.scale_mm:
        if med_mm < 0.5:
            print("    ⚠️ 變形場幾乎靜態（中位最大位移 < 0.5mm）—— Stage 2 的共享運動場前提有疑慮")
        elif med_mm < 3:
            print(f"    中位最大位移 {med_mm:.2f} mm：有運動但幅度偏小，心臟收縮應有數 mm 等級")
        else:
            print(f"    ✓ 中位最大位移 {med_mm:.2f} mm，符合心動週期的合理幅度")


if __name__ == "__main__":
    main()
