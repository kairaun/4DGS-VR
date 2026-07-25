import os, glob, argparse
import numpy as np
from plyfile import PlyData


def load_frame(path, opacity_min):
    v = PlyData.read(path)["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    op = 1.0 / (1.0 + np.exp(-np.asarray(v["opacity"], np.float64)))
    m = op > opacity_min
    return xyz[m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", required=True, help="外層模型目錄（心臟）")
    ap.add_argument("--inner", required=True, help="內層模型目錄（瓣膜）")
    ap.add_argument("--label", default="")
    ap.add_argument("--opacity_min", type=float, default=0.1)
    ap.add_argument("--knn_r", type=float, default=0.03,
                    help="判定「被外層包覆」的近鄰半徑（正規化單位，物體半徑=1）")
    ap.add_argument("--scale_mm", type=float, default=103.75)
    a = ap.parse_args()

    fo = sorted(glob.glob(os.path.join(a.outer, "gaussian_pertimestamp", "time_*.ply")))
    fi = sorted(glob.glob(os.path.join(a.inner, "gaussian_pertimestamp", "time_*.ply")))
    n = min(len(fo), len(fi))
    if n == 0:
        raise SystemExit("[X] 找不到逐幀 .ply，請先 export")
    print(f"=== {a.label or a.inner} ===")
    print(f"外層 {a.outer}\n內層 {a.inner}\n幀數 {n}\n")

    from scipy.spatial import cKDTree
    contain, cdist, icent, ocent = [], [], [], []
    for k in range(n):
        O = load_frame(fo[k], a.opacity_min)
        I = load_frame(fi[k], a.opacity_min)
        tree = cKDTree(O)
        d, _ = tree.query(I, k=1)
        contain.append(float((d < a.knn_r).mean()))
        cdist.append(float(np.median(d)))
        icent.append(I.mean(0)); ocent.append(O.mean(0))

    icent = np.array(icent); ocent = np.array(ocent)
    rel = icent - ocent
    rel_drift = np.linalg.norm(rel - rel[0], axis=1)

    C = np.array(contain); D = np.array(cdist) * a.scale_mm
    print("逐幀「內層高斯被外層包覆」的比例（半徑 %.3f = %.2f mm）:" % (a.knn_r, a.knn_r * a.scale_mm))
    print("   平均 %.3f   最小 %.3f   最大 %.3f" % (C.mean(), C.min(), C.max()))
    print("內層到外層最近點的中位距離 (mm):")
    print("   平均 %.3f   最小 %.3f   最大 %.3f" % (D.mean(), D.min(), D.max()))
    print("內層質心相對外層質心的漂移 (mm，相對第 0 幀):")
    print("   平均 %.3f   最大 %.3f" % (rel_drift.mean() * a.scale_mm,
                                        rel_drift.max() * a.scale_mm))
    print()
    print("判讀：包覆比例越高、最近點距離越小、相對漂移越小 = 解剖一致性越好。")
    print("      兩種變體要用「相同的外層」比較才有意義。")


if __name__ == "__main__":
    main()
