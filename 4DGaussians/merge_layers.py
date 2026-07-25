import os, sys, json, glob, argparse
import numpy as np
from plyfile import PlyData, PlyElement


def load_ply(path):
    v = PlyData.read(path)["vertex"]
    return v, v.count, list(v.data.dtype.names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expbase", required=True)
    ap.add_argument("--layers", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_dir = a.out or os.path.join(a.expbase, "merged")
    os.makedirs(out_dir, exist_ok=True)

    per_layer = {}
    for L in a.layers:
        fs = sorted(glob.glob(os.path.join(a.expbase, L, "gaussian_pertimestamp", "time_*.ply")))
        if not fs:
            sys.exit(f"[X] 找不到 {L}  gaussian_pertimestamp/*.ply")
        per_layer[L] = fs
    n = len(next(iter(per_layer.values())))
    for L, fs in per_layer.items():
        if len(fs) != n:
            sys.exit(f"[X] 層 {L} 有 {len(fs)} 幀，與其他層({n})不一致 —— 相機必須相同")

    total = 0
    for i in range(n):
        chunks = []
        names0 = None
        for lid, L in enumerate(a.layers):
            v, cnt, names = load_ply(per_layer[L][i])
            if names0 is None:
                names0 = names
            elif names != names0:
                sys.exit(f"[X] 層 {L} 的 PLY 欄位與第一層不同，無法合併")
            arr = np.zeros(cnt, dtype=v.data.dtype)
            arr[:] = v.data
            chunks.append((arr, lid, cnt))

        cnt_all = sum(c for _, _, c in chunks)
        dtype_new = list(chunks[0][0].dtype.descr) + [("layer", "u1")]
        merged = np.empty(cnt_all, dtype=dtype_new)
        off = 0
        for arr, lid, cnt in chunks:
            for nm in arr.dtype.names:
                merged[nm][off:off + cnt] = arr[nm]
            merged["layer"][off:off + cnt] = lid
            off += cnt
        PlyData([PlyElement.describe(merged, "vertex")]).write(
            os.path.join(out_dir, f"time_{i:05d}.ply"))
        total += cnt_all
        if i == 0 or (i + 1) % 20 == 0:
            print(f"  幀 {i+1}/{n}  合併 {cnt_all} 高斯（{[c for _,_,c in chunks]}）")

    json.dump({str(i): L for i, L in enumerate(a.layers)},
              open(os.path.join(out_dir, "layer_map.json"), "w"), indent=2)
    print(f"\n完成：{out_dir}  共 {n} 幀、平均 {total//n} 高斯/幀")
    print("layer_map.json:", {i: L for i, L in enumerate(a.layers)})


if __name__ == "__main__":
    main()
