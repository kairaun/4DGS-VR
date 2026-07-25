import os, sys, argparse, subprocess, glob

INNER_LAYER = "valve"
OUTER_LAYER = "heart_valve"
LAYERS = [INNER_LAYER, OUTER_LAYER]
ITER = 20000
PORT = 6017


def run(cmd, dry):
    print("\n$ " + " ".join(cmd))
    if not dry:
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"[X] Faile（return {r.returncode}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--expbase", default="cine_layered")
    ap.add_argument("--iter", type=int, default=ITER)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--only", choices=["inner", "outer", "export", "all"], default="all")
    ap.add_argument("--n_frames", type=int, default=44)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    dry = a.dry_run
    py = sys.executable

    outdir = lambda layer: f"output/{a.expbase}/{layer}"
    inner_pcd = os.path.join(outdir(INNER_LAYER), f"point_cloud/iteration_{a.iter}")

    if a.only in ("inner", "all"):
        run([py, "train.py", "-s", f"{a.data}/{INNER_LAYER}",
             "--expname", f"{a.expbase}/{INNER_LAYER}",
             "--configs", a.config, "--port", str(a.port)], dry)

    if a.only in ("outer", "all"):
        run([py, "train.py", "-s", f"{a.data}/{OUTER_LAYER}",
             "--expname", f"{a.expbase}/{OUTER_LAYER}",
             "--configs", a.config, "--port", str(a.port),
             "--frozen_layers", inner_pcd], dry)

    if a.only in ("export", "all"):
        for layer in LAYERS:
            run([py, "export_perframe_3DGS.py", "--iteration", str(a.iter),
                 "--configs", a.config, "--model_path", outdir(layer),
                 "--n_frames", str(a.n_frames)], dry)
        print("\n下一步：python merge_layers.py --expbase output/%s --layers %s"
              % (a.expbase, " ".join(LAYERS)))


if __name__ == "__main__":
    main()
