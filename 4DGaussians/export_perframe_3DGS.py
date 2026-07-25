import os
import numpy as np
import torch
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement

from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from utils.render_utils import get_state_at_time


class _TimeOnlyCam:
    def __init__(self, t):
        self.time = t


def construct_list_of_attributes(feature_dc_shape, feature_rest_shape, scaling_shape, rotation_shape):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(feature_dc_shape[1] * feature_dc_shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(feature_rest_shape[1] * feature_rest_shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling_shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotation_shape[1]):
        l.append('rot_{}'.format(i))
    return l


def init_3DGaussians_ply(points, scales, rotations, opactiy, shs, feature_shape):
    xyz = points.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    feature_dc = shs[:, 0:feature_shape[0], :]
    feature_rest = shs[:, feature_shape[0]:, :]
    f_dc = shs[:, :feature_shape[0], :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = shs[:, feature_shape[0]:, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opactiy.detach().cpu().numpy()
    scale = scales.detach().cpu().numpy()
    rotation = rotations.detach().cpu().numpy()

    dtype_full = [(a, 'f4') for a in construct_list_of_attributes(
        feature_dc.shape, feature_rest.shape, scales.shape, rotations.shape)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    return PlyData([PlyElement.describe(elements, 'vertex')])


if __name__ == "__main__":
    parser = ArgumentParser(description="Export per-frame deformed 3DGS")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n_frames", type=int, default=132)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_video", action="store_true")

    args = get_combined_args(parser)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        args = merge_hparams(args, mmcv.Config.fromfile(args.configs))
    safe_state(args.quiet)

    ds, hyper = model.extract(args), hyperparam.extract(args)
    pcd = os.path.join(args.model_path, f"point_cloud/iteration_{args.iteration}")

    with torch.no_grad():
        gaussians = GaussianModel(ds.sh_degree, hyper)
        gaussians.load_ply(os.path.join(pcd, "point_cloud.ply"))
        gaussians.load_model(pcd)
        gaussians._deformation = gaussians._deformation.cuda().eval()
        n_gauss = gaussians.get_xyz.shape[0]
        print(f"Exporting {args.model_path}  ({n_gauss:,} gaussians)")

        if args.n_frames and args.n_frames > 0:
            times = np.linspace(0.0, 1.0, args.n_frames, endpoint=False)
            cams = [_TimeOnlyCam(float(t)) for t in times]
        else:
            from scene import Scene
            scene = Scene(ds, gaussians, load_iteration=args.iteration, shuffle=False)
            cams = list(scene.getTestCameras())

        out_dir = os.path.join(args.model_path, "gaussian_pertimestamp")
        os.makedirs(out_dir, exist_ok=True)
        fdc = gaussians._features_dc.shape[1]
        frest = gaussians._features_rest.shape[1]

        for i, cam in enumerate(cams):
            pts, sc, rot, op, shs = get_state_at_time(gaussians, cam)
            ply = init_3DGaussians_ply(pts, sc, rot, op, shs, [fdc, frest])
            ply.write(os.path.join(out_dir, "time_{0:05d}.ply".format(i)))
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {i+1}/{len(cams)}")

    print(f"done -> {out_dir}")
