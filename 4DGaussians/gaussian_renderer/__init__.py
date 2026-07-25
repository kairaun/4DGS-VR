import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, stage="fine", cam_type=None):

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass


    means3D = pc.get_xyz
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    else:
        raster_settings = viewpoint_camera['camera']
        time=torch.tensor(viewpoint_camera['time']).to(means3D.device).repeat(means3D.shape[0],1)


    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    means2D = screenspace_points
    opacity = pc._opacity
    shs = pc.get_features

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc._scaling
        rotations = pc._rotation
    deformation_point = pc._deformation_table
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = means3D, scales, rotations, opacity, shs
    elif "fine" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time)
    else:
        raise NotImplementedError


    scales_final = pc.scaling_activation(scales_final)
    rotations_final = pc.rotation_activation(rotations_final)
    opacity = pc.opacity_activation(opacity_final)
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pass
    else:
        colors_precomp = override_color
        shs_final = None

    rendered_image, radii, depth = rasterizer(
        means3D = means3D_final,
        means2D = means2D,
        shs = shs_final,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales_final,
        rotations = rotations_final,
        cov3D_precomp = cov3D_precomp)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth":depth}


def render_layered(viewpoint_camera, models, pipe, bg_color: torch.Tensor,
                   scaling_modifier=1.0, override_color=None, stage="fine", cam_type=None):
    n_each = [m.get_xyz.shape[0] for m in models]
    N = sum(n_each)
    train = models[-1]

    screenspace_points = torch.zeros((N, 3), dtype=train.get_xyz.dtype,
                                     requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    sh_deg = max(int(m.active_sh_degree) for m in models)
    if cam_type != "PanopticSports":
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=sh_deg,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug)
        t_val = viewpoint_camera.time
    else:
        raster_settings = viewpoint_camera['camera']
        t_val = viewpoint_camera['time']
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    def layer_state(m):
        means3D = m.get_xyz
        time = torch.tensor(t_val).to(means3D.device).repeat(means3D.shape[0], 1)
        opacity, shs = m._opacity, m.get_features
        scales, rotations = m._scaling, m._rotation
        if "coarse" in stage:
            mf, sf, rf, of, hf = means3D, scales, rotations, opacity, shs
        elif "fine" in stage:
            mf, sf, rf, of, hf = m._deformation(means3D, scales, rotations, opacity, shs, time)
        else:
            raise NotImplementedError
        return mf, m.scaling_activation(sf), m.rotation_activation(rf), m.opacity_activation(of), hf

    M3, SC, RO, OP, SH = [], [], [], [], []
    for i, m in enumerate(models):
        is_trainable = (i == len(models) - 1)
        if is_trainable:
            mf, sf, rf, of, hf = layer_state(m)
        else:
            with torch.no_grad():
                mf, sf, rf, of, hf = layer_state(m)
        M3.append(mf); SC.append(sf); RO.append(rf); OP.append(of); SH.append(hf)

    rendered_image, radii, depth = rasterizer(
        means3D=torch.cat(M3, 0),
        means2D=screenspace_points,
        shs=torch.cat(SH, 0),
        colors_precomp=None,
        opacities=torch.cat(OP, 0),
        scales=torch.cat(SC, 0),
        rotations=torch.cat(RO, 0),
        cov3D_precomp=None)

    start = sum(n_each[:-1])
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth,
            "train_slice": (start, N)}
