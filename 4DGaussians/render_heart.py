#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import imageio
import numpy as np
import torch
from scene import Scene
import os
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from time import time
import threading
import concurrent.futures
def multithread_write(image_list, path):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)
    def write_image(image, count, path):
        try:
            torchvision.utils.save_image(image, os.path.join(path, '{0:05d}'.format(count) + ".png"))
            return count, True
        except:
            return count, False
        
    tasks = []
    for index, image in enumerate(image_list):
        tasks.append(executor.submit(write_image, image, index, path))
    executor.shutdown()
    for index, status in enumerate(tasks):
        if status == False:
            write_image(image_list[index], index, path)
    
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)
def render_set(model_path, name, iteration, views, gaussians, pipeline, background, cam_type):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    render_images = []
    gt_list = []
    render_list = []
    print("point nums:",gaussians._xyz.shape[0])
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if idx == 0:time1 = time()
        
        rendering = render(view, gaussians, pipeline, background,cam_type=cam_type)["render"]
        render_images.append(to8b(rendering).transpose(1,2,0))
        render_list.append(rendering)
        if name in ["train", "test"]:
            if cam_type != "PanopticSports":
                gt = view.original_image[0:3, :, :]
            else:
                gt  = view['image'].cuda()
            gt_list.append(gt)

    time2=time()
    print("FPS:",(len(views)-1)/(time2-time1))

    multithread_write(gt_list, gts_path)

    multithread_write(render_list, render_path)

    
    imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=60)

# 請將這段程式碼覆蓋 render.py 裡的 render_sets 函式
# 記得要引入這個工具來重新計算投影矩陣
from utils.graphics_utils import getProjectionMatrix 

def render_sets(dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, hyperparam)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        cam_type=scene.dataset_type
        
        # 強制黑色背景
        bg_color = [0,0,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,cam_type)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,cam_type)
            
        if not skip_video:
            print("Processing Crystal Clear Fixed-View Video...")
            
            # ================= 自定義參數區 =================
            TARGET_IMAGE_NAME = "r_000_038"  # 填入你喜歡的圖片檔名 (不含 .png)
            ZOOM_SCALE = 1.5                     # 放大倍率
            video_length = 60                  # 影片長度
            # ===============================================

            import copy
            import json
            import os
            
            # ★ 終極破解法：去查 JSON 字典！
            json_path = os.path.join(dataset.source_path, "transforms_train.json")
            target_idx = -1
            
            try:
                with open(json_path, 'r') as f:
                    meta = json.load(f)
                    for i, frame in enumerate(meta['frames']):
                        if TARGET_IMAGE_NAME in frame['file_path']:
                            target_idx = i
                            break
            except Exception as e:
                print(f"⚠️ 讀取 JSON 失敗: {e}")

            base_view = None
            if target_idx != -1:
                # 去相機列表裡找名字是這個序號 (str) 的相機
                for cam in scene.getTrainCameras():
                    # 4DGS 載入器通常把 image_name 設為字串數字，或是用 uid 儲存
                    if getattr(cam, "image_name", "") == str(target_idx) or getattr(cam, "uid", -1) == target_idx:
                        base_view = cam
                        print(f"🎯 成功鎖定完美視角：{TARGET_IMAGE_NAME} (系統內部編號: {target_idx})")
                        break
            
            # 防呆機制
            if base_view is None:
                print(f"⚠️ 警告：找不到名為 '{TARGET_IMAGE_NAME}' 的視角。退回預設第 0 號相機。")
                base_view = scene.getTrainCameras()[0]
            
            # ★關鍵修正：自動偵測相機所在的裝置 (CPU 或 CUDA)
            device = base_view.world_view_transform.device
            
            # 計算新的 FOV (光學變焦)
            new_fovY = base_view.FoVy / ZOOM_SCALE
            new_fovX = base_view.FoVx / ZOOM_SCALE
            
            # 將新的矩陣放到同一個裝置上 (.to(device))
            new_projection_matrix = getProjectionMatrix(znear=base_view.znear, zfar=base_view.zfar, fovX=new_fovX, fovY=new_fovY).transpose(0,1).to(device)
            
            # 重新計算完整的投影矩陣
            new_full_proj_transform = (base_view.world_view_transform.unsqueeze(0).bmm(new_projection_matrix.unsqueeze(0))).squeeze(0)

            custom_cam_list = []
            
            for i in range(video_length):
                # 複製相機
                new_view = copy.deepcopy(base_view)
                
                # 套用新的變焦參數
                new_view.FoVy = new_fovY
                new_view.FoVx = new_fovX
                
                # =================== [修改] 強制將 Tensor 放在 CPU 上 ===================
                # 原始 4DGS 的 Camera 類別預期屬性是 CPU Tensor，到 render 裡面才 .cuda()
                new_view.projection_matrix = new_projection_matrix.cpu()
                new_view.world_view_transform = new_view.world_view_transform.cpu()
                new_view.full_proj_transform = new_full_proj_transform.cpu()
                new_view.camera_center = new_view.camera_center.cpu()
                # =========================================================================
                
                # 設定時間 (從 0.0 到 1.0 循環播放心臟跳動)
                new_view.time = i / video_length
                
                custom_cam_list.append(new_view)

            # 渲染
            render_set(dataset.model_path,"video",scene.loaded_iter,custom_cam_list,gaussians,pipeline,background,cam_type)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    args = get_combined_args(parser)
    print("Rendering " , args.model_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), hyperparam.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_video)