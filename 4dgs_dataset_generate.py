import os
import json
import math
import numpy as np
import vtk
import slicer

OUTPUT_BASE_DIR = ""

IMG_WIDTH = 1000
IMG_HEIGHT = 1000
VIEWS_PER_RING = 24  

def setup_render_window(width, height):
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    threeDWidget = layoutManager.threeDWidget(0)
    renderWindow = threeDWidget.threeDView().renderWindow()
    renderWindow.SetSize(width, height)
    slicer.app.processEvents()
    return threeDWidget, renderWindow

def get_camera_poses(center, radius):
    poses = []
    elevations = [-20, 0, 30, 60] 
    for elev_deg in elevations:
        elev = math.radians(elev_deg)
        for i in range(VIEWS_PER_RING):
            azimuth = (360.0 / VIEWS_PER_RING) * i
            azi = math.radians(azimuth)
            x = radius * math.cos(elev) * math.cos(azi) + center[0]
            y = radius * math.cos(elev) * math.sin(azi) + center[1]
            z = radius * math.sin(elev) + center[2]
            poses.append(np.array([x, y, z]))
    return poses

def get_camera_matrix(cameraNode):
    vtk_mat = cameraNode.GetCamera().GetModelViewTransformMatrix()
    c2w = np.eye(4)
    for i in range(4):
        for j in range(4):
            c2w[i, j] = vtk_mat.GetElement(i, j)
    return np.linalg.inv(c2w)

def save_frame(renderWindow, filepath, cameraNode, time, center, scale, split, frames_list):
    renderWindow.SetSize(IMG_WIDTH, IMG_HEIGHT)
    
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(renderWindow)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    
    c2w = get_camera_matrix(cameraNode)
    c2w[:3, 3] = (c2w[:3, 3] - center) * scale
    
    rel_path = "./" + split + "/" + os.path.basename(filepath).replace(".png", "")
    frames_list.append({
        "file_path": rel_path,
        "rotation": 0,
        "time": time,
        "transform_matrix": c2w.tolist()
    })

def main():
    browserNodes = slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')
    if len(browserNodes) == 0:
        print("Can't find 4D sequence controller")
        return
    browserNode = browserNodes[0]

    fidNodes = slicer.util.getNodesByClass('vtkMRMLMarkupsFiducialNode')
    if len(fidNodes) == 0:
        print("❌ Can't find middle point")
        return
    fidNode = fidNodes[0]

    if fidNode.GetDisplayNode():
        fidNode.GetDisplayNode().SetVisibility(False)

    sourceSeqNode = browserNode.GetMasterSequenceNode()
    volNode = browserNode.GetProxyNode(sourceSeqNode)
    fullBounds = [0,0,0,0,0,0]
    volNode.GetRASBounds(fullBounds)
    xmin, xmax, ymin, ymax, zmin, zmax = fullBounds

    center_coords = [0,0,0]
    fidNode.GetNthControlPointPosition(0, center_coords)
    cx, cy, cz = center_coords
    
    vol_ren_logic = slicer.modules.volumerendering.logic()
    displayNode = vol_ren_logic.GetFirstVolumeRenderingDisplayNode(volNode)
    if not displayNode:
        displayNode = vol_ren_logic.CreateDefaultVolumeRenderingNodes(volNode)
        
    displayNode.SetVisibility(True)
    displayNode.SetCroppingEnabled(True) 
    
    roiNode = displayNode.GetROINode()
    if not roiNode:
        vol_ren_logic.CreateROINode(displayNode)
        roiNode = displayNode.GetROINode()
    
    threeDWidget, renderWindow = setup_render_window(IMG_WIDTH, IMG_HEIGHT)
    cameraNode = slicer.util.getNodesByClass('vtkMRMLCameraNode')[0]
    cam = cameraNode.GetCamera()
    
    dims = [xmax-xmin, ymax-ymin, zmax-zmin]
    scene_radius = max(dims)
    fixed_cam_radius = scene_radius * 3.5
    scale_factor = 1.0 / fixed_cam_radius

    cut_configs = {
        "Whole": [xmin, xmax, ymin, ymax, zmin, zmax],
        #"Left_Half": [xmin, cx, ymin, ymax, zmin, zmax],
        #"Front_Half": [xmin, xmax, cy, ymax, zmin, zmax],
        #"Top_Half": [xmin, xmax, ymin, ymax, cz, zmax],
    }

    camera_poses = get_camera_poses(center_coords, fixed_cam_radius)
    num_time_steps = browserNode.GetNumberOfItems()

    for cut_name, cut_bounds in cut_configs.items():
        print(f"\nGenerate: {cut_name}")
        
        roi_center = [(cut_bounds[0] + cut_bounds[1]) / 2.0,
                      (cut_bounds[2] + cut_bounds[3]) / 2.0,
                      (cut_bounds[4] + cut_bounds[5]) / 2.0]
        roi_radius = [(cut_bounds[1] - cut_bounds[0]) / 2.0,
                      (cut_bounds[3] - cut_bounds[2]) / 2.0,
                      (cut_bounds[5] - cut_bounds[4]) / 2.0]
        
        roiNode.SetXYZ(roi_center)
        roiNode.SetRadiusXYZ(roi_radius)
        slicer.app.processEvents()

        layer_dir = os.path.join(OUTPUT_BASE_DIR, cut_name)
        splits = ["train", "val", "test"]
        dirs = {k: os.path.join(layer_dir, k) for k in splits}
        for d in dirs.values(): os.makedirs(d, exist_ok=True)
        json_frames = {k: [] for k in splits}
        
        img_counter = 0

        for t_idx in range(num_time_steps):
            browserNode.SetSelectedItemNumber(t_idx)
            slicer.app.processEvents()
            current_time = t_idx / (num_time_steps - 1) if num_time_steps > 1 else 0.0
            markup_rois = slicer.util.getNodesByClass("vtkMRMLMarkupsROINode")
            for roi in markup_rois:
                roi.SetDisplayVisibility(False)

            annotation_rois = slicer.util.getNodesByClass("vtkMRMLAnnotationROINode")
            for roi in annotation_rois:
                roi.SetDisplayVisibility(0)
                
            for p_idx, pos in enumerate(camera_poses):
                cam.SetPosition(pos)
                cam.SetFocalPoint(center_coords)
                cam.SetViewUp(0, 0, 1)
                cam.OrthogonalizeViewUp()
                
                threeDWidget.threeDView().forceRender()
                slicer.app.processEvents()
                
                cycle_idx = img_counter % 8
                if cycle_idx == 0: split_name = "test"
                elif cycle_idx == 1: split_name = "val"
                else: split_name = "train"
                
                filename = f"r_{t_idx:03d}_{p_idx:03d}.png"
                filepath = os.path.join(dirs[split_name], filename)
                
                save_frame(renderWindow, filepath, cameraNode, current_time, center_coords, scale_factor, split_name, json_frames[split_name])
                img_counter += 1
                
            print(f"[{cut_name}] ✅ Frame {t_idx+1}/{num_time_steps} done.")

        fov_rad = cam.GetViewAngle() * (math.pi / 180.0)
        for split in splits:
            json_out = {"camera_angle_x": fov_rad, "frames": json_frames[split]}
            json_path = os.path.join(layer_dir, f"transforms_{split}.json")
            with open(json_path, 'w') as f:
                json.dump(json_out, f, indent=4)
        print(f"{cut_name} JSON Saved, Total: {img_counter} ")
main()