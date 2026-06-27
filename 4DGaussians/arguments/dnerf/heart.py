_base_ = './dnerf_default.py'

ModelParams = dict(
    white_background = False, 

    sh_degree = 1,
    
    source_path = "",
    
    resolution = 1, 
)

OptimizationParams = dict(
    #whole
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [128, 128, 128, 50]
    }, 
    time_smoothness_weight = 0.1,    
    plane_tv_weight = 0.01,          
    l1_time_planes = 0.01,     
)