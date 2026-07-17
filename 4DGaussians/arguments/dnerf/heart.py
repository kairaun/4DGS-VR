_base_ = './dnerf_default.py'

ModelParams = dict(
    # ★關鍵修正：CT 影像是黑底，一定要設 False！
    white_background = False, 
    
    # ★優化：CT 沒有反光，不需要複雜的光影計算 (SH)。
    sh_degree = 1,
    
    # 這是你的資料路徑
    source_path = "data/dnerf/heart_cut_data2/Whole",
    
    # 設為 1 代表使用原圖解析度。
    resolution = 1, 
)

OptimizationParams = dict(
    # [修改] 將閾值降到極低。搭配等一下的邊緣損失，
    # 讓微血管邊緣能順利觸發分裂，長出大量細緻的小點。

    #whole
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    #cut
    #densify_grad_threshold_fine_init = 0.0005,
    #densify_grad_threshold_after = 0.0005,
)

ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 32,
     'resolution': [128, 128, 128, 50]
    }, 
    # [動靜態解耦約束]
    time_smoothness_weight = 0.1,    
    plane_tv_weight = 0.01,          
    # ★ 關鍵：L1 Sparsity 放大 100 倍至 0.01，徹底釘死沒有動態特徵的靜態骨頭！
    l1_time_planes = 0.01,     
)