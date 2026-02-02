from stereo_system import StereoVisionSystem
import config

def example_stereo_reconstruction():
    """
    展示 3D 立體重建系統的範例用法
    """
    print("=== 啟動 3D 重建系統範例 ===")
    
    # 1. 初始化系統 (路徑由 config 提供)
    system = StereoVisionSystem(
        left_calib=config.LEFT_NPZ,
        right_calib=config.RIGHT_NPZ,
        stereo_calib=config.STEREO_NPZ
    )
    
    # 2. 載入校正參數
    system.load_parameters()
    
    # 3. 取得圖像點位
    if config.POINTS_MODE == "json":
        system.set_image_points(mode="json", json_path=config.POINTS_JSON)
    else:
        system.set_image_points(
            mode="manual",
            left_img=config.IMG_L,
            right_img=config.IMG_R,
            num_points=config.NUM_POINTS,
            save_path=config.MANUAL_POINTS_JSON
        )
    
    # 4. 執行重建與誤差分析
    system.process_reconstruction(vicon_csv=config.VICON_CSV)
    
    print("=== 範例執行結束 ===")

def main():
    """主程式進入點"""
    # 呼叫範例功能
    example_stereo_reconstruction()

if __name__ == "__main__":
    main()
