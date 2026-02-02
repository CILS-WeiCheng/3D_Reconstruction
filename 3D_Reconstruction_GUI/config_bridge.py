import os
import sys

# 將 3D_Reconstruction_System 加入路徑以利匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3D_Reconstruction_System")))

import config
from stereo_system import StereoVisionSystem

class GUIConfigBridge:
    """
    橋接 GUI 與 StereoVisionSystem。
    允許開發者從 GUI 注入參數，而不需要修改原始系統的 config.py。
    """
    
    def __init__(self, gui_params: dict):
        self.params = gui_params
        self._override_config()
        
        self.system = StereoVisionSystem(
            left_calib=self.params.get('LEFT_NPZ'),
            right_calib=self.params.get('RIGHT_NPZ'),
            stereo_calib=self.params.get('STEREO_NPZ')
        )

    def _override_config(self):
        """
        將 GUI 的參數覆蓋到 config 模組中，
        確保 StereoVisionSystem 在執行時使用的是 GUI 選取的參數。
        """
        for key, value in self.params.items():
            if hasattr(config, key):
                setattr(config, key, value)
            # 有些參數名稱可能略有不同，這裡做對應
            elif key == 'POINTS_JSON_PATH':
                config.POINTS_JSON = value
            elif key == 'MANUAL_SAVE_PATH':
                config.MANUAL_POINTS_JSON = value

    def run_reconstruction(self):
        """
        啟動重建流程。
        """
        # 1. 載入校正參數
        self.system.load_parameters()
        
        # 2. 取得圖像點位
        mode = self.params.get('POINTS_MODE', 'json')
        if mode == "json":
            self.system.set_image_points(mode="json", json_path=self.params.get('POINTS_JSON'))
        else:
            self.system.set_image_points(
                mode="manual",
                left_img=self.params.get('IMG_L'),
                right_img=self.params.get('IMG_R'),
                num_points=int(self.params.get('NUM_POINTS', 15)),
                save_path=self.params.get('MANUAL_POINTS_JSON')
            )
        
        # 3. 執行重建與誤差分析
        vicon_path = self.params.get('VICON_CSV')
        # 如果 vicon_path 是空的，傳入 None
        if not vicon_path or vicon_path.strip() == "":
            vicon_path = None
            
        self.system.process_reconstruction(vicon_csv=vicon_path)
