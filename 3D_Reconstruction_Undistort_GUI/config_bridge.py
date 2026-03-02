"""
config_bridge.py — GUI 與 StereoVisionSystem (先去畸變版本) 的橋接模組

此模組負責：
1. 將 GUI 輸入的參數覆蓋到 config 模組
2. 初始化並執行 StereoVisionSystem 的重建流程
"""

import os
import sys

# 將 3D_Reconstruction_Undistort 加入路徑以利匯入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3D_Reconstruction_Undistort")))

import config
from stereo_system import StereoVisionSystem

class GUIConfigBridge:
    """
    橋接 GUI 與 StereoVisionSystem (先去畸變版本)。
    允許從 GUI 注入參數，而不需要修改原始系統的 config.py。
    """

    def __init__(self, gui_params: dict):
        """
        初始化橋接器

        Args:
            gui_params: 從 GUI 收集的參數字典
        """
        self.params = gui_params
        self._override_config()

        # 初始化立體視覺系統
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
            # 處理可能的參數名稱差異
            elif key == 'POINTS_JSON_PATH':
                config.POINTS_JSON = value
            elif key == 'MANUAL_SAVE_PATH':
                config.MANUAL_POINTS_JSON = value

    def run_reconstruction(self):
        """
        啟動完整的 3D 重建流程：
        1. 載入校正參數並執行影像預去畸變
        2. 取得圖像點位 (JSON 載入或手動選點)
        3. 執行三角測量與座標對齊
        """
        # 1. 載入校正參數 (內部會自動執行影像預去畸變)
        self.system.load_parameters()

        # 2. 取得圖像點位
        mode = self.params.get('POINTS_MODE', 'json')
        if mode == "json":
            self.system.set_image_points(
                mode="json",
                json_path=self.params.get('POINTS_JSON')
            )
        else:
            self.system.set_image_points(
                mode="manual",
                num_points=int(self.params.get('NUM_POINTS', 15)),
                save_path=self.params.get('MANUAL_POINTS_JSON')
            )

        # 3. 執行重建與誤差分析
        vicon_path = self.params.get('VICON_CSV')
        # 若 VICON 路徑為空，傳入 None 以跳過座標對齊
        if not vicon_path or vicon_path.strip() == "":
            vicon_path = None

        self.system.process_reconstruction(vicon_csv=vicon_path)
