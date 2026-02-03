import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from config import VICON_MM_TO_M
from util import sort_point_keys

def load_calibration_params(left_path: str, right_path: str, stereo_path: str) -> Dict[str, np.ndarray]:
    """載入單目與雙目校正參數"""
    left_data = np.load(left_path, allow_pickle=True)
    right_data = np.load(right_path, allow_pickle=True)
    stereo_data = np.load(stereo_path, allow_pickle=True)
    
    return {
        "mtxL": left_data["camera_matrix"],
        "distL": left_data["dist_coeffs"],
        "mtxR": right_data["camera_matrix"],
        "distR": right_data["dist_coeffs"],
        "R": stereo_data["R"],
        "T": stereo_data["T"]
    }

def load_points_json(json_path: str) -> Dict[str, Dict[str, List[int]]]:
    """從 JSON 載入點位座標"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_points_json(points: Dict, json_path: str) -> None:
    """將點位座標儲存為 JSON，並自動建立中間目錄"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=4, ensure_ascii=False)

def load_vicon_csv(csv_path: str, point_keys: List[str]) -> Dict[str, List[float]]:
    """從 CSV 載入 VICON Ground Truth"""
    df = pd.read_csv(csv_path)
    vicon_points = {}
    
    for point_key in point_keys:
        try:
            point_idx = int(point_key.replace("point", "")) - 1
            col_start = 2 + point_idx * 3
            # 取得數值並轉為公尺 (mm -> m)
            vals = df.iloc[1, col_start:col_start+3].values.astype(float) / VICON_MM_TO_M
            if not np.isnan(vals).any():
                vicon_points[point_key] = vals.tolist()
        except (IndexError, ValueError):
            continue
            
    return vicon_points
