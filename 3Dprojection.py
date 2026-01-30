import os
from typing import Dict, List, Tuple, Optional

import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ESC_KEY = 27
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3
VICON_MM_TO_M = 1000.0

BASE_DIR = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate"
PARAM_DIR = r"./20260114_實驗/標定參數"

LEFT_NPZ = os.path.join(PARAM_DIR, "20260114_left_single.npz")
RIGHT_NPZ = os.path.join(PARAM_DIR, "20260114_right_single.npz")
STEREO_NPZ = os.path.join(PARAM_DIR, "stereo_rt.npz")

POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_point.json")
VICON_CSV = os.path.join(BASE_DIR, "20260114_實驗/Vicon_court/court_15points.csv")
RAW_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints.json")
ALIGNED_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints_svd.json")
ERROR_PLOT_PATH = os.path.join(BASE_DIR, "20260114_實驗/court/xy_error_plot.png")

# 手動選點相關設定
IMG_L = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_L.jpg")
IMG_R = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_R.jpg")
MANUAL_POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/code_point/court_point.json")
NUM_POINTS = 3

# 點位來源模式: 'json' 或 'manual'
POINTS_MODE = "json"

class StereoVisionSystem:
    """雙目視覺系統，用於立體視覺重建與座標轉換"""
    
    def __init__(self, left_calib_path: str, right_calib_path: str, stereo_calib_path: str):
        """
        初始化雙目視覺系統
        
        Args:
            left_calib_path: 左相機校正參數檔案路徑
            right_calib_path: 右相機校正參數檔案路徑
            stereo_calib_path: 雙目校正參數檔案路徑
        """
        self.left_calib_path = left_calib_path
        self.right_calib_path = right_calib_path
        self.stereo_calib_path = stereo_calib_path
        
        # 參數容器
        self.mtxL: Optional[np.ndarray] = None
        self.distL: Optional[np.ndarray] = None
        self.mtxR: Optional[np.ndarray] = None
        self.distR: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None
        self.T: Optional[np.ndarray] = None
        self.P_L: Optional[np.ndarray] = None
        self.P_R: Optional[np.ndarray] = None
        
        # 資料容器
        self.img_points: Dict[str, Dict[str, List[int]]] = {}       # 像素座標
        self.raw_3d_points: Dict[str, List[float]] = {}             # 三角測量後的原始 3D 座標 (左相機座標系)
        self.vicon_3d_points: Dict[str, List[float]] = {}           # VICON Ground Truth
        self.aligned_points: Dict[str, List[float]] = {}            # 經 SVD 轉換後對齊的 3D 座標
    
    @staticmethod
    def _sort_point_keys(keys: List[str]) -> List[str]:
        """將點名稱按數字順序排序"""
        return sorted(keys, key=lambda x: int(x.replace("point", "")))

    @staticmethod
    def imread_unicode(file_path: str) -> np.ndarray:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"無法讀取圖片: {file_path}")
        return img

    def load_parameters(self) -> None:
        """載入單目與雙目參數並建構投影矩陣"""
        print("--- 載入校正參數 ---")

        left_data = np.load(self.left_calib_path, allow_pickle=True)
        self.mtxL = left_data["camera_matrix"]
        self.distL = left_data["dist_coeffs"]
        
        right_data = np.load(self.right_calib_path, allow_pickle=True)
        self.mtxR = right_data["camera_matrix"]
        self.distR = right_data["dist_coeffs"]
        
        stereo_data = np.load(self.stereo_calib_path, allow_pickle=True)
        self.R = stereo_data["R"]
        self.T = stereo_data["T"]
        
        print(f"Baseline: {np.linalg.norm(self.T):.4f} m")

        self.P_L = self.mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P_R = self.mtxR @ np.hstack((self.R, self.T))
        
        print("投影矩陣建構完成。")

    def get_image_points(
        self,
        mode: str = 'json',
        json_path: Optional[str] = None,
        left_img_path: Optional[str] = None,
        right_img_path: Optional[str] = None,
        num_points: int = 15,
        save_path: str = "manual_points.json"
    ) -> None:
        """
        取得圖像座標點（由 JSON 或手動選點）
        """
        if mode == "json":
            if not json_path:
                raise ValueError("json_path 參數為必填")
            self._load_points_from_json(json_path)
        elif mode == "manual":
            if not (left_img_path and right_img_path):
                raise ValueError("left_img_path 和 right_img_path 參數為必填")
            self._manual_select_points(left_img_path, right_img_path, num_points, save_path)
        else:
            raise ValueError(f"不支援的模式: {mode}，請使用 'json' 或 'manual'")
    
    def _load_points_from_json(self, json_path: str) -> None:
        print(f"--- 從 JSON 載入點位: {json_path} ---")
        with open(json_path, "r", encoding="utf-8") as f:
            self.img_points = json.load(f)
        
        sorted_keys = self._sort_point_keys(list(self.img_points.keys()))
        print(f"已載入 {len(self.img_points)} 個點: {', '.join(sorted_keys)}")
    
    def _manual_select_points(
        self,
        left_img_path: str,
        right_img_path: str,
        num_points: int,
        save_path: str
    ) -> None:
        print(f"--- 啟動手動選點模式 (目標: {num_points} 點) ---")
        
        img_L = self.imread_unicode(left_img_path)
        img_R = self.imread_unicode(right_img_path)
        
        pts_L = self._interactive_select(img_L, "Select Left Image", num_points)
        pts_R = self._interactive_select(img_R, "Select Right Image", num_points)
        
        if len(pts_L) != num_points or len(pts_R) != num_points:
            print(f"警告: 選取的點數不足 (左: {len(pts_L)}, 右: {len(pts_R)})，請重新執行。")
            return

        self.img_points = {
            f"point{i+1}": {
                "left": [int(pts_L[i][0]), int(pts_L[i][1])],
                "right": [int(pts_R[i][0]), int(pts_R[i][1])]
            }
            for i in range(num_points)
        }
        
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.img_points, f, indent=4, ensure_ascii=False)
            print(f"選點完成！座標已儲存至: {save_path}")
            print("格式已對齊，下次可直接使用 mode='json' 讀取此檔案。")
        except Exception as e:
            print(f"儲存 JSON 失敗: {e}")

    def _interactive_select(self, img: np.ndarray, win_name: str, num_points: int) -> List[Tuple[int, int]]:
        points: List[Tuple[int, int]] = []
        img_disp = img.copy()
        
        def mouse_cb(event: int, x: int, y: int, flags: int, param) -> None:
            nonlocal img_disp
            
            if event == cv2.EVENT_MOUSEMOVE:
                # 顯示放大鏡效果
                img_disp = img.copy()
                
                # 重繪已選取的點
                for i, pt in enumerate(points):
                    cv2.circle(img_disp, pt, 5, (0, 0, 255), -1)
                    cv2.putText(img_disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 放大鏡參數
                zoom_size = 150  # 放大鏡視窗大小
                zoom_factor = 3  # 放大倍數
                
                # 計算放大區域
                half_region = zoom_size // (2 * zoom_factor)
                y1 = max(0, y - half_region)
                y2 = min(img.shape[0], y + half_region)
                x1 = max(0, x - half_region)
                x2 = min(img.shape[1], x + half_region)
                
                # 提取並放大區域
                roi = img[y1:y2, x1:x2]
                if roi.size > 0:
                    zoomed = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
                    
                    # 在放大區域中心繪製十字線
                    center = zoom_size // 2
                    cv2.line(zoomed, (center, 0), (center, zoom_size), (0, 255, 0), 1)
                    cv2.line(zoomed, (0, center), (zoom_size, center), (0, 255, 0), 1)
                    
                    # 決定放大鏡位置（避免超出邊界）
                    mag_x = x + 20
                    mag_y = y + 20
                    if mag_x + zoom_size > img.shape[1]:
                        mag_x = x - zoom_size - 20
                    if mag_y + zoom_size > img.shape[0]:
                        mag_y = y - zoom_size - 20
                    
                    mag_x = max(0, min(mag_x, img.shape[1] - zoom_size))
                    mag_y = max(0, min(mag_y, img.shape[0] - zoom_size))
                    
                    # 將放大鏡貼到圖像上
                    img_disp[mag_y:mag_y+zoom_size, mag_x:mag_x+zoom_size] = zoomed
                    cv2.rectangle(img_disp, (mag_x, mag_y), 
                                  (mag_x+zoom_size, mag_y+zoom_size), (255, 255, 0), 2)
                
                cv2.imshow(win_name, img_disp)
            
            elif event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
                points.append((x, y))
                img_disp = img.copy()
                for i, pt in enumerate(points):
                    cv2.circle(img_disp, pt, 5, (0, 0, 255), -1)
                    cv2.putText(img_disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(win_name, img_disp)
                print(f"[{win_name}] 已選取第 {len(points)}/{num_points} 點: ({x}, {y})")

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(win_name, mouse_cb)
        
        print(f"\n請在『{win_name}』視窗中依序點選 {num_points} 個對應點。")
        print("按 'ESC' 鍵可提前結束 (若點數未滿將不會儲存)。")
        
        cv2.imshow(win_name, img_disp)
        
        while len(points) < num_points:
            key = cv2.waitKey(10) & 0xFF
            if key == ESC_KEY:
                print("使用者取消選點。")
                break
        
        cv2.destroyWindow(win_name)
        return points

    def triangulate_points(self) -> None:
        """執行線性三角測量，計算左相機座標系下的 3D 座標"""
        print("--- 執行線性三角測量 ---")
        self.raw_3d_points = {}
        sorted_keys = self._sort_point_keys(list(self.img_points.keys()))
        
        for key in sorted_keys:
            pt = self.img_points[key]
            pt_L = np.array([[[pt['left'][0], pt['left'][1]]]], dtype=np.float32)
            pt_R = np.array([[[pt['right'][0], pt['right'][1]]]], dtype=np.float32)
            
            undist_L = cv2.undistortPoints(pt_L, self.mtxL, self.distL, P=self.mtxL)
            undist_R = cv2.undistortPoints(pt_R, self.mtxR, self.distR, P=self.mtxR)
            
            pts_L_2xN = undist_L.reshape(-1, 2).T
            pts_R_2xN = undist_R.reshape(-1, 2).T
            
            points_4d = cv2.triangulatePoints(self.P_L, self.P_R, pts_L_2xN, pts_R_2xN)
            
            coord_3d = (points_4d[:3] / points_4d[3]).flatten()
            self.raw_3d_points[key] = coord_3d.tolist()
            
        print(f"已計算 {len(self.raw_3d_points)} 組原始 3D 座標")

        try:
            with open(RAW_3D_JSON, "w", encoding="utf-8") as f:
                json.dump(self.raw_3d_points, f, indent=4, ensure_ascii=False)
            print(f"原始 3D 座標已輸出至: {RAW_3D_JSON}")
        except Exception as e:
            print(f"輸出原始 3D 座標 JSON 失敗: {e}")

    def load_vicon_data(self, csv_path: str, expected_points: Optional[List[str]] = None) -> None:
        print(f"--- 載入 VICON 資料: {csv_path} ---")
        df = pd.read_csv(csv_path)
        self.vicon_3d_points = {}
        
        point_keys = self._determine_point_keys(expected_points)
        
        loaded_count = 0
        for point_key in point_keys:
            try:
                point_idx = int(point_key.replace("point", "")) - 1
                col_start = 2 + point_idx * 3
                vals = df.iloc[1, col_start:col_start+3].values.astype(float) / VICON_MM_TO_M
                if np.isnan(vals).any():
                    print(f"警告: {point_key} 的 VICON 資料包含 NaN，跳過")
                    continue
                self.vicon_3d_points[point_key] = vals.tolist()
                loaded_count += 1
            except (IndexError, ValueError) as e:
                print(f"警告: 無法讀取 {point_key} 的 VICON 資料: {e}")
                continue
        
        print(f"已載入 {loaded_count} 組 VICON 座標")
        if loaded_count < len(point_keys):
            print(f"注意: 僅成功載入 {loaded_count}/{len(point_keys)} 個點的 VICON 資料")
    
    def _determine_point_keys(self, expected_points: Optional[List[str]]) -> List[str]:
        if expected_points is not None:
            return expected_points
        
        if not hasattr(self, 'img_points') or len(self.img_points) == 0:
            print("警告: img_points 尚未載入，將嘗試讀取所有可能的點（最多 15 個）")
            return [f"point{i+1}" for i in range(15)]
        
        point_keys = self._sort_point_keys(list(self.img_points.keys()))
        return point_keys

    def _rigid_transform_3D(self, A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert A.shape == B.shape
        
        centroid_A = np.mean(A, axis=1, keepdims=True)
        centroid_B = np.mean(B, axis=1, keepdims=True)

        Am = A - centroid_A
        Bm = B - centroid_B
        H = Am @ Bm.T

        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = centroid_B - R @ centroid_A
        return R, t

    def align_coordinates_svd(self, output_path: Optional[str] = None) -> None:
        print("--- 執行 SVD 座標轉換 (Camera -> Vicon) ---")

        common_keys = self._sort_point_keys(
            [k for k in self.raw_3d_points.keys() if k in self.vicon_3d_points]
        )
        
        if len(common_keys) < MIN_POINTS_FOR_SVD:
            raise ValueError(f"共同點過少 ({len(common_keys)} < {MIN_POINTS_FOR_SVD})，無法執行 SVD 轉換")

        A = np.array([self.raw_3d_points[k] for k in common_keys]).T
        B = np.array([self.vicon_3d_points[k] for k in common_keys]).T
        
        R_opt, t_opt = self._rigid_transform_3D(A, B)
        
        print("最佳旋轉矩陣 R:\n", R_opt)
        print("最佳平移向量 t:\n", t_opt.flatten())

        self.aligned_points = {}
        for key, val in self.raw_3d_points.items():
            pt = np.array(val).reshape(3, 1)
            pt_new = R_opt @ pt + t_opt
            self.aligned_points[key] = pt_new.flatten().tolist()
            
        target_path = output_path or ALIGNED_3D_JSON
        try:
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(self.aligned_points, f, indent=4, ensure_ascii=False)
            print(f"座標轉換完成，已儲存至 {target_path}")
        except Exception as e:
            print(f"輸出對齊後 3D 座標 JSON 失敗: {e}")

    def _get_common_keys(self) -> List[str]:
        common_keys = set(self.aligned_points.keys()) & set(self.vicon_3d_points.keys())
        return self._sort_point_keys(list(common_keys))

    def compute_errors(self) -> None:
        print("\n--- 誤差分析 (Aligned vs Vicon) ---")
        errors_3d: List[float] = []
        errors_2d: List[float] = []
        errors_z: List[float] = []
        
        keys = self._get_common_keys()
        
        print(f"{'Point':<8} | {'3D Error (m)':<12} | {'2D(XY) Error (m)':<15} | {'Z Error (m)':<12}")
        print("-" * 65)
        
        for k in keys:
            calc = np.array(self.aligned_points[k])
            gt = np.array(self.vicon_3d_points[k])
            
            e3 = np.linalg.norm(calc - gt)
            e2 = np.linalg.norm(calc[:2] - gt[:2])
            ez = abs(calc[2] - gt[2])
            
            errors_3d.append(e3)
            errors_2d.append(e2)
            errors_z.append(ez)
            print(f"{k:<8} | {e3:.4f}       | {e2:.4f}           | {ez:.4f}")
            
        print("-" * 65)
        print(f"平均 3D 誤差 (RMSE): {np.mean(errors_3d):.4f} m (Std: {np.std(errors_3d):.4f})")
        print(f"平均 2D 誤差 (RMSE): {np.mean(errors_2d):.4f} m (Std: {np.std(errors_2d):.4f})")
        print(f"平均 Z 誤差  (MAE): {np.mean(errors_z):.4f} m (Std: {np.std(errors_z):.4f})")

    def plot_2d_comparison(self) -> None:
        keys = self._get_common_keys()
        
        calc_xy = np.array([self.aligned_points[k][:2] for k in keys])
        gt_xy = np.array([self.vicon_3d_points[k][:2] for k in keys])
        
        plt.figure(figsize=(10, 8))
        plt.scatter(calc_xy[:, 0], calc_xy[:, 1], c='red', marker='o', s=80,
                    label='Calculated (SVD Aligned)')
        plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c='blue', marker='^', s=80, label='Vicon GT')
        
        for i, k in enumerate(keys):
            plt.plot([calc_xy[i, 0], gt_xy[i, 0]], [calc_xy[i, 1], gt_xy[i, 1]],
                     'gray', linestyle='--', alpha=0.5)
            plt.text(calc_xy[i, 0], calc_xy[i, 1], k, fontsize=9, color='darkred')
        
        plt.title("2D (X-Y) Coordinate Comparison after SVD Alignment")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()

        try:
            plt.savefig(ERROR_PLOT_PATH, dpi=300)
            print(f"2D 誤差圖已儲存至: {ERROR_PLOT_PATH}")
        except Exception as e:
            print(f"儲存 2D 誤差圖失敗: {e}")

        plt.show(block=True)

def main() -> None:
    try:
        system = StereoVisionSystem(LEFT_NPZ, RIGHT_NPZ, STEREO_NPZ)
        system.load_parameters()

        if POINTS_MODE == "json":
            system.get_image_points(
                mode="json",
                json_path=POINTS_JSON,
            )
        elif POINTS_MODE == "manual":
            system.get_image_points(
                mode="manual",
                left_img_path=IMG_L,
                right_img_path=IMG_R,
                num_points=NUM_POINTS,
                save_path=MANUAL_POINTS_JSON,
            )
        else:
            raise ValueError(f"不支援的 POINTS_MODE: {POINTS_MODE}")

        system.triangulate_points()
        system.load_vicon_data(VICON_CSV)
        system.align_coordinates_svd()
        system.compute_errors()
        system.plot_2d_comparison()

        plt.close('all')
    except Exception as e:
        print(f"\n程式執行發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')


if __name__ == "__main__":
    main()