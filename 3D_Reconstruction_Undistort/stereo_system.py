import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import config, util, data_io, core_math, visualizer

class StereoVisionSystem:
    """雙目視覺系統 (先去畸變版本)，流程為：先校正影像 -> 選取特徵點 -> 三角測量"""
    
    def __init__(self, left_calib: str, right_calib: str, stereo_calib: str):
        self.calib_paths = (left_calib, right_calib, stereo_calib)
        self.params: Dict[str, np.ndarray] = {}
        self.P_L: Optional[np.ndarray] = None
        self.P_R: Optional[np.ndarray] = None
        
        # 去畸變後的影像
        self.undistorted_img_L: Optional[np.ndarray] = None
        self.undistorted_img_R: Optional[np.ndarray] = None
        
        self.img_points: Dict[str, Dict[str, List[int]]] = {}
        self.raw_3d_points: Dict[str, List[float]] = {}
        self.vicon_3d_points: Dict[str, List[float]] = {}
        self.aligned_points: Dict[str, List[float]] = {}

    def load_parameters(self) -> None:
        """載入參數、建構投影矩陣，並對原始影像進行去畸變處理"""
        print("--- 載入校正參數 ---")
        self.params = data_io.load_calibration_params(*self.calib_paths)
        
        # 讀取左圖以取得影像尺寸
        img_L_raw = util.imread_unicode(config.IMG_L)
        h, w = img_L_raw.shape[:2]
        
        # 1. 計算優化的新相機矩陣 (Alpha=0 表示只保留有效像素，不留黑邊)
        print(f"計算優化的新相機矩陣 (圖像尺寸: {w}x{h})...")
        self.new_mtxL, _ = cv2.getOptimalNewCameraMatrix(
            self.params['mtxL'], self.params['distL'], (w, h), 0, (w, h)
        )
        self.new_mtxR, _ = cv2.getOptimalNewCameraMatrix(
            self.params['mtxR'], self.params['distR'], (w, h), 0, (w, h)
        )
        print("優化的新相機矩陣計算完成。")

        # 2. 建構投影矩陣 P = K @ [R|t]
        # 左相機為世界座標原點 [I|0]
        self.P_L = self.new_mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
        # 右相機相對位置 [R|T]
        self.P_R = self.new_mtxR @ np.hstack((self.params['R'], self.params['T']))
        
        print(f"Baseline: {np.linalg.norm(self.params['T']):.4f} m")
        print("投影矩陣建構完成。")
        
        # 3. 執行影像預去畸變
        self._pre_undistort_images(img_L_raw)
        print("影像去畸變完成。")

    def _pre_undistort_images(self, img_L_raw: Optional[np.ndarray] = None) -> None:
        """根據優化後的新相機矩陣，對原始影像進行完整去畸變處理"""
        print("--- 執行影像預去畸變 ---")
        if img_L_raw is None:
            img_L_raw = util.imread_unicode(config.IMG_L)
        img_R_raw = util.imread_unicode(config.IMG_R)
        
        # 使用優化後的 new_mtxL/R 作為 newCameraMatrix
        # 這樣去畸變後的座標空間與我們建構 P_L/R 時使用的 K 矩陣一致
        self.undistorted_img_L = cv2.undistort(
            img_L_raw, self.params['mtxL'], self.params['distL'], None, self.new_mtxL
        )
        self.undistorted_img_R = cv2.undistort(
            img_R_raw, self.params['mtxR'], self.params['distR'], None, self.new_mtxR
        )


    def set_image_points(self, mode: str, **kwargs) -> None:
        """設定圖像點位"""
        if mode == "json":
            path = kwargs.get("json_path", config.POINTS_JSON)
            self.img_points = data_io.load_points_json(path)
            print(f"已從 JSON 載入 {len(self.img_points)} 個點。")
            print("警告: 若 JSON 內的點是在原始畸變影像上選取的，重建結果將會有誤差。")
        elif mode == "manual":
            num_points = kwargs.get("num_points", config.NUM_POINTS)
            save_path = kwargs.get("save_path", config.MANUAL_POINTS_JSON)
            self.img_points = self._handle_manual_selection(num_points, save_path)
        else:
            raise ValueError(f"不支援的模式: {mode}")

    def _handle_manual_selection(self, num_points: int, save_path: str) -> Dict:
        """在去畸變後的影像上執行手動選點邏輯"""
        print(f"--- 啟動手動選點模式 (目標: {num_points} 點) ---")
        
        if self.undistorted_img_L is None or self.undistorted_img_R is None:
            self._pre_undistort_images()
        
        pts_L = self._interactive_select(self.undistorted_img_L, "Select Left Image (Undistorted)", num_points)
        pts_R = self._interactive_select(self.undistorted_img_R, "Select Right Image (Undistorted)", num_points)
        
        if len(pts_L) != num_points or len(pts_R) != num_points:
            print("警告: 選點未完成。")
            return {}

        points = {
            f"point{i+1}": {
                "left": [int(pts_L[i][0]), int(pts_L[i][1])],
                "right": [int(pts_R[i][0]), int(pts_R[i][1])]
            }
            for i in range(num_points)
        }
        data_io.save_points_json(points, save_path)
        return points

    def _interactive_select(self, img: np.ndarray, win_name: str, num: int) -> List[Tuple[int, int]]:
        points = []
        img_disp = img.copy()

        def mouse_cb(event, x, y, flags, param):
            nonlocal img_disp
            if event == cv2.EVENT_MOUSEMOVE:
                self._draw_magnifier(img, img_disp, points, x, y, win_name)
            elif event == cv2.EVENT_LBUTTONDOWN and len(points) < num:
                points.append((x, y))
                self._redraw_points(img, img_disp, points, win_name)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        cv2.setMouseCallback(win_name, mouse_cb)
        cv2.imshow(win_name, img_disp)
        
        while len(points) < num:
            if cv2.waitKey(10) & 0xFF == config.ESC_KEY: break
        cv2.destroyWindow(win_name)
        return points

    def _draw_magnifier(self, img, img_disp, points, x, y, win_name):
        tmp_disp = img.copy()
        for i, pt in enumerate(points):
            cv2.circle(tmp_disp, pt, 5, (0, 0, 255), -1)
            cv2.putText(tmp_disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        img_disp[:] = tmp_disp[:]

        zoom_size = 150
        zoom_factor = 3
        half_region = zoom_size // (2 * zoom_factor)
        y1, y2 = max(0, y - half_region), min(img.shape[0], y + half_region)
        x1, x2 = max(0, x - half_region), min(img.shape[1], x + half_region)
        
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            zoomed = cv2.resize(roi, (zoom_size, zoom_size), interpolation=cv2.INTER_LINEAR)
            center = zoom_size // 2
            cv2.line(zoomed, (center, 0), (center, zoom_size), (0, 255, 0), 1)
            cv2.line(zoomed, (0, center), (zoom_size, center), (0, 255, 0), 1)
            
            mag_x, mag_y = x + 20, y + 20
            if mag_x + zoom_size > img.shape[1]: mag_x = x - zoom_size - 20
            if mag_y + zoom_size > img.shape[0]: mag_y = y - zoom_size - 20
            
            mag_x, mag_y = max(0, min(mag_x, img.shape[1] - zoom_size)), max(0, min(mag_y, img.shape[0] - zoom_size))
            img_disp[mag_y:mag_y+zoom_size, mag_x:mag_x+zoom_size] = zoomed
            cv2.rectangle(img_disp, (mag_x, mag_y), (mag_x+zoom_size, mag_y+zoom_size), (255, 255, 0), 2)
            
        cv2.imshow(win_name, img_disp)


    def _redraw_points(self, img, img_disp, points, win_name):
        img_disp[:] = img.copy()
        for i, pt in enumerate(points):
            cv2.circle(img_disp, pt, 5, (0, 0, 255), -1)
            cv2.putText(img_disp, f"{i+1}", (pt[0]+5, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win_name, img_disp)

    def process_reconstruction(self, vicon_csv: Optional[str] = None):
        """執行重建與對齊流程"""
        # 1. 三角測量 (傳入 is_undistorted=True，因為點位是在去畸變影像上選取的)
        print("--- 執行三角測量 ---")
        self.raw_3d_points = core_math.triangulate_points(
            self.img_points, 
            self.params['mtxL'], self.params['distL'], self.P_L,
            self.params['mtxR'], self.params['distR'], self.P_R,
            is_undistorted=True
        )
        data_io.save_points_json(self.raw_3d_points, config.RAW_3D_JSON)

        # 2. 座標對齊 (若有 VICON 資料)
        if vicon_csv:
            print("--- 執行座標對齊 ---")
            sorted_keys = util.sort_point_keys(list(self.raw_3d_points.keys()))
            self.vicon_3d_points = data_io.load_vicon_csv(vicon_csv, sorted_keys)
            
            common_keys = util.sort_point_keys([k for k in self.raw_3d_points if k in self.vicon_3d_points])
            if len(common_keys) >= config.MIN_POINTS_FOR_SVD:
                A = np.array([self.raw_3d_points[k] for k in common_keys]).T
                B = np.array([self.vicon_3d_points[k] for k in common_keys]).T
                R, t = core_math.rigid_transform_3D(A, B)
                self.aligned_points = core_math.apply_alignment(self.raw_3d_points, R, t)
                data_io.save_points_json(self.aligned_points, config.ALIGNED_3D_JSON)
                
                # 3. 視覺化
                visualizer.print_error_report(self.aligned_points, self.vicon_3d_points, common_keys)
                visualizer.plot_2d_comparison(self.aligned_points, self.vicon_3d_points, common_keys, config.ERROR_PLOT_PATH)
            else:
                print("共同點不足，跳過 SVD 對齊。")
