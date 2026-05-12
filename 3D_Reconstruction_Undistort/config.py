import os

# 目錄與路徑設定 (改成自己的參數資料夾)
BASE_DIR = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point"
PARAM_DIR = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\camera_parm"

# 相機參數檔案
LEFT_NPZ = os.path.join(PARAM_DIR, "./calibration_left.npz")
RIGHT_NPZ = os.path.join(PARAM_DIR, "./calibration_right.npz")
STEREO_NPZ = os.path.join(PARAM_DIR, "./stereo_rt.npz")

# 輸出與資料檔案
POINTS_JSON = os.path.join(BASE_DIR, "court_point_Undistort.json")             # 左右原始圖像對應點
VICON_CSV = os.path.join(BASE_DIR, "./Vicon_point/Vicon_court_01.csv")         # Vicon 原始資料

RAW_3D_JSON = os.path.join(BASE_DIR, "court_3Dpoints_Undistort.json")          # 重建後原始 3D 資料
ALIGNED_3D_JSON = os.path.join(BASE_DIR, "court_3Dpoints_svd_Undistort.json")  # SVD 對齊後的 3D 資料
SVD_RT_NPZ = os.path.join(BASE_DIR, "svd_rt_Undistort.npz")                    # SVD 對齊的旋轉與平移矩陣
ERROR_PLOT_PATH = os.path.join(BASE_DIR, "xy_error_plot_Undistort.png")        # 2D 誤差圖

# 影片輸入設定: 左右影片、儲存的 json 位置
VIDEO_L = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\court\court_01_origin_L.mp4"
VIDEO_R = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\court\court_01_origin_R.mp4"
MANUAL_POINTS_JSON = os.path.join(BASE_DIR, "court_point_Undistort.json")

# 系統常數
ESC_KEY = 27 
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3 # SVD 對齊的最小點數
VICON_MM_TO_M = 1000.0

# 預設模式
POINTS_MODE = "manual"   # 'json' 或 'manual'
NUM_POINTS = 15          # 手動選點模式的點數

# 對齊模式設定
USE_VICON = False              # 是否使用 Vicon 真實資料進行 SVD 對齊
USE_SYNTHETIC_COURT = True     # 是否使用虛擬球場座標進行 SVD 對齊 (若 USE_VICON 為 True，則優先使用 Vicon)

# 視覺化設定
SHOW_PLOT = True               # 是否在執行完畢後彈出誤差圖視窗 (GUI 模式下建議設為 False)

# 定義羽球場標準座標 (單位: 公尺)，用於虛擬球場對齊模式
# STANDARD_COURT_3D = {
#     "point1": [-3.03, -0.76, 0.0],
#     "point2": [-3.03, 0.0, 0.0],
#     "point3": [-3.03, 3.92, 0.0],
#     "point4": [-2.57, -0.76, 0.0],
#     "point5": [-2.57, 0.0, 0.0],
#     "point6": [-2.57, 3.92, 0.0],
#     "point7": [0.0, -0.76, 0.0],
#     "point8": [0.0, 0.0, 0.0],
#     "point9": [0.0, 3.92, 0.0],
#     "point10": [2.57, -0.76, 0.0],
#     "point11": [2.57, 0.0, 0.0],
#     "point12": [2.57, 3.92, 0.0],
#     "point13": [3.03, -0.76, 0.0],
#     "point14": [3.03, 0.0, 0.0],
#     "point15": [3.03, 3.92, 0.0],
# }
STANDARD_COURT_3D = {
    "point1": [-3.03, -0.76, 0.0],
    "point2": [-3.03, 0.0, 0.0],
    # "point3": [-3.03, 3.92, 0.0],         
    "point3": [-2.57, -0.76, 0.0],
    "point4": [-2.57, 0.0, 0.0],
    "point5": [-2.57, 3.92, 0.0],
    "point6": [0.0, -0.76, 0.0],
    "point7": [0.0, 0.0, 0.0],
    "point8": [0.0, 3.92, 0.0],
    "point9": [2.57, -0.76, 0.0],
    "point10": [2.57, 0.0, 0.0],
    "point11": [2.57, 3.92, 0.0],
    "point12": [3.03, -0.76, 0.0],
    "point13": [3.03, 0.0, 0.0],
    "point14": [3.03, 3.92, 0.0],
}
