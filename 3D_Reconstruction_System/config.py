import os

# 目錄與路徑設定
BASE_DIR = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate"
PARAM_DIR = r"./20260114_實驗/標定參數"

# 相機參數檔案
LEFT_NPZ = os.path.join(PARAM_DIR, "20260114_left_single.npz")
RIGHT_NPZ = os.path.join(PARAM_DIR, "20260114_right_single.npz")
STEREO_NPZ = os.path.join(PARAM_DIR, "stereo_rt.npz")

# 輸出與資料檔案
POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_point.json")
VICON_CSV = os.path.join(BASE_DIR, "20260114_實驗/Vicon_court/court_15points.csv")
RAW_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints.json")
ALIGNED_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints_svd.json")
ERROR_PLOT_PATH = os.path.join(BASE_DIR, "20260114_實驗/court/xy_error_plot.png")

# 手動選點設定
IMG_L = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_L.jpg")
IMG_R = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_R.jpg")
MANUAL_POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/code_point/court_point.json")

# 系統常數
ESC_KEY = 27
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3
VICON_MM_TO_M = 1000.0

# 預設模式
POINTS_MODE = "json"  # 'json' 或 'manual'
NUM_POINTS = 3
