import os

# 目錄與路徑設定 (改成自己的參數資料夾)
BASE_DIR = r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate"
PARAM_DIR = r"./20260114_實驗/標定參數"

# 相機參數檔案
LEFT_NPZ = os.path.join(PARAM_DIR, "20260114_left_single.npz")
RIGHT_NPZ = os.path.join(PARAM_DIR, "20260114_right_single.npz")
STEREO_NPZ = os.path.join(PARAM_DIR, "stereo_rt.npz")

# 輸出與資料檔案
POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_point.json")             # 左右原始圖像對應點
VICON_CSV = os.path.join(BASE_DIR, "20260114_實驗/Vicon_court/court_15points.csv")       # Vicon 原始資料
RAW_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints.json")          # 重建後原始 3D 資料
ALIGNED_3D_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/court_3Dpoints_svd.json")  # SVD 對齊後的 3D 資料
ERROR_PLOT_PATH = os.path.join(BASE_DIR, "20260114_實驗/court/xy_error_plot.png")        # 2D 誤差圖

# 手動選點設定: 左右圖、儲存的 json 位置
IMG_L = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_L.jpg")
IMG_R = os.path.join(BASE_DIR, "20260114_實驗/court/Image/origin_img_R.jpg")
MANUAL_POINTS_JSON = os.path.join(BASE_DIR, "20260114_實驗/court/code_point/court_point.json")

# 系統常數
ESC_KEY = 27 
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3 # 對應點數
VICON_MM_TO_M = 1000.0 

# 預設模式
POINTS_MODE = "json"  # 'json' 或 'manual'
NUM_POINTS = 3
