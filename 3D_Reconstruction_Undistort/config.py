import os

# 目錄與路徑設定 (改成自己的參數資料夾)
BASE_DIR = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\court"
PARAM_DIR = r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point"

# 相機參數檔案
LEFT_NPZ = os.path.join(PARAM_DIR, "./left/calibration_result.npz")
RIGHT_NPZ = os.path.join(PARAM_DIR, "./right/calibration_result.npz")
STEREO_NPZ = os.path.join(PARAM_DIR, "./stereo/stereo_rt.npz")

# 輸出與資料檔案
POINTS_JSON = os.path.join(BASE_DIR, "court_point.json") # 左右原始圖像對應點
VICON_CSV = os.path.join(BASE_DIR, "./Vicon_pointVicon_court_01.csv")       # Vicon 原始資料

RAW_3D_JSON = os.path.join(BASE_DIR, "court_3Dpoints_Undistort.json")          # 重建後原始 3D 資料
ALIGNED_3D_JSON = os.path.join(BASE_DIR, "court_3Dpoints_svd_Undistort.json")  # SVD 對齊後的 3D 資料
ERROR_PLOT_PATH = os.path.join(BASE_DIR, "xy_error_plot_Undistort.png")        # 2D 誤差圖

# 手動選點設定: 左右圖、儲存的 json 位置
IMG_L =r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\court\court_01_origin_L.jpg"
IMG_R =r"C:\Users\f1410\Desktop\vicon_chessbord_img\20260204_chessboard_img\court_point\court\court_01_origin_R.jpg"
MANUAL_POINTS_JSON = os.path.join(BASE_DIR, "court_point_Undistort.json")

# 系統常數
ESC_KEY = 27 
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
MIN_POINTS_FOR_SVD = 3 # SVD對齊的最小點數
VICON_MM_TO_M = 1000.0 

# 預設模式
POINTS_MODE = "manual"   # 'json' 或 'manual'
NUM_POINTS = 15          # 手動選點模式的點數

