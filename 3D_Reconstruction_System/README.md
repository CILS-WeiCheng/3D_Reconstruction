# 3D Reconstruction System (3D 重建系統)
---

## 目錄結構與程式碼概述

### 1. `main.py`
**主程式進入點**。
- 提供範例功能 `example_stereo_reconstruction()`，展示完整的系統執行流程。
- 包含參數初始化、載入校正檔、選點、執行重建及誤差分析。

### 2. `config.py`
**系統設定檔**。
- 定義所有檔案路徑（如相機參數檔 NPZ、圖像路徑、輸出 JSON 路徑）。
- 設定系統常數（如選點個數 `NUM_POINTS`、SVD 最小點數、單位轉換係數）。
- 提供選點模式開關（手動 `manual` 或從 JSON 讀取 `json`）。

### 3. `stereo_system.py`
**系統核心類別 (`StereoVisionSystem`)**。
- 模組化的核心整合器，協調資料輸入、數學計算與結果輸出。
- 包含**手動選點互動介面**，支援實時放大鏡功能以提高點位選取精確度。
- 管理不同座標系之間的轉換流程。

### 4. `data_io.py`
**資料輸入與輸出模組**。
- 負責載入錄製好的單目與雙目校正參數 (`.npz`)。
- 處理 JSON 檔案的讀寫（如點位座標、3D 結果）。
- 支援讀取 VICON 原始資料 (`.csv`) 並進行單位轉換。
- **自動化功能**：在儲存檔案時會自動建立不存在的目錄。

### 5. `core_math.py`
**核心數學演算法**。
- `triangulate_points`: 執行線性三角測量 (Triangulation)，將 2D 像素點轉換為 3D 原始座標。
- `rigid_transform_3D`: 使用 SVD (Singular Value Decomposition) 計算跨座標系的剛體轉換矩陣（旋轉與平移）。
- `apply_alignment`: 將計算出的轉換矩陣應用到原始點位。

### 6. `visualizer.py`
**資料視覺化與報告**。
- `plot_2d_comparison`: 繪製重建結果與 VICON Ground Truth 的 X-Y 平面對比圖。
- `print_error_report`: 計算並輸出各點的 3D 誤差、2D 誤差與 Z 軸誤差，並統計 RMSE 與平均誤差。

### 7. `util.py`
**輔助工具集**。
- `imread_unicode`: 支援包含中文路徑的圖片讀取功能。
- `sort_point_keys`: 針對點位名稱（如 `point1`, `point10`）進行邏輯排序。

---

## 快速開始

1. **配置環境**：確保已安裝 `opencv-python`, `numpy`, `pandas`, `matplotlib` 等必要套件。
2. **設定路徑**：修改 `config.py` 中的 `BASE_DIR` 與 `PARAM_DIR` 為您電腦上的實際路徑。
3. **執行**：
   ```bash
   python main.py
   ```
