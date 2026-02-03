# 3D Reconstruction Project
---

## 版本區分與選擇

您可以根據需求選擇使用 **「程式版」** 或 **「GUI 版」**：

| 特性 | 程式版 (`3D_Reconstruction_System`) | GUI 版 (`3D_Reconstruction_GUI`) |
| :--- | :--- | :--- |
| **操作方式** | 修改 `config.py` 程式碼 | 滑鼠點選與介面輸入 |
| **路徑設定** | 相對路徑搭配 `BASE_DIR` | **絕對路徑** (隨選隨用) |
| **執行程式碼** | `3D_Reconstruction_System/main.py` | `3D_Reconstruction_GUI/gui_main.py` |

---

## 1. GUI 版 (`3D_Reconstruction_GUI`)

這是為了提高操作靈活性而開發的版本，使用者無需接觸程式碼即可完成所有設定。

### 主要功能
- **絕對路徑支援**：直接選取電腦中任何位置的校正檔 (.npz) 或資料檔 (.json, .csv)。
- **直覺化參數設定**：透過介面切換「JSON 載入」或「手動選點」模式，記得設定選點數量。
- **即時日誌視窗**：原本在終端機 (Console) 顯示的訊息會直接導向至介面下方的文字區域。
- **大尺寸介面**：優化後的佈局支援高解析度螢幕 (1600x1200)，確保路徑完整顯示。

### 執行方式
```powershell
python 3D_Reconstruction_GUI/gui_main.py
```

---

## 2. 程式版 (`3D_Reconstruction_System`)

需要進行二次開發或固定流程的使用者可以使用這個版本內的程式碼

### 主要功能
- **模組化設計**：數學運算 (`core_math`)、資料讀寫 (`data_io`) 與視覺化 (`visualizer`) 等功能。
- **批次處理潛力**：適合整合進其他自動化流程中。
- **手動選點機制**：內建 OpenCV 放大鏡互動介面，支援高精度選點。

### 執行方式
1. 開啟 `3D_Reconstruction_System/config.py` 修改各項路徑參數。
2. 執行主程式：
```powershell
python 3D_Reconstruction_System/main.py
```

---

## 環境安裝

確保 Python 環境已安裝以下必要套件：

```powershell
pip install opencv-python numpy pandas matplotlib
```
也可以執行 `pip install -r requirements.txt` 來安裝所有套件。

> [!TIP]
> 如果執行時遇到 `numpy.core.multiarray` 相關錯誤，請嘗試執行 `pip install --upgrade numpy pandas` 以確保版本相容性。

---

## 專案結構
- `3D_Reconstruction_System/`: 核心運算邏輯與腳本執行環境。
- `3D_Reconstruction_GUI/`: 基於 `tkinter` 開發的圖形化包裹層。
- `requirements.txt`: 專案所需的套件清單。
