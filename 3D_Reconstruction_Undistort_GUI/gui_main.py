"""
gui_main.py — 3D 重建系統 (先去畸變版本) 的 GUI 主程式

使用 tkinter 建構圖形化介面，提供：
- 檔案路徑選擇（相機校正檔、點位檔、圖像檔）
- 模式切換（JSON 載入 / 手動選點）
- 輸出路徑設定
- 即時日誌輸出
- 一鍵執行重建流程
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import threading
import io

from config_bridge import GUIConfigBridge


class ConsoleRedirector(io.StringIO):
    """
    將 stdout/stderr 重新導向至 tkinter Text 元件，
    同時保留實體 console 輸出以便除錯。
    """

    def __init__(self, text_widget):
        """
        初始化重新導向器

        Args:
            text_widget: 目標 tkinter Text 元件
        """
        super().__init__()
        self.text_widget = text_widget

    def write(self, str_val):
        """將文字寫入 Text 元件"""
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, str_val)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        # 同時輸出到實體 console 以便除錯
        sys.__stdout__.write(str_val)

    def flush(self):
        """flush 方法（相容性需求）"""
        pass


class ReconstructionUndistortGUI:
    """3D 重建系統 (先去畸變版本) 的圖形化介面"""

    def __init__(self, root):
        """
        初始化 GUI

        Args:
            root: tkinter 根視窗
        """
        self.root = root
        self.root.title("3D Reconstruction System GUI (先去畸變版本)")
        self.root.geometry("1600x1200")

        # 設定樣式
        style = ttk.Style()
        style.configure("TLabel", font=("Microsoft JhengHei", 10))
        style.configure("TButton", font=("Microsoft JhengHei", 10))
        style.configure(
            "Header.TLabel",
            font=("Microsoft JhengHei", 12, "bold")
        )

        # 初始化參數字典（預設值取自 Undistort 版 config.py）
        self.params = {
            'LEFT_NPZ': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\left\calibration_result.npz"
            ),
            'RIGHT_NPZ': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\right\calibration_result.npz"
            ),
            'STEREO_NPZ': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\stereo\stereo_rt.npz"
            ),
            'POINTS_JSON': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_point_Undistort.json"
            ),
            'VICON_CSV': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\Vicon_point\Vicon_court_01.csv"
            ),
            'RAW_3D_JSON': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_3Dpoints_Undistort.json"
            ),
            'ALIGNED_3D_JSON': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_3Dpoints_svd_Undistort.json"
            ),
            'ERROR_PLOT_PATH': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\xy_error_plot_Undistort.png"
            ),
            'VIDEO_L': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_01_origin_L.mp4"
            ),
            'VIDEO_R': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_01_origin_R.mp4"
            ),
            'MANUAL_POINTS_JSON': (
                r"C:\Users\f1410\Desktop\vicon_chessbord_img"
                r"\20260204_chessboard_img\court_point"
                r"\court\court_point_Undistort.json"
            ),
            'POINTS_MODE': 'manual',
            'NUM_POINTS': '15'
        }

        self.entries = {}
        self._build_ui()

        # 重新導向輸出至 GUI 日誌區
        self.redirector = ConsoleRedirector(self.log_text)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

    def _build_ui(self):
        """建構 GUI 介面"""
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 中間佈局：上下分區
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # ========== 上方：檔案路徑區 ==========
        file_frame = ttk.LabelFrame(
            middle_frame,
            text=" 檔案路徑 (請選擇絕對路徑) ",
            padding="10"
        )
        file_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))

        self._add_path_row(file_frame, "LEFT_NPZ", "左相機校正檔 (.npz)")
        self._add_path_row(file_frame, "RIGHT_NPZ", "右相機校正檔 (.npz)")
        self._add_path_row(file_frame, "STEREO_NPZ", "雙目校正檔 (.npz)")
        self._add_path_row(
            file_frame, "POINTS_JSON", "左右對應 2D點 (.json)"
        )
        self._add_path_row(
            file_frame, "VICON_CSV", "VICON 已標註之 3D點 (.csv)"
        )
        self._add_path_row(
            file_frame, "VIDEO_L", "左影片 (取第1幀去畸變)"
        )
        self._add_path_row(
            file_frame, "VIDEO_R", "右影片 (取第1幀去畸變)"
        )

        # ========== 下方：設定與模式區 ==========
        settings_frame = ttk.LabelFrame(
            middle_frame,
            text=" 設定與模式 ",
            padding="10"
        )
        settings_frame.pack(
            side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 0)
        )

        # 模式切換
        ttk.Label(
            settings_frame,
            text="選點模式 (POINTS_MODE):"
        ).pack(anchor=tk.W)

        self.mode_var = tk.StringVar(value=self.params['POINTS_MODE'])
        mode_frame = ttk.Frame(settings_frame)
        mode_frame.pack(fill=tk.X, pady=2)

        ttk.Radiobutton(
            mode_frame, text="JSON 載入",
            variable=self.mode_var, value="json"
        ).pack(side=tk.LEFT, padx=10)

        ttk.Radiobutton(
            mode_frame, text="手動選點 (在去畸變影像上)",
            variable=self.mode_var, value="manual"
        ).pack(side=tk.LEFT, padx=10)

        # 其他設定欄位
        self._add_entry_row(
            settings_frame, "NUM_POINTS", "手動選點數量:"
        )
        self._add_path_row(
            settings_frame, "MANUAL_POINTS_JSON", "手動選點儲存路徑"
        )
        self._add_path_row(
            settings_frame, "RAW_3D_JSON", "重建 3D 點儲存路徑"
        )
        self._add_path_row(
            settings_frame, "ALIGNED_3D_JSON", "對齊 3D 點儲存路徑"
        )
        self._add_path_row(
            settings_frame, "ERROR_PLOT_PATH", "誤差圖儲存路徑"
        )

        # ========== 底部：執行按鈕與日誌區 ==========
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        btn_run = ttk.Button(
            bottom_frame,
            text=" 開始執行 3D 重建 (先去畸變) ",
            command=self.run_process
        )
        btn_run.pack(pady=5)

        ttk.Label(
            bottom_frame,
            text=" 執行日誌 (Console Output): ",
            style="Header.TLabel"
        ).pack(anchor=tk.W)

        self.log_text = tk.Text(
            bottom_frame, height=20, state='disabled',
            background="#f0f0f0", font=("Consolas", 10)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_path_row(self, parent, key, label_text, is_dir=False):
        """
        新增一列路徑選擇列

        Args:
            parent: 父元件
            key: 參數鍵名
            label_text: 顯示的標籤文字
            is_dir: 是否選擇資料夾（預設為選擇檔案）
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)

        entry = ttk.Entry(frame)
        entry.insert(0, self.params[key])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entries[key] = entry

        btn_text = "瀏覽資料夾" if is_dir else "瀏覽檔案"
        btn = ttk.Button(
            frame, text=btn_text, width=12,
            command=lambda: self._browse_path(key, is_dir)
        )
        btn.pack(side=tk.LEFT)

    def _add_entry_row(self, parent, key, label_text):
        """
        新增一列一般輸入列

        Args:
            parent: 父元件
            key: 參數鍵名
            label_text: 顯示的標籤文字
        """
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)

        entry = ttk.Entry(frame)
        entry.insert(0, self.params[key])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entries[key] = entry

    def _browse_path(self, key, is_dir):
        """
        開啟檔案/資料夾瀏覽對話框

        Args:
            key: 參數鍵名
            is_dir: 是否選擇資料夾
        """
        current_val = self.entries[key].get()
        initial_dir = (
            os.path.dirname(current_val)
            if current_val and os.path.exists(os.path.dirname(current_val))
            else os.getcwd()
        )

        if is_dir:
            path = filedialog.askdirectory(initialdir=initial_dir)
        else:
            path = filedialog.askopenfilename(initialdir=initial_dir)

        if path:
            # 使用絕對路徑
            abs_path = os.path.abspath(path)
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, abs_path)

    def get_full_params(self):
        """
        整合所有輸入參數，直接使用 entry 中的路徑

        Returns:
            dict: 完整的參數字典
        """
        current_params = {k: e.get() for k, e in self.entries.items()}
        current_params['POINTS_MODE'] = self.mode_var.get()
        return current_params

    def run_process(self):
        """使用獨立執行緒啟動重建流程，避免 GUI 凍結"""
        params = self.get_full_params()

        def task():
            try:
                print("\n--- GUI 啟動重建程序 (先去畸變版本) ---")
                bridge = GUIConfigBridge(params)
                bridge.run_reconstruction()
                print("\n--- 程序執行完畢 ---")
                messagebox.showinfo("成功", "3D 重建處理完成！")
            except Exception as e:
                import traceback
                print(f"\n發生錯誤: {str(e)}")
                print(traceback.format_exc())
                messagebox.showerror(
                    "錯誤", f"執行過程中發生錯誤:\n{str(e)}"
                )

        thread = threading.Thread(target=task)
        thread.daemon = True
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ReconstructionUndistortGUI(root)
    root.mainloop()
