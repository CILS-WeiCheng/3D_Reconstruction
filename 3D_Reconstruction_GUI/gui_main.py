import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os
import threading
import io

# 確保可以匯入 bridge
from config_bridge import GUIConfigBridge

class ConsoleRedirector(io.StringIO):
    """
    將 stdout/stderr 重新導向至 tkinter Text 元件
    """
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, str_val):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, str_val)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        # 同時輸出到實體 console 以便除錯
        sys.__stdout__.write(str_val)

    def flush(self):
        pass

class ReconstructionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Reconstruction System GUI")
        self.root.geometry("1600x1200")
        
        # 設定樣式
        style = ttk.Style()
        style.configure("TLabel", font=("Microsoft JhengHei", 10))
        style.configure("TButton", font=("Microsoft JhengHei", 10))
        style.configure("Header.TLabel", font=("Microsoft JhengHei", 12, "bold"))

        # 初始化參數字典 (預設值參考原 config.py，轉換為絕對路徑示例)
        self.params = {
            'LEFT_NPZ': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_雙目參數檔\exp1\20251125_left_single_1_2.npz",
            'RIGHT_NPZ': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_雙目參數檔\exp1\20251125_right_single_1_2.npz",
            'STEREO_NPZ': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_雙目參數檔\exp1\stereo_rt_result_1_2.npz",
            'POINTS_JSON': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\court\court_point.json",
            'VICON_CSV': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\Vicon_court\court_15points.csv",
            'RAW_3D_JSON': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\court\court_3Dpoints.json",
            'ALIGNED_3D_JSON': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\court\court_3Dpoints_svd.json",
            'ERROR_PLOT_PATH': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\court\xy_error_plot.png",
            'IMG_L': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_校正後圖片\court\origin_L.jpg",
            'IMG_R': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_exp1_校正後圖片\court\origin_R.jpg",
            'MANUAL_POINTS_JSON': r"C:\Users\f1410\Desktop\defensive_gap-develop@calibration\3D_coordinate\20251125_實驗\court\code_point\court_point.json",
            'POINTS_MODE': 'json',
            'NUM_POINTS': '15'
        }

        self.entries = {}
        self._build_ui()
        
        # 重新導向輸出
        self.redirector = ConsoleRedirector(self.log_text)
        sys.stdout = self.redirector
        sys.stderr = self.redirector

    def _build_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 中間佈局：改為一上一下
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 上方：校正與資料檔案
        left_col = ttk.LabelFrame(middle_frame, text=" 檔案路徑 (請選擇絕對路徑) ", padding="10")
        left_col.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))
        
        self._add_path_row(left_col, "LEFT_NPZ", "左相機校正檔 (.npz)")
        self._add_path_row(left_col, "RIGHT_NPZ", "右相機校正檔 (.npz)")
        self._add_path_row(left_col, "STEREO_NPZ", "雙目校正檔 (.npz)")
        self._add_path_row(left_col, "POINTS_JSON", "左右對應 2D點 (.json)")
        self._add_path_row(left_col, "VICON_CSV", "VICON 已標註之 3D點 (.csv)")
        self._add_path_row(left_col, "IMG_L", "左原始圖像 (手動選點用)")
        self._add_path_row(left_col, "IMG_R", "右原始圖像 (手動選點用)")

        # 下方：參數設定與模式
        right_col = ttk.LabelFrame(middle_frame, text=" 設定與模式 ", padding="10")
        right_col.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(5, 0))

        # 模式切換
        ttk.Label(right_col, text="選點模式 (POINTS_MODE):").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value=self.params['POINTS_MODE'])
        mode_frame = ttk.Frame(right_col)
        mode_frame.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(mode_frame, text="JSON 載入", variable=self.mode_var, value="json").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="手動選點", variable=self.mode_var, value="manual").pack(side=tk.LEFT, padx=10)

        self._add_entry_row(right_col, "NUM_POINTS", "手動選點數量:")
        self._add_path_row(right_col, "MANUAL_POINTS_JSON", "手動選點儲存路徑")
        self._add_path_row(right_col, "RAW_3D_JSON", "重建 3D 點儲存路徑")
        self._add_path_row(right_col, "ALIGNED_3D_JSON", "對齊 3D 點儲存路徑")
        self._add_path_row(right_col, "ERROR_PLOT_PATH", "誤差圖儲存路徑")

        # 底部：按鈕與日誌
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        btn_run = ttk.Button(bottom_frame, text=" 開始執行 3D 重建 ", command=self.run_process)
        btn_run.pack(pady=5)

        ttk.Label(bottom_frame, text=" 執行日誌 (Console Output): ", style="Header.TLabel").pack(anchor=tk.W)
        self.log_text = tk.Text(bottom_frame, height=20, state='disabled', background="#f0f0f0", font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _add_path_row(self, parent, key, label_text, is_dir=False):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        
        entry = ttk.Entry(frame)
        entry.insert(0, self.params[key])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entries[key] = entry
        
        btn_text = "瀏覽資料夾" if is_dir else "瀏覽檔案"
        btn = ttk.Button(frame, text=btn_text, width=12, 
                         command=lambda: self._browse_path(key, is_dir))
        btn.pack(side=tk.LEFT)

    def _add_entry_row(self, parent, key, label_text):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label_text, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(frame)
        entry.insert(0, self.params[key])
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.entries[key] = entry

    def _browse_path(self, key, is_dir):
        current_val = self.entries[key].get()
        initial_dir = os.path.dirname(current_val) if current_val and os.path.exists(os.path.dirname(current_val)) else os.getcwd()
        
        if is_dir:
            path = filedialog.askdirectory(initialdir=initial_dir)
        else:
            path = filedialog.askopenfilename(initialdir=initial_dir)
        
        if path:
            # 直接使用絕對路徑
            abs_path = os.path.abspath(path)
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, abs_path)

    def get_full_params(self):
        """整合所有輸入參數，直接使用 entry 中的路徑"""
        current_params = {k: e.get() for k, e in self.entries.items()}
        current_params['POINTS_MODE'] = self.mode_var.get()
        return current_params

    def run_process(self):
        # 使用執行緒避免 GUI 凍結
        params = self.get_full_params()
        
        def task():
            try:
                print("\n--- GUI 啟動重建程序 ---")
                bridge = GUIConfigBridge(params)
                bridge.run_reconstruction()
                print("\n--- 程序執行完畢 ---")
                messagebox.showinfo("成功", "3D 重建處理完成！")
            except Exception as e:
                import traceback
                print(f"\n發生錯誤: {str(e)}")
                print(traceback.format_exc())
                messagebox.showerror("錯誤", f"執行過程中發生錯誤:\n{str(e)}")

        thread = threading.Thread(target=task)
        thread.daemon = True
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = ReconstructionGUI(root)
    root.mainloop()
