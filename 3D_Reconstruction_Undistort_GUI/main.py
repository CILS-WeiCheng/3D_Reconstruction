"""
main.py — 3D 重建系統 (先去畸變版本) GUI 的程式進入點

啟動 tkinter GUI 視窗，提供圖形化操作介面。
"""

from gui_main import ReconstructionUndistortGUI
import tkinter as tk


def example_undistort_gui():
    """
    啟動 3D 重建系統 (先去畸變版本) 的 GUI 介面
    """
    print("=== 啟動 3D 重建系統 GUI (先去畸變版本) ===")
    root = tk.Tk()
    app = ReconstructionUndistortGUI(root)
    root.mainloop()


def main():
    """主程式進入點"""
    example_undistort_gui()


if __name__ == "__main__":
    main()
