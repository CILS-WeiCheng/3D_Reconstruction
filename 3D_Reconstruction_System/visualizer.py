import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_2d_comparison(
    aligned_points: Dict[str, List[float]],
    vicon_points: Dict[str, List[float]],
    common_keys: List[str],
    save_path: str = None
) -> None:
    """繪製 2D (X-Y) 座標對比圖"""
    calc_xy = np.array([aligned_points[k][:2] for k in common_keys])
    gt_xy = np.array([vicon_points[k][:2] for k in common_keys])
    
    plt.figure(figsize=(10, 8))
    plt.scatter(calc_xy[:, 0], calc_xy[:, 1], c='red', marker='o', s=80, label='Calculated (SVD Aligned)')
    plt.scatter(gt_xy[:, 0], gt_xy[:, 1], c='blue', marker='^', s=80, label='Vicon GT')
    
    for i, k in enumerate(common_keys):
        plt.plot([calc_xy[i, 0], gt_xy[i, 0]], [calc_xy[i, 1], gt_xy[i, 1]], 'gray', linestyle='--', alpha=0.5)
        plt.text(calc_xy[i, 0], calc_xy[i, 1], k, fontsize=9, color='darkred')
    
    plt.title("2D (X-Y) Coordinate Comparison after SVD Alignment")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show(block=True)

def print_error_report(
    aligned_points: Dict[str, List[float]],
    vicon_points: Dict[str, List[float]],
    common_keys: List[str]
) -> None:
    """輸出誤差分析報告"""
    errors_3d = []
    errors_2d = []
    errors_z = []
    
    print(f"\n{'Point':<8} | {'3D Error (m)':<12} | {'2D(XY) Error (m)':<15} | {'Z Error (m)':<12}")
    print("-" * 65)
    
    for k in common_keys:
        calc = np.array(aligned_points[k])
        gt = np.array(vicon_points[k])
        
        e3 = np.linalg.norm(calc - gt)
        e2 = np.linalg.norm(calc[:2] - gt[:2])
        ez = abs(calc[2] - gt[2])
        
        errors_3d.append(e3)
        errors_2d.append(e2)
        errors_z.append(ez)
        print(f"{k:<8} | {e3:.4f}       | {e2:.4f}           | {ez:.4f}")
        
    print("-" * 65)
    print(f"平均 3D 誤差 (RMSE): {np.mean(errors_3d):.4f} m (Std: {np.std(errors_3d):.4f})")
    print(f"平均 2D 誤差 (RMSE): {np.mean(errors_2d):.4f} m (Std: {np.std(errors_2d):.4f})")
    print(f"平均 Z 誤差  (MAE): {np.mean(errors_z):.4f} m (Std: {np.std(errors_z):.4f})")
