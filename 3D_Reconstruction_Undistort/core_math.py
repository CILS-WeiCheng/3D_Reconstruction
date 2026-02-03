import cv2
import numpy as np
from typing import Dict, List, Tuple

def triangulate_points(
    img_points: Dict[str, Dict[str, List[int]]],
    mtxL: np.ndarray, distL: np.ndarray, P_L: np.ndarray,
    mtxR: np.ndarray, distR: np.ndarray, P_R: np.ndarray,
    is_undistorted: bool = False
) -> Dict[str, List[float]]:
    """
    執行線性三角測量，計算原始 3D 座標
    
    Args:
        img_points: 左右影像的點位座標
        mtxL, distL, P_L: 左相機的內參、畸變係數與投影矩陣
        mtxR, distR, P_R: 右相機的內參、畸變係數與投影矩陣
        is_undistorted: 傳入的點位是否已經是去畸變後的座標。
                       如果是 True，則跳過 cv2.undistortPoints。
    """
    raw_3d_points = {}
    
    for key, pt in img_points.items():
        # 轉換為 OpenCV 格式
        pt_L_np = np.array([[[pt['left'][0], pt['left'][1]]]], dtype=np.float32)
        pt_R_np = np.array([[[pt['right'][0], pt['right'][1]]]], dtype=np.float32)
        
        if not is_undistorted:
            # 原始影像上的點，需要進行去畸變
            undist_L = cv2.undistortPoints(pt_L_np, mtxL, distL, P=mtxL)
            undist_R = cv2.undistortPoints(pt_R_np, mtxR, distR, P=mtxR)
        else:
            # 已經是在去畸變影像上選點，直接使用 (但需確保符合三角測量所需的尺度)
            # 注意：如果是在 cv2.undistort 後的影像上選點，座標已經是相對於 P=mtxL 的
            undist_L = pt_L_np
            undist_R = pt_R_np
        
        # 三角測量 (注意 P_L, P_R 的形狀與 cv2.triangulatePoints 的要求)
        pts_L_2x1 = undist_L.reshape(-1, 2).T
        pts_R_2x1 = undist_R.reshape(-1, 2).T
        
        points_4d = cv2.triangulatePoints(P_L, P_R, pts_L_2x1, pts_R_2x1)
        coord_3d = (points_4d[:3] / points_4d[3]).flatten()
        
        raw_3d_points[key] = coord_3d.tolist()
        
    return raw_3d_points

def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """計算最佳的剛體轉換 (Rotation & Translation)"""
    assert A.shape == B.shape
    
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ Bm.T

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 處理特殊情況：鏡射
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t

def apply_alignment(raw_points: Dict[str, List[float]], R: np.ndarray, t: np.ndarray) -> Dict[str, List[float]]:
    """應用座標對齊轉換"""
    aligned_points = {}
    for key, val in raw_points.items():
        pt = np.array(val).reshape(3, 1)
        pt_new = R @ pt + t
        aligned_points[key] = pt_new.flatten().tolist()
    return aligned_points
