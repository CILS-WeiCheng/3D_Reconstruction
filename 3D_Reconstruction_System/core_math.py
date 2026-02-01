import cv2
import numpy as np
from typing import Dict, List, Tuple

def triangulate_points(
    img_points: Dict[str, Dict[str, List[int]]],
    mtxL: np.ndarray, distL: np.ndarray, P_L: np.ndarray,
    mtxR: np.ndarray, distR: np.ndarray, P_R: np.ndarray
) -> Dict[str, List[float]]:
    """執行線性三角測量，計算原始 3D 座標"""
    raw_3d_points = {}
    
    for key, pt in img_points.items():
        # 轉換為 OpenCV 格式
        pt_L_np = np.array([[[pt['left'][0], pt['left'][1]]]], dtype=np.float32)
        pt_R_np = np.array([[[pt['right'][0], pt['right'][1]]]], dtype=np.float32)
        
        # 去畸變
        undist_L = cv2.undistortPoints(pt_L_np, mtxL, distL, P=mtxL)
        undist_R = cv2.undistortPoints(pt_R_np, mtxR, distR, P=mtxR)
        
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
