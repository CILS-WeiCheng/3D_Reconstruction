import cv2
import numpy as np
from typing import List

def imread_unicode(file_path: str) -> np.ndarray:
    """
    支援中文路徑的圖片讀取
    
    Args:
        file_path: 圖片檔案路徑
        
    Returns:
        np.ndarray: 讀取的圖片內容
        
    Raises:
        FileNotFoundError: 當檔案無法讀取時
    """
    img_array = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"無法讀取圖片: {file_path}")
    return img

def sort_point_keys(keys: List[str]) -> List[str]:
    """
    將點位名稱按數字順序排序 (例如 'point1', 'point10', 'point2' -> 'point1', 'point2', 'point10')
    
    Args:
        keys: 點位名稱列表
        
    Returns:
        List[str]: 排序後的列表
    """
    return sorted(keys, key=lambda x: int(x.replace("point", "")))
