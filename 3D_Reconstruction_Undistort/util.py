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


def extract_frame_from_video(video_path: str) -> np.ndarray:
    """
    從影片中擷取第一幀作為圖像，若第一幀為空則取第二幀

    Args:
        video_path: 影片檔案路徑

    Returns:
        np.ndarray: 擷取的影像幀

    Raises:
        FileNotFoundError: 當影片無法開啟時
        RuntimeError: 當無法擷取有效幀時
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {video_path}")

    # 嘗試讀取第一幀
    ret, frame = cap.read()
    if ret and frame is not None and frame.size > 0:
        cap.release()
        print(f"已從影片擷取第 1 幀: {video_path}")
        return frame

    # 第一幀為空，嘗試讀取第二幀
    print(f"第 1 幀為空，嘗試讀取第 2 幀: {video_path}")
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None and frame.size > 0:
        print(f"已從影片擷取第 2 幀: {video_path}")
        return frame

    raise RuntimeError(f"無法從影片擷取有效幀: {video_path}")
