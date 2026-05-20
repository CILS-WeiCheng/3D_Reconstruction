# 雙目視覺 3D 重建：學術概念完整說明

> 本文件對應專案 `3D_Reconstruction_Undistort`，逐一說明其 3D 座標建立流程中的學術概念與數學推導。  
> 所有引用均標明來源；若某細節超出程式碼直接涉及的範疇，會明確標示信心程度。

---

## 整體流程概覽

```
原始左右影像
    │
    ▼
① 相機標定：取得 K_L, dist_L, K_R, dist_R, R, T
    │
    ▼
② 計算最優新相機矩陣 K_L_new, K_R_new（getOptimalNewCameraMatrix）
    │
    ▼
③ 影像去畸變（undistort → undistorted_img_L, undistorted_img_R）
    │
    ▼
④ 建立投影矩陣 P_L = K_L_new·[I|0]，P_R = K_R_new·[R|T]
    │
    ▼
⑤ 在去畸變影像上選取對應特徵點
    │
    ▼
⑥ 線性三角測量（cv2.triangulatePoints → 齊次座標 → 除以 w）
    │
    ▼
⑦ SVD 剛體對齊（相機座標系 → 世界/運動捕捉座標系）
    │
    ▼
⑧ 誤差分析與視覺化
```

---

## 1. 單相機針孔模型與內參矩陣

**信心程度：高**

### 1.1 針孔相機模型

相機成像的基礎模型為**針孔相機模型（Pinhole Camera Model）**。  
一個空間中的三維點 $\mathbf{X}_w = [X, Y, Z]^\top$（世界座標）透過以下投影關係映射到影像像素座標 $\mathbf{p} = [u, v]^\top$：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{K} \begin{bmatrix} \mathbf{R} & \mathbf{t} \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

其中 $s$ 是任意非零尺度因子（齊次座標的縮放量）。

### 1.2 內參矩陣 K

**內參矩陣（Camera Intrinsic Matrix / Camera Matrix）** $\mathbf{K}$ 描述相機本身的光學幾何屬性：

$$
\mathbf{K} = \begin{bmatrix} f_x & \gamma & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

各參數含義：

| 符號 | 含義 |
|------|------|
| $f_x, f_y$ | 焦距（以像素為單位），$f_x = f / d_x$，$f_y = f / d_y$（$f$=實體焦距，$d_x, d_y$=像素物理尺寸） |
| $c_x, c_y$ | 主點（Principal Point），即光軸與像平面的交點（理想情況在影像中心） |
| $\gamma$ | 像素傾斜係數（Skew Factor），現代數位相機通常為 0 |

### 1.3 對應程式碼

```python
# data_io.py，載入左右相機內參
"mtxL": left_data["mtxL_opt"],   # 即 K_L，shape (3,3)
"distL": left_data["distL_opt"],
"mtxR": right_data["mtxR_opt"],
"distR": right_data["distR_opt"],
```

### 1.4 標定方法（Calibration）

標定通常使用棋盤格標定板，透過 Zhang (2000) 的平面標定法求解 $\mathbf{K}$。  
OpenCV 的 `cv2.calibrateCamera()` 實作即為此方法。

**關鍵文獻：**
> Zhang, Z. (2000). A flexible new technique for camera calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330–1334. https://doi.org/10.1109/34.888718

---

## 2. 畸變係數與影像去畸變

**信心程度：高**

### 2.1 透鏡畸變的成因

真實鏡頭因製造公差及光學折射，造成影像幾何失真，主要有兩類：

1. **徑向畸變（Radial Distortion）**：影像邊緣向內（桶狀）或向外（枕狀）彎曲，由透鏡曲率不均引起。
2. **切向畸變（Tangential Distortion）**：由鏡頭與感光元件平面不完全平行引起。

### 2.2 畸變數學模型

設理想（無畸變）歸一化相機座標為 $(x', y')$，實際帶有畸變的座標為 $(x'_d, y'_d)$，則：

$$
r^2 = x'^2 + y'^2
$$

**徑向畸變校正：**

$$
x'_d = x'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$
$$
y'_d = y'(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)
$$

**切向畸變校正：**

$$
x'_d \mathrel{+}= 2p_1 x' y' + p_2(r^2 + 2x'^2)
$$
$$
y'_d \mathrel{+}= p_1(r^2 + 2y'^2) + 2p_2 x' y'
$$

畸變係數向量：$\mathbf{dist} = [k_1, k_2, p_1, p_2, k_3]$（OpenCV 預設順序）

### 2.3 去畸變（Undistortion）

去畸變是畸變的**逆過程**：給定帶畸變的像素座標，求其無畸變的理想位置。  
由於上述方程式無封閉解析解，OpenCV 採用迭代法求解反向對應，再透過重映射（Remap）完成整幅影像的去畸變。

**最優新相機矩陣（Optimal New Camera Matrix）：**

```python
# stereo_system.py L36-41
self.new_mtxL, _ = cv2.getOptimalNewCameraMatrix(
    self.params['mtxL'], self.params['distL'], (w, h), 0, (w, h)
)
```

`alpha=0` 表示去畸變後**裁去所有黑邊**，只保留有效像素區域，但新相機矩陣 $\mathbf{K}_{new}$ 的主點與焦距會因此被調整。  
`alpha=1` 則保留所有像素（含邊角黑邊）。

此步驟的數學本質：對原始 $\mathbf{K}$ 做縮放與平移，使去畸變後影像的「有效感興趣區域（ROI）」能充滿整個圖像尺寸。

**實際去畸變：**

```python
# stereo_system.py L66-71
self.undistorted_img_L = cv2.undistort(
    img_L_raw, self.params['mtxL'], self.params['distL'], None, self.new_mtxL
)
```

`cv2.undistort` 的流程：  
1. 對每個輸出像素位置，用 $\mathbf{K}_{new}^{-1}$ 反投影到歸一化平面  
2. 套用畸變模型（前向），找到對應的畸變原始座標  
3. 雙線性插值取色

**關鍵文獻：**
> Bradski, G., & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media. (Chapter 11: Camera Models and Calibration)

> Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (Section 6.1)

---

## 3. 雙相機間的相對外參 R 與 T

**信心程度：高**

### 3.1 外參的幾何意義

雙目標定（Stereo Calibration）求解的 $\mathbf{R}$ 和 $\mathbf{T}$ 描述的是：  
**右相機座標系相對於左相機座標系的剛體變換（Rigid Body Transformation）。**

即：若某點在左相機座標系的座標為 $\mathbf{X}_L$，則其在右相機座標系的座標 $\mathbf{X}_R$ 為：

$$
\mathbf{X}_R = \mathbf{R} \mathbf{X}_L + \mathbf{T}
$$

- $\mathbf{R} \in \mathbb{R}^{3 \times 3}$：旋轉矩陣，滿足 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$，$\det(\mathbf{R}) = +1$（即 $\mathbf{R} \in SO(3)$）
- $\mathbf{T} \in \mathbb{R}^{3 \times 1}$：平移向量，其 $L_2$ 範數即為基線長（Baseline）

```python
# stereo_system.py L50
print(f"Baseline: {np.linalg.norm(self.params['T']):.4f} m")
```

### 3.2 雙目標定的求解方式

雙目標定由以下步驟組成（OpenCV `cv2.stereoCalibrate`）：

1. 分別對左右相機進行單目標定，得到各自的 $\mathbf{K}$、$\mathbf{dist}$
2. 使用同一棋盤格的多幀左右影像，在固定 $\mathbf{K}_L$、$\mathbf{K}_R$ 的前提下，最小化**重投影誤差（Reprojection Error）**：

$$
\min_{\mathbf{R}, \mathbf{T}} \sum_{i} \left\| \mathbf{p}^i_R - \pi_R(\mathbf{R} \mathbf{X}^i + \mathbf{T}) \right\|^2
$$

其中 $\pi_R(\cdot)$ 表示透過右相機 $\mathbf{K}_R$ 的投影操作。

### 3.3 基本矩陣與本質矩陣（補充理論）

雙相機的幾何關係也可由以下矩陣描述（對應點約束）：

- **本質矩陣（Essential Matrix）** $\mathbf{E} = [\mathbf{T}]_\times \mathbf{R}$，僅在歸一化座標下有效
- **基本矩陣（Fundamental Matrix）** $\mathbf{F} = \mathbf{K}_R^{-\top} \mathbf{E} \mathbf{K}_L^{-1}$，在像素座標下有效

這兩個矩陣描述的對極幾何（Epipolar Geometry）是三角測量的理論基礎，但本專案直接使用 $\mathbf{R}$、$\mathbf{T}$ 建構投影矩陣，不需顯式計算 $\mathbf{F}$。

**關鍵文獻：**
> Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (Chapter 9: Epipolar Geometry)

> Bouguet, J.-Y. (2004). Camera calibration toolbox for Matlab. *Caltech Technical Report*. http://www.vision.caltech.edu/bouguetj/calib_doc/

---

## 4. 投影矩陣 P 的建立

**信心程度：高**

### 4.1 定義

**投影矩陣（Projection Matrix）** $\mathbf{P} \in \mathbb{R}^{3 \times 4}$ 將齊次 3D 世界座標直接映射到齊次 2D 像素座標：

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{P} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}, \quad \mathbf{P} = \mathbf{K} [\mathbf{R} | \mathbf{t}]
$$

### 4.2 本專案的建構方式

本專案以**左相機座標系為世界座標原點**（零外參）：

$$
\mathbf{P}_L = \mathbf{K}_{L,new} \cdot [\mathbf{I} | \mathbf{0}] = \mathbf{K}_{L,new} \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$

$$
\mathbf{P}_R = \mathbf{K}_{R,new} \cdot [\mathbf{R} | \mathbf{T}]
$$

```python
# stereo_system.py L46-48
self.P_L = self.new_mtxL @ np.hstack((np.eye(3), np.zeros((3, 1))))
self.P_R = self.new_mtxR @ np.hstack((self.params['R'], self.params['T']))
```

> [!IMPORTANT]
> 此處使用的是 $\mathbf{K}_{new}$（去畸變優化後的新內參），而非原始 $\mathbf{K}$，這是因為去畸變後的影像座標空間對應的是 $\mathbf{K}_{new}$ 定義的投影幾何。若使用原始 $\mathbf{K}$ 建構 $\mathbf{P}$ 但在去畸變影像上選點，將導致三角測量誤差。

### 4.3 展開形式

$$
\mathbf{P}_L = \begin{bmatrix} f_{x,L}' & 0 & c_{x,L}' & 0 \\ 0 & f_{y,L}' & c_{y,L}' & 0 \\ 0 & 0 & 1 & 0 \end{bmatrix}
$$

$$
\mathbf{P}_R = \mathbf{K}_{R,new} \begin{bmatrix} r_{11} & r_{12} & r_{13} & T_x \\ r_{21} & r_{22} & r_{23} & T_y \\ r_{31} & r_{32} & r_{33} & T_z \end{bmatrix}
$$

其中 $f'$、$c'$ 為 $\mathbf{K}_{new}$ 的焦距與主點（可能與原始 $\mathbf{K}$ 的值不同）。

**關鍵文獻：**
> Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (Chapter 6: Camera Models, Section 6.1)

---

## 5. 三角測量（Triangulation）

**信心程度：高**

### 5.1 幾何直覺

三角測量的目標：給定同一個三維點 $\mathbf{X}$ 在左影像的 2D 觀測 $\mathbf{p}_L$ 和在右影像的 2D 觀測 $\mathbf{p}_R$，以及兩相機的投影矩陣 $\mathbf{P}_L$、$\mathbf{P}_R$，求 $\mathbf{X}$。

幾何上，即求兩條空間射線的交點。但由於測量噪聲，兩條射線通常不會嚴格相交，因此需要**最小二乘（Least Squares）**方法。

### 5.2 線性三角測量（DLT - Direct Linear Transform）

對左相機，有：

$$
s_L \begin{bmatrix} u_L \\ v_L \\ 1 \end{bmatrix} = \mathbf{P}_L \tilde{\mathbf{X}}, \quad \tilde{\mathbf{X}} = \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

由此可寫出兩個線性方程（消去 $s_L$）：

$$
u_L (\mathbf{p}_3^{L\top} \tilde{\mathbf{X}}) = \mathbf{p}_1^{L\top} \tilde{\mathbf{X}}
$$
$$
v_L (\mathbf{p}_3^{L\top} \tilde{\mathbf{X}}) = \mathbf{p}_2^{L\top} \tilde{\mathbf{X}}
$$

其中 $\mathbf{p}_i^{L\top}$ 表示 $\mathbf{P}_L$ 的第 $i$ 行。整理為：

$$
\begin{bmatrix}
u_L \mathbf{p}_3^{L\top} - \mathbf{p}_1^{L\top} \\
v_L \mathbf{p}_3^{L\top} - \mathbf{p}_2^{L\top} \\
u_R \mathbf{p}_3^{R\top} - \mathbf{p}_1^{R\top} \\
v_R \mathbf{p}_3^{R\top} - \mathbf{p}_2^{R\top}
\end{bmatrix}
\tilde{\mathbf{X}} = \mathbf{A} \tilde{\mathbf{X}} = \mathbf{0}
$$

這是一個 $4 \times 4$ 的齊次線性方程組。

### 5.3 求解方式

對矩陣 $\mathbf{A}$ 進行 SVD 分解，$\tilde{\mathbf{X}}$ 即為對應最小奇異值的右奇異向量（即 $\mathbf{V}$ 的最後一列）：

$$
\mathbf{A} = \mathbf{U} \Sigma \mathbf{V}^\top, \quad \tilde{\mathbf{X}} = \mathbf{V}_{:, -1}
$$

最後將齊次座標 $[X, Y, Z, W]^\top$ 轉換為非齊次座標：

$$
\mathbf{X}_{3D} = \begin{bmatrix} X/W \\ Y/W \\ Z/W \end{bmatrix}
$$

### 5.4 對應程式碼

```python
# core_math.py L28-43
if not is_undistorted:
    undist_L = cv2.undistortPoints(pt_L_np, mtxL, distL, P=mtxL)
    undist_R = cv2.undistortPoints(pt_R_np, mtxR, distR, P=mtxR)
else:
    undist_L = pt_L_np   # 在去畸變影像上選點，座標直接使用
    undist_R = pt_R_np

pts_L_2x1 = undist_L.reshape(-1, 2).T   # shape: (2, 1)
pts_R_2x1 = undist_R.reshape(-1, 2).T

points_4d = cv2.triangulatePoints(P_L, P_R, pts_L_2x1, pts_R_2x1)
coord_3d = (points_4d[:3] / points_4d[3]).flatten()   # 齊次除以 w
```

> [!NOTE]
> `cv2.triangulatePoints` 要求輸入的 2D 座標在**無畸變**的像素空間（即與 $\mathbf{P}$ 建構時使用的 $\mathbf{K}$ 一致）。本專案已在去畸變影像上選點，因此 `is_undistorted=True` 可直接傳入。

### 5.5 關於「去畸變點」的注意事項

`cv2.undistortPoints(pt, K, dist, P=K)` 的作用：  
1. 先用 $\mathbf{K}^{-1}$ 反投影到歸一化平面  
2. 迭代去除畸變  
3. 用 $\mathbf{P} = \mathbf{K}$ 重新投影回像素座標  

結果等效於「該點在去畸變影像上的位置」，單位仍為像素。

**關鍵文獻：**
> Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (Section 12.2: Linear Triangulation Methods)

> Longuet-Higgins, H. C. (1981). A computer algorithm for reconstructing a scene from two projections. *Nature*, 293, 133–135. https://doi.org/10.1038/293133a0

---

## 6. SVD 剛體對齊（座標系轉換）

**信心程度：高**

### 6.1 問題定義

三角測量的結果在**左相機座標系**下。若要與 Vicon 動作捕捉系統（世界座標系）或標準球場座標比較，需要求解一個剛體變換 $(\mathbf{R}_{svd}, \mathbf{t}_{svd})$，使得：

$$
\mathbf{B} \approx \mathbf{R}_{svd} \mathbf{A} + \mathbf{t}_{svd}
$$

其中 $\mathbf{A}$ 為相機座標系下的 3D 點集，$\mathbf{B}$ 為對應的世界座標系點集（如標準球場或 Vicon 資料）。

這等價於求解**點集配準（Point Set Registration）**問題中的剛體情況。

### 6.2 Umeyama-Arun 演算法（基於 SVD）

給定 $n$ 對對應點 $\{\mathbf{a}_i, \mathbf{b}_i\}$，最小化：

$$
\min_{\mathbf{R}, \mathbf{t}} \sum_{i=1}^{n} \|\mathbf{b}_i - (\mathbf{R} \mathbf{a}_i + \mathbf{t})\|^2
$$

**步驟一：去中心化（Centering）**

$$
\bar{\mathbf{a}} = \frac{1}{n}\sum_i \mathbf{a}_i, \quad \bar{\mathbf{b}} = \frac{1}{n}\sum_i \mathbf{b}_i
$$
$$
\mathbf{a}'_i = \mathbf{a}_i - \bar{\mathbf{a}}, \quad \mathbf{b}'_i = \mathbf{b}_i - \bar{\mathbf{b}}
$$

**步驟二：建構協方差矩陣 H**

$$
\mathbf{H} = \sum_{i=1}^{n} \mathbf{a}'_i \mathbf{b}'^{\top}_i = \mathbf{A}' \mathbf{B}'^{\top} \in \mathbb{R}^{3 \times 3}
$$

（其中 $\mathbf{A}' = [\mathbf{a}'_1, \ldots, \mathbf{a}'_n]$，$\mathbf{B}' = [\mathbf{b}'_1, \ldots, \mathbf{b}'_n]$）

**步驟三：SVD 分解 H**

$$
\mathbf{H} = \mathbf{U} \mathbf{S} \mathbf{V}^\top
$$

**步驟四：求解最優旋轉矩陣 R**

$$
\mathbf{R}_{svd} = \mathbf{V} \mathbf{U}^\top
$$

**步驟五：處理反射（Reflection）**

若 $\det(\mathbf{R}_{svd}) < 0$，代表解為鏡射而非旋轉。修正方法：

$$
\mathbf{V}' = \mathbf{V} \cdot \text{diag}(1, 1, -1), \quad \mathbf{R}_{svd} = \mathbf{V}' \mathbf{U}^\top
$$

（即翻轉 $\mathbf{V}$ 的最後一列的正負號）

**步驟六：求解平移向量 t**

$$
\mathbf{t}_{svd} = \bar{\mathbf{b}} - \mathbf{R}_{svd} \bar{\mathbf{a}}
$$

### 6.3 對應程式碼

```python
# core_math.py L49-68
def rigid_transform_3D(A, B):
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    Am = A - centroid_A   # 去中心化
    Bm = B - centroid_B
    H = Am @ Bm.T         # 協方差矩陣

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T        # 最優旋轉

    if np.linalg.det(R) < 0:   # 處理鏡射
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A  # 最優平移
    return R, t
```

> [!NOTE]
> 此處 `A` 的 shape 為 $(3, n)$（每一列為一個維度，每一行為一個點），`np.mean(A, axis=1)` 計算的是各維度的均值，即重心向量。

### 6.4 應用對齊

```python
# core_math.py L71-78（apply_alignment）
pt_new = R @ pt + t
```

即對每個三角測量所得的 3D 點 $\mathbf{x}$，變換到目標座標系：

$$
\mathbf{x}' = \mathbf{R}_{svd} \mathbf{x} + \mathbf{t}_{svd}
$$

### 6.5 使用條件

- 最少需要 3 個不共線的對應點（`config.py: MIN_POINTS_FOR_SVD = 3`），因為 3 個點決定了 3D 空間中的座標系方向。
- 點越多，解越穩健（最小二乘意義下的最優解）。

**關鍵文獻：**
> Arun, K. S., Huang, T. S., & Blostein, S. D. (1987). Least-squares fitting of two 3-D point sets. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 9(5), 698–700. https://doi.org/10.1109/TPAMI.1987.4767965

> Umeyama, S. (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(4), 376–380. https://doi.org/10.1109/34.88573

> Horn, B. K. P. (1987). Closed-form solution of absolute orientation using unit quaternions. *Journal of the Optical Society of America A*, 4(4), 629–642. https://doi.org/10.1364/JOSAA.4.000629

---

## 7. 補充：getOptimalNewCameraMatrix 的數學意義

**信心程度：中（涉及 OpenCV 內部實現細節）**

### 7.1 目的

去畸變後的影像在邊角會出現黑色無效區域（因為畸變是向內縮）。  
`cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha)` 計算一個新的內參矩陣 $\mathbf{K}_{new}$，使得：

- `alpha=0`：輸出影像的所有像素都有效（裁去黑邊），但視角（Field of View）縮小
- `alpha=1`：保留原始所有像素（含黑邊），視角最大

### 7.2 數學本質

新矩陣 $\mathbf{K}_{new}$ 等效於在原始去畸變影像上疊加一個**仿射裁剪（Affine Crop）**。具體而言，它調整 $f_x', f_y', c_x', c_y'$ 使得去畸變後的影像能充滿目標解析度，同時控制黑邊的保留程度。

此步驟確保了後續三角測量的一致性：
- **投影矩陣用 $\mathbf{K}_{new}$ 建構**
- **去畸變影像的座標空間由 $\mathbf{K}_{new}$ 定義**
- **在去畸變影像上選點 → 直接傳入三角測量 → 一致**

**關鍵文獻：**
> OpenCV Documentation. `cv::getOptimalNewCameraMatrix`. https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7a6c4e032c97f03ba747966e6ad862b1

---

## 8. 完整流程的關鍵一致性檢查

以下是各步驟間必須保持一致的對應關係：

```
K_new (來自 getOptimalNewCameraMatrix)
    ├─── 建構 P_L = K_L_new · [I|0]
    ├─── 建構 P_R = K_R_new · [R|T]
    └─── 用於去畸變 cv2.undistort(..., newCameraMatrix=K_new)
             └─── 在去畸變影像上選點 (u, v)
                      └─── 直接傳入 cv2.triangulatePoints(P_L, P_R, pt_L, pt_R)
                               └─── 輸出左相機座標系下的 3D 點 X_L
                                        └─── SVD 剛體對齊 → 世界座標系 X_world
```

若任一環節的 $\mathbf{K}$ 不一致（例如用原始 $\mathbf{K}$ 建構 $\mathbf{P}$ 但在去畸變影像選點），三角測量結果將產生系統性偏差。

---

## 文獻彙整

| 概念 | 關鍵文獻 |
|------|---------|
| 針孔模型與相機標定 | Zhang (2000), *IEEE TPAMI* |
| 畸變模型 | Bradski & Kaehler (2008), *Learning OpenCV* |
| 多視圖幾何（含外參、投影矩陣、三角測量） | Hartley & Zisserman (2004), *Multiple View Geometry* |
| 三角測量（線性 DLT） | Hartley & Zisserman (2004) §12.2 |
| 對極幾何初始發現 | Longuet-Higgins (1981), *Nature* |
| SVD 點集剛體配準 | Arun et al. (1987), *IEEE TPAMI* |
| SVD 點集配準（含尺度） | Umeyama (1991), *IEEE TPAMI* |
| SVD 點集配準（四元數法） | Horn (1987), *JOSA A* |
| OpenCV 實作細節 | OpenCV 官方文件 https://docs.opencv.org/4.x/ |
