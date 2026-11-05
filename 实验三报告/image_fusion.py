"""实验三：SIFT + FLANN + RANSAC 图像拼接，使用 skimage 内置真实摄影测试图。"""
from pathlib import Path
import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '图片'
OUT.mkdir(exist_ok=True)

# 一、读取原始图像，沿竖直方向裁成两张部分重叠的图像
original = data.camera()  # scikit-image 内置的真实摄影灰度图，512x512
original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
h, w = original.shape[:2]
cut_left, cut_right = 340, 172
img_a = original[:, :cut_left].copy()
img_b = original[:, cut_right:].copy()
cv2.imwrite(str(OUT / '01_原始图片.png'), original)
cv2.imwrite(str(OUT / '02_左半部分.png'), img_a)
cv2.imwrite(str(OUT / '03_右半部分.png'), img_b)

# 二、提取 SIFT 关键点及描述符
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY), None)
kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY), None)
if des_a is None or des_b is None:
    raise RuntimeError('SIFT 无法找到足够的特征点')

# 三、FLANN 最近邻匹配，Lowe 比率筛选
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
knn_matches = flann.knnMatch(des_a, des_b, k=2)
good_matches = [m for m, n in knn_matches if m.distance < 0.7 * n.distance]
if len(good_matches) < 4:
    raise RuntimeError('有效匹配点不足 4 个，不能求单应矩阵')

# 四、估计右图到左图的单应矩阵
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
if H is None:
    raise RuntimeError('单应矩阵估计失败')

# 五、展示特征匹配：仅绘制 RANSAC 内点
inlier_matches = [m for m, inside in zip(good_matches, inlier_mask.ravel()) if inside]
match_img = cv2.drawMatches(img_a, kp_a, img_b, kp_b, inlier_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(str(OUT / '04_SIFT特征匹配.png'), match_img)

# 六、根据变换后右图与左图的角点，求出完整画布尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]
corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
transformed = cv2.perspectiveTransform(corners_b, H)
corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
all_corners = np.concatenate([transformed, corners_a], axis=0)
x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
canvas = cv2.warpPerspective(img_b, T @ H, (x_max - x_min, y_max - y_min))
# 与实验指导书一致：用左图覆盖重叠区域
canvas[-y_min:h_a-y_min, -x_min:w_a-x_min] = img_a
cv2.imwrite(str(OUT / '05_拼接结果_完整画布.png'), canvas)

# 七、为了便于报告展示，裁除多余黑边（不会替换未经变换的原始图像）
non_black = np.any(canvas != 0, axis=2)
ys, xs = np.where(non_black)
cropped = canvas[ys.min():ys.max()+1, xs.min():xs.max()+1]
cv2.imwrite(str(OUT / '06_拼接结果_裁边.png'), cropped)

print('原图尺寸:', f'{w}x{h}')
print('左图尺寸:', f'{w_a}x{h_a}')
print('右图尺寸:', f'{w_b}x{h_b}')
print('水平重叠宽度:', cut_left-cut_right)
print('左图 SIFT 特征点:', len(kp_a))
print('右图 SIFT 特征点:', len(kp_b))
print('Lowe 筛选后有效匹配数:', len(good_matches))
print('RANSAC 内点数:', len(inlier_matches))
print('RANSAC 内点率:', f'{len(inlier_matches)/len(good_matches)*100:.2f}%')
print('单应矩阵 H:\n', np.array2string(H, precision=4, suppress_small=True))
print('完整画布尺寸:', f'{canvas.shape[1]}x{canvas.shape[0]}')
print('裁边后尺寸:', f'{cropped.shape[1]}x{cropped.shape[0]}')
print('全部图片已保存至:', OUT)
