"""实验二：图像增强。与 input.png 放在同一目录后直接运行。"""
from pathlib import Path
import csv
import json
import platform
import time

import cv2
import numpy as np
import skimage
from skimage.util import random_noise
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parent
FIG = BASE / 'images'
DATA = BASE / 'data'
for folder in (FIG, DATA):
    folder.mkdir(exist_ok=True)
for candidate in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
    if candidate in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = [candidate]
        break
plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False})


def to_u8(image):
    """统一到 0..255 的 uint8，供所有滤波方法公平比较。"""
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def manual_median_filter_color(image, kernel_size=5):
    """逐像素、逐通道计算中值，边界使用复制填充。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image 必须是 H×W×3 彩色图像')
    if image.dtype != np.uint8:
        raise ValueError('image 必须是 uint8')
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError('kernel_size 必须是大于等于 3 的奇数')
    pad = kernel_size // 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    result = np.empty_like(image)
    for c in range(3):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                window = padded[y:y + kernel_size, x:x + kernel_size, c]
                result[y, x, c] = np.median(window)
    return result


def grid(items, shape, filename, title, figsize):
    fig, axes = plt.subplots(*shape, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=17, fontweight='bold')
    for ax, (image, label) in zip(np.asarray(axes).ravel(), items):
        ax.imshow(image, cmap='gray' if image.ndim == 2 else None,
                  vmin=0 if image.ndim == 2 else None,
                  vmax=255 if image.ndim == 2 else None)
        ax.set_title(label, fontsize=12, pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(FIG / filename, dpi=180, facecolor='white')
    plt.close(fig)


def evaluate(reference, image):
    delta = image.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {
        'MSE': mse,
        'PSNR_dB': float(10 * np.log10(255 ** 2 / mse)) if mse else float('inf'),
        'SSIM': float(structural_similarity(reference, image, channel_axis=-1,
                                          data_range=255)),
        'bias': float(np.mean(delta)),
    }


def main():
    # imdecode 兼容 Windows 中文路径；读入后的通道顺序为 BGR。
    raw = np.fromfile(BASE / 'input.png', dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None or min(bgr.shape[:2]) <= 100:
        raise ValueError('input.png 无法读取或尺寸过小')
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pixel_bgr = bgr[100, 100].tolist()
    pixel_rgb = rgb[100, 100].tolist()
    print('Python:', platform.python_version())
    print('OpenCV:', cv2.__version__)
    print('NumPy:', np.__version__)
    print('scikit-image:', skimage.__version__)
    print('Matplotlib:', matplotlib.__version__)
    print('Input shape:', rgb.shape)
    print('BGR[100,100]:', pixel_bgr)
    print('RGB[100,100]:', pixel_rgb)
    print('Gray[100,100]:', int(gray[100, 100]))
    grid([(bgr, '直接用 imshow 显示 BGR（颜色错误）'),
          (rgb, '转换为 RGB 后显示（颜色正确）'),
          (gray, 'BGR 转灰度')], (1, 3), '01_color_spaces.png',
         '实验二：颜色空间转换', (14, 5))

    # 两次使用独立且固定的种子，保证能够复现每一种噪声。
    noisy = {
        '椒盐噪声': to_u8(random_noise(rgb, mode='s&p', amount=0.4, rng=42)),
        '高斯噪声': to_u8(random_noise(rgb, mode='gaussian', mean=0.2,
                                     var=0.03, rng=43)),
    }
    grid([(rgb, '无噪声输入'), (noisy['椒盐噪声'], '椒盐：amount=0.4，seed=42'),
          (noisy['高斯噪声'], '高斯：mean=0.2，var=0.03，seed=43')],
         (1, 3), '02_noise_comparison.png', '实验二：添加噪声', (14, 5))

    rows, filtered, items = [], {}, []
    for noise_name, image in noisy.items():
        filtered[noise_name] = {
            '均值滤波': cv2.blur(image, (5, 5)),
            '中值滤波': cv2.medianBlur(image, 5),
            '高斯滤波': cv2.GaussianBlur(image, (5, 5), 0),
        }
        for method, result in {'未滤波': image, **filtered[noise_name]}.items():
            row = {'noise': noise_name, 'method': method, **evaluate(rgb, result)}
            rows.append(row)
            print(f'{noise_name} | {method}: MSE={row["MSE"]:.4f}, '
                  f'PSNR={row["PSNR_dB"]:.4f} dB, SSIM={row["SSIM"]:.6f}, '
                  f'bias={row["bias"]:.4f}')
            if method != '未滤波':
                items.append((result, f'{noise_name} + {method}\n'
                                      f'PSNR={row["PSNR_dB"]:.2f} dB'))
    grid(items, (2, 3), '03_filter_results.png',
         '实验二：三种 5×5 滤波方法对比', (14, 10))

    start = time.perf_counter()
    manual = manual_median_filter_color(noisy['椒盐噪声'], 5)
    elapsed = time.perf_counter() - start
    opencv = filtered['椒盐噪声']['中值滤波']
    difference = np.abs(manual.astype(np.int16) - opencv.astype(np.int16))
    max_error = int(difference.max())
    differing_values = int(np.count_nonzero(difference))
    assert max_error == 0, '手写算法与 OpenCV 结果不一致'
    print('Manual median max absolute error:', max_error)
    print('Manual median differing channel values:', differing_values)
    print('Manual median time (seconds):', round(elapsed, 4))
    grid([(noisy['椒盐噪声'], '椒盐噪声输入'), (opencv, 'OpenCV 中值滤波'),
          (manual, f'手写中值滤波（最大误差={max_error}）')], (1, 3),
         '04_manual_median.png', '实验二：手写中值滤波与 OpenCV 对照', (14, 5))

    # 补充零均值对照，区分随机扰动和参考参数带来的亮度偏移。
    zero = to_u8(random_noise(rgb, mode='gaussian', mean=0.0, var=0.03, rng=43))
    zero_results = {
        '未滤波': zero,
        '均值滤波': cv2.blur(zero, (5, 5)),
        '中值滤波': cv2.medianBlur(zero, 5),
        '高斯滤波': cv2.GaussianBlur(zero, (5, 5), 0),
    }
    zero_rows = [{'noise': '高斯噪声（零均值补充）', 'method': method,
                  **evaluate(rgb, result)} for method, result in zero_results.items()]
    grid([(rgb, '无噪声输入'), (noisy['高斯噪声'], '参考参数 mean=0.2'),
          (zero, '补充对照 mean=0.0')], (1, 3), '05_gaussian_bias.png',
         '补充观察：高斯噪声均值对亮度的影响', (14, 5))

    with (DATA / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + zero_rows)
    results = {
        'versions': {'Python': platform.python_version(), 'OpenCV': cv2.__version__,
                     'NumPy': np.__version__, 'scikit-image': skimage.__version__,
                     'Matplotlib': matplotlib.__version__},
        'shape': list(rgb.shape), 'pixel_bgr': pixel_bgr, 'pixel_rgb': pixel_rgb,
        'pixel_gray': int(gray[100, 100]), 'rows': rows, 'zero_mean_rows': zero_rows,
        'manual': {'max_absolute_error': max_error,
                   'differing_channel_values': differing_values, 'seconds': elapsed},
    }
    (DATA / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                     encoding='utf-8')
    for name, image in [('rgb', rgb), ('gray', gray), ('salt_pepper', noisy['椒盐噪声']),
                        ('gaussian', noisy['高斯噪声']), ('manual_median', manual)]:
        save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        cv2.imencode('.png', save)[1].tofile(DATA / f'{name}.png')
    print('Experiment completed. Manual/OpenCV equality: PASS')


if __name__ == '__main__':
    main()
