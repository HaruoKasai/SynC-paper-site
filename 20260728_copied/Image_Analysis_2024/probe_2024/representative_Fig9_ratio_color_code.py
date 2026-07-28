import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.colors import Normalize
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, binary_dilation
from matplotlib.colors import LinearSegmentedColormap

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"] = "Arial"

# ファイルダイアログを使って画像ファイルを選択
root = tk.Tk()
root.withdraw()
channel1_path = filedialog.askopenfilename(title="Select Channel 1 Image")
channel2_path = filedialog.askopenfilename(title="Select Channel 2 Image")

# 画像の読み込みと型の変換
image_channel1 = io.imread(channel1_path).astype(np.float32)
image_channel2 = io.imread(channel2_path).astype(np.float32)

# 3次元の画像データの場合、最初のスライスのみを選択
if image_channel1.ndim == 3:
    image_channel1 = image_channel1[0, :, :]
if image_channel2.ndim == 3:
    image_channel2 = image_channel2[1, :, :]

# 背景をしきい値で除去したマスク
background_threshold = 10  # 背景と見なす閾値（調整可能）
safe_image_channel1 = np.where(image_channel1 < background_threshold, np.nan, image_channel1)
safe_image_channel2 = np.where(image_channel2 < 0.001, np.nan, image_channel2)

# safe_image_channel2のしきい値で二値化してバイナリダイレーションを適用
binary_mask_channel2 = ~np.isnan(safe_image_channel2)  # NaN以外の領域をTrueとするバイナリマスク
dilated_binary_mask = binary_dilation(binary_mask_channel2, structure=np.ones((7, 7)))

# バイナリダイレーションを適用したマスクを元のデータに適用
dilated_safe_image_channel2 = np.where(dilated_binary_mask, image_channel2, np.nan)

# 比率計算（背景部分はNaN）
ratio_image = dilated_safe_image_channel2 / safe_image_channel1

# 比率画像の異常値を抑えるために上位1%をクリップ
ratio_image = np.clip(ratio_image, 0, np.nanpercentile(ratio_image, 99.9))

# 正規化
normalized_ratio = ratio_image / np.nanmax(ratio_image)

# 背景を黒に設定
fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor('black')
ax.set_title("Overlayed Ratio Image with Contours (Background Masked)")
ax.axis('off')

# カラーマップの指定
colors = [(0.0, "purple"), (0.1, "blue"), (0.2, "skyblue"), (0.4, "green"), (0.6, "yellow"), (0.8, "orange"), (1.0, "red")]
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
overlay = ax.imshow(normalized_ratio, cmap=custom_cmap, norm=Normalize(vmin=0, vmax=1))
cbar = plt.colorbar(overlay, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04, label="iAS / filler ratio")
cbar.ax.tick_params(axis='x', colors='white', labelcolor='white')  # 目盛りとラベルを白色に設定
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.set_label("iAS / filler ratio", color="white")


# 画像の保存
fig.savefig(os.path.join(channel1_path, "..", "ratio_mapping_with_dilation.pdf"), dpi=300, facecolor='black', bbox_inches='tight', transparent=False)
plt.show()
