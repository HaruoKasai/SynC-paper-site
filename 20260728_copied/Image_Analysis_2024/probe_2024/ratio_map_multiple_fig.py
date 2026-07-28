import numpy as np
import os
import tkinter as tk
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
from skimage import io
from matplotlib.colors import Normalize
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, binary_dilation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"] = "Arial"

def binning_image(image, binsize=2):
    """
    画像を指定されたbin sizeで平均化する
    bin sizeで平均化できるようにNanパディング　(画像の端を0やNaNで埋める)→ binning

    """
    height, width = image.shape
    pad_height = (bin_size - (height % bin_size)) % bin_size
    pad_width = (bin_size - (width % bin_size)) % bin_size

    return np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=np.nan)

#複数の画像をそれぞれch毎に読み込む
root =tk.Tk()
root.withdraw()
channel1_paths = filedialog.askopenfilenames(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate", title= "select channel 1 Images")
channel2_paths = filedialog.askopenfilenames(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate", title= "select channel 2 Images")


if len(channel1_paths) != len(channel2_paths):
    raise ValueError("チャンネル1とチャンネル2の画像数が一致していません！")

# 画像データをcsvで保存するためにリストを作成
all_ratios =[]
image_data_list =[]

# 画像ごとの比率を計算
for ch1_path, ch2_path in zip(channel1_paths, channel2_paths):
    image_channel1 = io.imread(ch1_path).astype(np.float32) #io.imreadとは画像をnumpy配列(numpy.ndarray)として読み込む
    image_channel2 = io.imread(ch2_path).astype(np.float32)

    if image_channel1.ndim == 3:
        image_channel1 = image_channel1[0, :, :]  # 1番目のスライスを使用
    if image_channel2.ndim == 3:
        image_channel2 = image_channel2[1, :, :]  # 2番目のスライスを使用

    #backを除く
    background_threshold = 1
    safe_image_channel1 = np.where(image_channel1 < background_threshold, np.nan, image_channel1) #np.where(condition, value_if_true, value_if_False)
    safe_image_channel2 = np.where(image_channel2 < 0.0001, np.nan, image_channel2)

    #binarydilation
    binary_mask_channel2 = ~np.isnan(safe_image_channel2) # ~はNOT演算子 ~np.isnanはNaNではない部分
    print("binary_mask_channel2 shape:", binary_mask_channel2.shape)
    dilated_binary_mask = binary_dilation(binary_mask_channel2, structure=np.ones((10,10)))
    dilated_safe_image_channel2 = np.where(dilated_binary_mask, safe_image_channel2, np.nan) #この処理でbinary dataからTrue部分を数値データに戻す

    #binning
    bin_size = 10
    binned_channel1 = binning_image(safe_image_channel1, bin_size)
    binned_channel2 = binning_image(dilated_safe_image_channel2, bin_size)

    #caluculate ratio
    ratio_image = binned_channel2 / binned_channel1
    ratio_image = np.clip(ratio_image, 0, np.percentile(ratio_image, 90.0))

    #append ratio data to list
    y_indices, x_indices = np.where(~np.isnan(ratio_image)) # np.where(condition); conditionがtrueである要素のindex(座標)を取得
    ratios = ratio_image[y_indices, x_indices] # 指定されたピクセルの値を取得する; y:行, x:列

    all_ratios.extend(ratios) # 全てのimageのratio_image valueをall_ratioに追加するために必要;ないとratiosがリスト毎にall_ratiosに追加される
    image_data_list.append((ch1_path, y_indices, x_indices, ratios))#image file path, y座標, x座標, 比率データ (NaNでない)

#global peak value
global_peak_99 = np.percentile(all_ratios, 99.8)
global_peak_95 = np.percentile(all_ratios, 90)
global_max = np.max(all_ratios)

global_peak = min(global_peak_95, global_max)

print(f"Global peak value (99th percentile): {global_peak_99}")
print(f"Global peak value (95th percentile): {global_peak_95}")
print(f"Global max value: {global_max}")
print(f"Selected global peak value: {global_peak}")
print(f"Global peak value (99th percentile): {global_peak}")
print(f"Length of all_ratios: {len(all_ratios)}")
print(f"Min of all_ratios: {np.min(all_ratios)}")
print(f"Max of all_ratios: {np.max(all_ratios)}")
print(f"99th percentile of all_ratios: {np.percentile(all_ratios, 99)}")
print(f"95th percentile of all_ratios: {np.percentile(all_ratios, 95)}")
#color map
colors = [(0.0, "purple"), (0.2, "blue"), (0.4, "skyblue"), (0.6, "green"), (0.8, "yellow"), (1.0, "red")]
custom_cmap = LinearSegmentedColormap.from_list("global_custom_gradient", colors)

# save to csv
csv_dir = os.path.join(os.path.dirname(channel1_paths[0]), "ratio_data") #csvを保存するpathを作る
os.makedirs(csv_dir, exist_ok=True)

for ch1_path, y_indices, x_indices, ratios in image_data_list: # image_data_listに格納されているimageのデータごとにpath, imageのpixel座標, ratioをcsvに書き込む
    normalized_ratios = ratios / global_peak  # クリップ前の値を確認する
    print(f"Before clipping, min: {np.min(normalized_ratios)}, max: {np.max(normalized_ratios)}")

    normalized_ratios = np.clip(normalized_ratios, 0, 1)  # クリップを適用

    print(f"After clipping, min: {np.min(normalized_ratios)}, max: {np.max(normalized_ratios)}")

    df = pd.DataFrame({
        'y' : y_indices,
        'x' : x_indices,
        'ratio': ratios,
        'normalized_ratio': normalized_ratios
    })

    csv_filename = os.path.basename(ch1_path).replace('.tif', '_ratio..csv')
    df.to_csv(os.path.join(csv_dir, csv_filename), index=False)

    print(f"save: {csv_filename}")

#color map 256 step setting ; color mapは通常256階調 (8-bit)に対応しているので0~1の範囲を256段階で均等分割
color_mapping = pd.DataFrame({'base_ratio': np.linspace(0, 1, 256)}) #このratioはratio dataではなく基準値
color_mapping['color'] = [custom_cmap(r) for r in color_mapping['base_ratio']]
color_mapping.to_csv(os.path.join(csv_dir, "color_mapping.csv"), index=False)

# reconstruction image from csv data
csv_files = filedialog.askopenfilenames(title="select ratio csv files", filetypes=[("CSV files", '*.csv')]) #まとめてcsv_fileにアクセスできるように複数選択　→ for loopで一括処理
color_mapping_df = pd.read_csv(os.path.join(csv_dir, "color_mapping.csv"))

#csvの数に応じてsubplotのgridを作成
N = len(csv_files)
rows = cols = int(np.ceil(np.sqrt(N))) # 最小の正方形で作成
pdf_path = os.path.join(csv_dir, "ratio_images.pdf")

with PdfPages(pdf_path) as pdf:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    if N == 1:
        axes = np.array([axes])

    axes = axes.flatten()

    for i, csv_file in enumerate(csv_files):
        df_loaded = pd.read_csv(csv_file)

        image_height = df_loaded["y"].max() + 1 #df_loaded["y"]はcsvの"y"列のデータを取得するためのコード　ここでは列の値+1の数値を取得　+1する理由は座標が0始まりのため+1しないと画像サイズが足りなくなる
        image_width = df_loaded["x"].max() + 1

        reconstructed_ratio = np.full((image_height, image_width), np.nan) #np.fullは指定したNaN配列を作製する　さきに　NaNを埋めておき,数値データは後から代入
        reconstructed_ratio[df_loaded["y"], df_loaded["x"]] = df_loaded["normalized_ratio"]

        ax = axes[i]

        #
        fig, ax = plt.subplots(figsize=(8,8))
        fig.patch.set_facecolor('black')
        ax.axis('off')

        overlay = ax.imshow(reconstructed_ratio, cmap=custom_cmap, norm=Normalize(vmin=0, vmax=1))
        cbar = plt.colorbar (overlay, ax=ax, orientation='horizontal', fraction=0.046, pad=0.04)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)





























