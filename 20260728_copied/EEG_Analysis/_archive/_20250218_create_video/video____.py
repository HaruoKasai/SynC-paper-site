import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, cheby1
from neo.io import BlackrockIO
import os
import tkinter as tk
from tkinter import filedialog
import ffmpeg

def butter_lowpass_filter(data, cutoff, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def cheby1_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4, ripple=0.5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby1(order, ripple, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    mean = np.mean(signal)
    signal -= mean
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[:n // 2], power_spectrum[:n // 2]

# GUIでディレクトリ選択
root = tk.Tk()
root.withdraw()
mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")
if not mouse_dir:
    print("No directory selected.")
    exit()

# .ns3 ファイル検索
ns3_files = []
for root_dir, _, files in os.walk(mouse_dir):
    ns3_files.extend([os.path.join(root_dir, f) for f in files if f.endswith('.ns3')])

if not ns3_files:
    print("No .ns3 files found.")
    exit()

file_path = ns3_files[0]
output_path = os.path.join(os.path.dirname(file_path), "output.mp4")

# データ読み込み
reader = BlackrockIO(filename=file_path)
block = reader.read_block()
raw_signals = [seg.analogsignals[0] for seg in block.segments]
sampling_rate = int(raw_signals[0].sampling_rate.magnitude)
signal = raw_signals[0][:, 1].magnitude.flatten()
lfp = raw_signals[0][:, 0].magnitude.flatten()
m1v1 = (raw_signals[0][:, 0] - raw_signals[0][:, 1]).magnitude.flatten()
time_points = np.arange(0, len(signal) // sampling_rate, 1)

# フィルタ適用
filtered_signal = butter_lowpass_filter(signal, 50, sampling_rate)
filtered_lfp = butter_lowpass_filter(lfp, 50, sampling_rate)
filtered_m1v1 = butter_lowpass_filter(m1v1, 50, sampling_rate)

filtered_signal_9_16 = cheby1_bandpass_filter(signal, 9, 16, sampling_rate)
filtered_lfp_9_16 = cheby1_bandpass_filter(lfp, 9, 16, sampling_rate)
filtered_m1v1_9_16 = cheby1_bandpass_filter(m1v1, 9, 16, sampling_rate)

# FigureとGridSpec設定（元の6×3を維持）
fig = plt.figure(figsize=(30, 20))
gs = gridspec.GridSpec(6, 3, height_ratios=[0.6, 0.4, 0.4, 0.4, 0.4, 0.4])

# 各サブプロット配置
ax_signal = fig.add_subplot(gs[0, 0])
ax_lfp = fig.add_subplot(gs[0, 1])
ax_m1v1 = fig.add_subplot(gs[0, 2])

ax_filtered_9_16 = fig.add_subplot(gs[3, 0])
ax_filtered_lfp_9_16 = fig.add_subplot(gs[3, 1])
ax_filtered_m1v1_9_16 = fig.add_subplot(gs[3, 2])

ax_filtered_0_50 = fig.add_subplot(gs[4, 0])
ax_filtered_lfp_0_50 = fig.add_subplot(gs[4, 1])
ax_filtered_m1v1_0_50 = fig.add_subplot(gs[4, 2])

# 初期プロット設定
lines = {
    "signal": ax_signal.plot([], [], lw=1.5, color='red')[0],
    "lfp": ax_lfp.plot([], [], lw=1.5, color='blue')[0],
    "m1v1": ax_m1v1.plot([], [], lw=1.5, color='green')[0],
    "filtered_9_16": ax_filtered_9_16.plot([], [], lw=1.5, color='purple')[0],
    "filtered_lfp_9_16": ax_filtered_lfp_9_16.plot([], [], lw=1.5, color='purple')[0],
    "filtered_m1v1_9_16": ax_filtered_m1v1_9_16.plot([], [], lw=1.5, color='purple')[0],
    "filtered_0_50": ax_filtered_0_50.plot([], [], lw=1.5, color='orange')[0],
    "filtered_lfp_0_50": ax_filtered_lfp_0_50.plot([], [], lw=1.5, color='orange')[0],
    "filtered_m1v1_0_50": ax_filtered_m1v1_0_50.plot([], [], lw=1.5, color='orange')[0],
}

# フレーム更新関数
def update(frame):
    time_point = time_points[frame]
    start_sample = max(0, int((time_point - 4) * sampling_rate))
    end_sample = min(len(signal), start_sample + 8 * sampling_rate)

    # 信号更新
    lines["signal"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_signal[start_sample:end_sample])
    lines["lfp"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_lfp[start_sample:end_sample])
    lines["m1v1"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_m1v1[start_sample:end_sample])

    lines["filtered_9_16"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_signal_9_16[start_sample:end_sample])
    lines["filtered_lfp_9_16"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_lfp_9_16[start_sample:end_sample])
    lines["filtered_m1v1_9_16"].set_data(np.linspace(0, 8, end_sample - start_sample), filtered_m1v1_9_16[start_sample:end_sample])

    return list(lines.values())

# アニメーション設定
ani = animation.FuncAnimation(fig, update, frames=len(time_points), interval=40, blit=False)

# FFmpegで動画として保存
ani.save(output_path, writer='ffmpeg', fps=25, codec='h264_nvenc')
print(f"動画を出力しました: {output_path}")
