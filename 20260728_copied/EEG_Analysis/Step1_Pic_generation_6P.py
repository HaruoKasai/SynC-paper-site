import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, filtfilt
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

file_path = ns3_files[0]  # 1つ目のファイルを処理
output_path = os.path.join(os.path.dirname(file_path), "output.mp4")

# データ読み込み
reader = BlackrockIO(filename=file_path)
block = reader.read_block()
raw_signals = [seg.analogsignals[0] for seg in block.segments]
sampling_rate = int(raw_signals[0].sampling_rate.magnitude)
signal = raw_signals[0][:, 1].magnitude.flatten()
lfp = raw_signals[0][:, 0].magnitude.flatten()
time_points = np.arange(0, len(signal) // sampling_rate, 1)

# フィルタ適用
lowpass_cutoff = 70
filtered_signal = butter_lowpass_filter(signal, lowpass_cutoff, sampling_rate)
filtered_lfp = butter_lowpass_filter(lfp, lowpass_cutoff, sampling_rate)

# 描画設定
fig, ax = plt.subplots(figsize=(10, 5))
line, = ax.plot([], [], lw=1.5, color='red')
ax.set_xlim(0, 8)
ax.set_ylim(-500, 500)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal (uV)")

# フレーム更新関数
def update(frame):
    time_point = time_points[frame]
    start_sample = max(0, int((time_point - 4) * sampling_rate))
    end_sample = min(len(signal), start_sample + 8 * sampling_rate)

    line.set_data(np.linspace(0, 8, end_sample - start_sample), filtered_signal[start_sample:end_sample])
    return line,

# アニメーション設定
ani = animation.FuncAnimation(fig, update, frames=len(time_points), interval=40, blit=True)

# FFmpegで動画として保存
ani.save(output_path, writer='ffmpeg', fps=25, codec='h264_nvenc')
print(f"動画を出力しました: {output_path}")
