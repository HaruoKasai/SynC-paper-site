import numpy as np
import matplotlib.pyplot as plt
from neo.io import BlackrockIO
import pandas as pd
import os

freq_bin = 1 #Hz #Fourier plot時のbin
time_bin = 10 #sec #heatmap
freq_max, freq_min = 50,0
time_list = [           #sec
    [[0,20], [20, 1000]], #状態A
    [[1000, 2000]],         #状態B  みたいな
    [[2000, 3000]]
]

# ns2ファイルのパス
file_path = r"\\DESKTOP-WS2\data\sawada\raw\Central\20240521_\z146_control_trial001.ns2"
dir = os.path.dirname(file_path)

# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)
    return freqs[:n//2], fft_vals[:n//2]

def spectrum_graph (signal, sampling_rate, time_list, freq_max, freq_min, dir):
    for status in range(len(time_list)):
        signal_status = np.arange(0)
        time_list_status = time_list[status]
        for epoch in range (len(time_list_status)):
            signal_within_epoch = signal[sampling_rate*time_list[status][epoch][0]:sampling_rate*time_list[status][epoch][1]]
            signal_status = np.concatenate((signal_status, signal_within_epoch)) #TODO 厳密には、異なるepochのデータを直接無理やりつなげてからfourier解析するのはよくないだろうが、とりあえず。
        freqs, fft_vals = calculate_spectrum(signal_status, sampling_rate)

        # ビンの数を計算
        num_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
        # ビンの境界を設定
        bins = np.linspace(freq_min, freq_max, num_bins + 1)
        # 各ビンに対応するインデックスを取得
        bin_indices = np.digitize(freqs, bins)
        # 各ビンごとのfft_valsの平均を計算
        fft_vals_means = [fft_vals[bin_indices == i].mean() for i in range(1, num_bins + 1)] / fft_vals.mean() * 100
        # ビンの中心を計算
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.plot(bin_centers, fft_vals_means, marker='.', linestyle='-', color=plt.get_cmap("Accent")(status), label=f'Status {status+1}')
    plt.xlim(freq_min,freq_max)
    plt.xlabel("Hz")
    plt.ylabel("Fourier amplitude (a.u.)")
    plt.legend()
    plt.savefig(os.path.join(dir, "_spectrum.pdf"), dpi=300, transparent=True)
    plt.close()

def heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir):
    df = pd.DataFrame()

    #heatmapのvmaxを適当に決める
    freqs, fft_vals = calculate_spectrum(signal, sampling_rate)
    mean = np.mean(fft_vals)
    std = np.std(fft_vals)
    median = np.median(fft_vals)
    vmax = median+std*3

    time_bin_num = int(len(signal) / sampling_rate / time_bin)
    for t in range(time_bin_num):
        bin_signal = signal[sampling_rate * time_bin * t:sampling_rate * time_bin * (t + 1)]
        freqs, fft_vals = calculate_spectrum(bin_signal, sampling_rate)
        df = df.append(pd.Series(fft_vals), ignore_index=True)

        num_bins = int(np.ceil((freqs.max() - freqs.min()) / freq_bin))
        bins = np.linspace(freqs.min(), freqs.max(), num_bins + 1)
        bin_indices = np.digitize(freqs, bins)
        fft_vals_means = [fft_vals[bin_indices == i].mean() for i in range(1, num_bins + 1)]
        bin_centers = (bins[:-1] + bins[1:]) / 2
    df = df.loc[:, freq_min:freq_max]
    plt.imshow(df.T, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=vmax)

    xticks = np.arange (0, time_bin_num, int(time_bin_num/4))
    record_time = len(signal)/sampling_rate/60 #min
    xtick_labels = np.arange(0,record_time,int(record_time/4))
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(xtick_labels)
    plt.xlabel("min")
    plt.ylabel("Hz")
    plt.savefig(os.path.join(dir, "_heatmap.pdf"), dpi=300, transparent=True)
    plt.close()

# Neoを使用してns2ファイルを読み込む
reader = BlackrockIO(filename=file_path)
block = reader.read_block()

# 生データの取得（RawSignalChannel）
raw_signals = [seg.analogsignals[0] for seg in block.segments]

#EEG (Ch1)
raw_signal = raw_signals[0][:,0]
sampling_rate = int(raw_signal.sampling_rate.magnitude)
signal = raw_signal.magnitude.flatten()
spectrum_graph(signal, sampling_rate, time_list, freq_max, freq_min, dir)
heatmap(signal, sampling_rate, time_bin, freq_max, freq_min, dir)

#EMG (Ch2, CH3)
ch2_signal, ch3_signal = raw_signals[0][:,1], raw_signals[0][:,2]
emg = ch2_signal - ch3_signal
plt.plot(emg)
n = len(emg)
xticks = np.arange (0, n, int(n/4))
record_time = n/sampling_rate/60 #min
xtick_labels = np.arange(0,record_time,int(record_time/4))
plt.gca().set_xticks(xticks)
plt.gca().set_xticklabels(xtick_labels)
plt.xlabel("min")
plt.ylabel("uV")
plt.show()
