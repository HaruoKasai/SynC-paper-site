import numpy as np
import matplotlib.pyplot as plt
from neo.io import BlackrockIO



# ns2ファイルのパス
file_path = r"\\DESKTOP-WS2\data\sawada\raw\Central\20240521_\z146_control_trial001.ns2"

# Neoを使用してns2ファイルを読み込む
reader = BlackrockIO(filename=file_path)
block = reader.read_block()

# 生データの取得（RawSignalChannel）
raw_signals = [seg.analogsignals[0] for seg in block.segments]
print(raw_signals)

# 生データの周波数スペクトルを計算するための関数
def calculate_spectrum(signal, sampling_rate):
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/sampling_rate)
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)

    return freqs[:n//2], fft_vals[:n//2]

# 生データの周波数スペクトルをプロット
# for i in range (raw_signals[0].shape[1]):
for i in range(1):
    print(i)
    raw_signal = raw_signals[0][:,i]
    sampling_rate = raw_signal.sampling_rate.magnitude
    signal = raw_signal.magnitude.flatten()
    print(raw_signal)
    print(signal)
    t = np.arange(len(signal)) / sampling_rate
    print(t)
    freqs, fft_vals = calculate_spectrum(signal, sampling_rate)

    # ビンサイズを設定
    bin_size = 0.5
    # ビンの数を計算
    num_bins = int(np.ceil((freqs.max() - freqs.min()) / bin_size))
    # ビンの境界を設定
    bins = np.linspace(freqs.min(), freqs.max(), num_bins + 1)
    # 各ビンに対応するインデックスを取得
    bin_indices = np.digitize(freqs, bins)
    # 各ビンごとのfft_valsの平均を計算
    fft_vals_means = [fft_vals[bin_indices == i].mean() for i in range(1, num_bins + 1)]
    # ビンの中心を計算
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, fft_vals_means, marker='.', linestyle='-', color='b')
    # plt.plot(freqs, fft_vals, label=f'Channel {i+1}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Frequency Spectrum of Raw Signals')
plt.xlim(0,60)
plt.legend()
plt.show()