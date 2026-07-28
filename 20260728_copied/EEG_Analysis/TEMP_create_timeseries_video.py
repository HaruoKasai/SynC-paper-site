#TODO
"""
他のsubplot追加
X軸を各フレームに対応する秒に変更
ちゃんとグラフタイトルつける
etc

暫定方針
1Hz以上の動画（特にマウス動画と合わせた状態で）をすべてのデータで作成するのは時間がかかりすぎそう。
時間を限定してつくる


"""


import numpy as np
import matplotlib.pyplot as plt
from neo.io import BlackrockIO
import os
import json
import glob
from scipy.signal import cheby1, bessel, butter, filtfilt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool


def load_ns3_data(file_path, Ch_dict):
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()
    Ch_list = list(Ch_dict.values())
    raw_signals = [seg.analogsignals[0] for seg in block.segments]
    Ch0_signal = raw_signals[0][:, 0]
    sampling_rate = int(Ch0_signal.sampling_rate.magnitude)

    Rec_signals = np.array([raw_signals[0][:, ch[0]].magnitude.flatten() for ch in Ch_list])
    Ref_signals = np.array([raw_signals[0][:, ch[1]].magnitude.flatten() if ch[1] is not None else np.zeros_like(
        raw_signals[0][:, ch[0]].magnitude.flatten()) for ch in Ch_list])
    signals = Rec_signals - Ref_signals

    return signals, sampling_rate

def extract_params(json_dir):
    dlc_json = os.path.join(json_dir, "_analysis_param.json")
    with open(dlc_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
        Ch_dict = data["Channels"]
    return Ch_dict

def compute_power_spectrum(signal, fs):
    """FFT 計算関数"""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.fftfreq(n, d=1/fs)
    return freqs, power


def _calc_psd_segment(args):
    """並列処理用"""
    seg, fs = args
    return compute_power_spectrum(seg, fs)


def prepare_psd_in_parallel(signal, fs, window_sec, slide_sec):
    """全フレームのPSDを並列計算"""
    n_chan, num_samples = signal.shape
    total_sec = num_samples / fs
    max_frames = int(np.floor((total_sec - window_sec) / slide_sec)) + 1

    tasks = []
    for frame in range(max_frames):
        start_idx = int(frame * slide_sec * fs)
        end_idx = start_idx + int(window_sec * fs)
        for ch_i in range(n_chan):
            seg = signal[ch_i, start_idx:end_idx]
            tasks.append((seg, fs))

    with Pool(processes=4) as p:
        results = p.map(_calc_psd_segment, tasks)

    psd_data = []
    idx = 0
    for frame in range(max_frames):
        frame_data = []
        for ch_i in range(n_chan):
            frame_data.append(results[idx])
            idx += 1
        psd_data.append(frame_data)

    return psd_data


def make_multi_channel_animation(filename, Ch_dict, window_sec=8, slide_sec=8, save_filename=None, frame_save_dir=None):
    signal, fs = load_ns3_data(filename, Ch_dict=Ch_dict)
    psd_data = prepare_psd_in_parallel(signal, fs, window_sec, slide_sec)

    n_chan, num_samples = signal.shape
    max_frames = len(psd_data)
    if max_frames == 0:
        raise ValueError("データ長が短すぎてアニメーションを作成できません。")

    fig, axes = plt.subplots(2, n_chan, figsize=(4*n_chan, 6), dpi=72)
    fig.tight_layout()

    if n_chan == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    line_raw_list = []
    line_psd_list = []
    global_min, global_max = np.min(signal), np.max(signal)

    # 事前に時間軸を作成
    t_values = np.linspace(0, window_sec, int(window_sec * fs), endpoint=False)

    for i, ch in enumerate(Ch_dict):
        ax_raw = axes[0, i]
        ax_psd = axes[1, i]

        ax_raw.set_xlim(0, window_sec)
        ax_raw.set_ylim(global_min, global_max)
        ax_raw.set_xlabel("Time [s]")
        ax_raw.set_ylabel("Amplitude")
        ax_raw.set_title(f"Raw (ch {ch})")

        line_raw, = ax_raw.plot([], [], 'b-')
        line_raw_list.append(line_raw)

        ax_psd.set_xlim(0, fs/2)
        ax_psd.set_ylim(0, np.max([np.max(power) for freqs, power in psd_data[0]]) * 1.1)
        ax_psd.set_xlabel("Freq [Hz]")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title(f"Power Spectrum (ch {ch})")

        line_psd, = ax_psd.plot([], [], 'r-')
        line_psd_list.append(line_psd)

    def init():
        """アニメーションの初期化"""
        artists = []
        for line_raw, line_psd in zip(line_raw_list, line_psd_list):
            line_raw.set_data([], [])
            line_psd.set_data([], [])
            artists.append(line_raw)
            artists.append(line_psd)
        return artists

    def update(frame):
        """フレームごとに更新し、各フレームを保存"""
        print(frame)
        start_idx = int(frame * slide_sec * fs)
        end_idx = start_idx + len(t_values)

        artists = []
        for i in range(n_chan):
            line_raw_list[i].set_data(t_values, signal[i, start_idx:end_idx])
            artists.append(line_raw_list[i])

            freqs, power = psd_data[frame][i]
            valid = (freqs >= 0) & (freqs <= fs/2)
            line_psd_list[i].set_data(freqs[valid], power[valid])
            artists.append(line_psd_list[i])

        # 各フレームを保存
        if frame_save_dir:
            os.makedirs(frame_save_dir, exist_ok=True)
            plt.savefig(os.path.join(frame_save_dir, f"frame_{frame:04d}.png"), dpi=72)

        return artists

    anim = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=30)

    if save_filename:
        anim.save(save_filename, writer='ffmpeg', fps=15, dpi=72)
        print(f"動画を保存しました: {save_filename}")

    plt.close(fig)


if __name__ == "__main__":
    mouse_dir = input("Enter the 'data' directory path: ")
    Ch_dict = extract_params(mouse_dir)

    for exp in glob.glob(os.path.join(mouse_dir, "[!_]*")):
        file_name = glob.glob(os.path.join(exp, "*.ns3"))[0]
        frame_dir = os.path.join(exp, "_frames")  # 各フレームの保存先

        make_multi_channel_animation(
            filename=file_name,
            Ch_dict=Ch_dict,
            window_sec=8,
            slide_sec=8,
            save_filename=os.path.join(exp, "multi_channel_output2.mp4"),
            frame_save_dir=frame_dir  # 追加
        )
