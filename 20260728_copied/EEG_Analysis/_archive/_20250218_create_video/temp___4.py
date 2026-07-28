import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from neo.io import BlackrockIO
import tkinter as tk
from tkinter import filedialog
import glob
import os
import json
import pandas as pd
from scipy.signal import cheby1, bessel, butter, filtfilt
from matplotlib.animation import FuncAnimation
from multiprocessing import Pool
from numba import jit



def load_ns3_data(file_path, Ch_dict):
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()
    Ch_list = list(Ch_dict.values())
    raw_signals = [seg.analogsignals[0] for seg in block.segments]
    Ch0_signal = raw_signals[0][:, 0]
    sampling_rate = int(Ch0_signal.sampling_rate.magnitude)
    signal_length = len(Ch0_signal.magnitude.flatten())
    # signals = np.empty((len(Ch_list), signal_length))
    Rec_signals = np.array([raw_signals[0][:, ch[0]].magnitude.flatten() for ch in Ch_list])
    Ref_signals = np.array([raw_signals[0][:, ch[1]].magnitude.flatten() if ch[1] is not None else np.zeros_like(
        raw_signals[0][:, ch[0]].magnitude.flatten()) for ch in Ch_list])
    signals = Rec_signals - Ref_signals

    return signals, sampling_rate

def extract_params(json_dir):
    dlc_json = os.path.join(json_dir, "_analysis_param.json")
    with open(dlc_json, 'r', encoding='utf-8') as file:
        data = json.load(file)
        # dlc_dir = data["DLC"]["DLC_dir"]
        dlc_type = data["DLC"].get("type", None)
        Ch_dict = data["Channels"]
    return dlc_type, Ch_dict

def fast_fft(signal):
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals) ** 2
    return power

def compute_power_spectrum(signal, fs):
    """任意のFFT計算関数（例）"""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals)**2
    freqs = np.fft.fftfreq(n, d=1/fs)
    return freqs, power

def _calc_psd_segment(args):
    """並列化で呼ばれる処理: (seg, fs) -> (freqs, power)"""
    seg, fs = args
    return compute_power_spectrum(seg, fs)

def prepare_psd_in_parallel(signal, fs, window_sec, slide_sec):
    """
    全フレームのPSDを一括で並列計算する関数。
    signal: shape = (n_chan, n_samples)
    """
    n_chan, num_samples = signal.shape
    total_sec = num_samples / fs
    max_frames = int(np.floor((total_sec - window_sec) / slide_sec)) + 1

    # 並列実行するタスクのリストを作成
    # frame × チャンネル分の (seg, fs) をまとめる
    tasks = []
    for frame in range(max_frames):
        start_idx = int(frame * slide_sec * fs)
        end_idx = start_idx + int(window_sec * fs)
        for ch_i in range(n_chan):
            seg = signal[ch_i, start_idx:end_idx]
            tasks.append((seg, fs))

    # multiprocessing を使って一気に計算
    with Pool(processes=8) as p:
        results = p.map(_calc_psd_segment, tasks)
    # results は [(freqs, power), (freqs, power), ... ] のリスト

    # frame & channel の形に戻す
    psd_data = []
    idx = 0
    for frame in range(max_frames):
        frame_data = []
        for ch_i in range(n_chan):
            freqs, power = results[idx]
            frame_data.append((freqs, power))
            idx += 1
        psd_data.append(frame_data)

    return psd_data  # psd_data[frame][ch_i] -> (freqs, power)

def make_multi_channel_animation(filename, Ch_dict,
                                 window_sec=8, slide_sec=1,
                                 save_filename=None):
    # 1) データ読み込み
    signal, fs = load_ns3_data(filename, Ch_dict=Ch_dict)
    psd_data = prepare_psd_in_parallel(signal=signal, fs=fs,
                                       window_sec=window_sec,
                                       slide_sec=slide_sec)

    n_chan, num_samples = signal.shape
    total_sec = num_samples / fs
    max_frames = len(psd_data)
    if max_frames == 0:
        raise ValueError("データ長が短すぎてアニメーションを作成できません。")

    # 2) Figure / Axes 用意
    fig, axes = plt.subplots(2, n_chan, figsize=(4*n_chan, 6), dpi=72)
    fig.tight_layout()

    # チャネルが1つだけの場合の対処
    if n_chan == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    line_raw_list = []
    line_psd_list = []

    # y軸範囲を固定
    global_min = np.min(signal)
    global_max = np.max(signal)

    for i, ch in enumerate(Ch_dict):
        ax_raw = axes[0, i]
        ax_psd = axes[1, i]

        # ----- Raw 波形 -----
        ax_raw.set_xlim(0, window_sec)
        ax_raw.set_ylim(global_min, global_max)
        ax_raw.set_xlabel("Time [s]")
        ax_raw.set_ylabel("Amplitude")
        # タイトルを固定で表示（あるいは削除してもOK）
        ax_raw.set_title(f"Raw (ch {ch})")

        line_raw, = ax_raw.plot([], [], 'b-')
        line_raw_list.append(line_raw)

        # ----- PSD -----
        # PSD全体の最大値を取得し、縦軸を固定
        freqs_all = []
        power_all = []
        for frame_data in psd_data:
            freqs, power = frame_data[i]
            freqs_all.append(freqs)
            power_all.append(power)
        freqs_all = np.concatenate(freqs_all)
        power_all = np.concatenate(power_all)

        ax_psd.set_xlim(0, fs/2)
        ax_psd.set_ylim(0, np.max(power_all) * 1.1)
        ax_psd.set_xlabel("Freq [Hz]")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title(f"Power Spectrum (ch {ch})")

        line_psd, = ax_psd.plot([], [], 'r-')
        line_psd_list.append(line_psd)

    # 3) アニメーション更新
    def init():
        """blit=True の場合に最初に呼ばれる初期化用関数"""
        artists = []
        for line_raw, line_psd in zip(line_raw_list, line_psd_list):
            line_raw.set_data([], [])
            line_psd.set_data([], [])
            artists.append(line_raw)
            artists.append(line_psd)
        return artists

    t_values = np.linspace(0, window_sec, int(window_sec * fs), endpoint=False)

    def update(frame):
        start_time = frame * slide_sec
        start_idx = int(start_time * fs)
        end_idx = start_idx + len(t_values)

        artists = []

        for i, ch in enumerate(Ch_dict):
            # --- Raw 波形 ---
            seg = signal[i, start_idx:end_idx]
            line_raw_list[i].set_data(t_values, seg)
            artists.append(line_raw_list[i])

            # --- PSD ---
            freqs, power = psd_data[frame][i]
            line_psd_list[i].set_data(freqs, power)  # valid フィルタリング不要
            artists.append(line_psd_list[i])

        return artists

    # 4) アニメーション作成
    anim = FuncAnimation(
        fig, update, frames=max_frames, init_func=init,
        blit=True, interval=30
    )

    if save_filename:
        anim.save(save_filename, writer='ffmpeg', fps=10, dpi=72)
        print(f"動画を保存しました: {save_filename}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")
    dlc_type, Ch_dict = extract_params(mouse_dir)
    for exp in glob.glob(os.path.join(mouse_dir, "[!_]*")):
        file_name = glob.glob(os.path.join(exp, "*.ns3"))[0]

        make_multi_channel_animation(
            filename=file_name,
            Ch_dict=Ch_dict,
            window_sec=8,
            slide_sec=1,
            save_filename=os.path.join(exp, "multi_channel_output2.mp4")
        )

