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
    n = len(signal)
    power = fast_fft(signal)
    freqs = np.fft.fftfreq(n, d=1 / fs)
    return freqs, power


def make_multi_channel_animation(filename, Ch_dict,
                                 window_sec=8, slide_sec=1,
                                 save_filename=None):
    # --- 1) データ読み込み ---
    signal, fs = load_ns3_data(filename, Ch_dict=Ch_dict)
    n_chan, num_samples = signal.shape
    total_sec = num_samples / fs

    # アニメーションのフレーム数を決定
    max_frames = int(np.floor((total_sec - window_sec) / slide_sec)) + 1
    if max_frames < 1:
        raise ValueError("データ長が短すぎてアニメーションを作成できません。")

    # --- 2) Figure / Axes の用意 ---
    #    行方向にチャネル数分、列方向に2つ(左: Raw、右: PSD) の配置
    fig, axes = plt.subplots(4, n_chan, figsize=(7, 3 * n_chan))
    plt.tight_layout()

    # チャネルが1つだけの場合、axesが2次元ではなく1次元になる可能性があるため対処
    if n_chan == 1:
        axes = np.array([axes])  # shapeを(1,2)にそろえる

    # 各チャネル用に、Raw波形とPSD用の"描画ライン"を保存するリスト
    line_raw_list = []
    line_psd_list = []

    # まずは各subplotの初期設定
    for i, ch in enumerate(Ch_dict):
        ax_raw = axes[0, i]
        ax_psd = axes[1, i]

        # Raw 波形用の初期描画
        signal_ch = signal[i, :]
        ax_raw.set_xlim(0, window_sec)
        ax_raw.set_ylim(np.min(signal_ch), np.max(signal_ch))
        ax_raw.set_xlabel("Time [sec]")
        ax_raw.set_ylabel("Amplitude")
        ax_raw.set_title(f"Raw (ch {ch})")
        line_raw, = ax_raw.plot([], [], 'b-')  # 更新対象

        # PSD 用の初期描画
        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel("Power")
        ax_psd.set_title(f"Power Spectrum (ch {ch})")
        line_psd, = ax_psd.plot([], [], 'r-')  # 更新対象

        line_raw_list.append(line_raw)
        line_psd_list.append(line_psd)

    # --- 3) アニメーション更新用関数 ---
    def init():
        """
        アニメーション開始時に呼ばれる初期化。
        """
        for line_raw, line_psd in zip(line_raw_list, line_psd_list):
            line_raw.set_data([], [])
            line_psd.set_data([], [])
        return line_raw_list + line_psd_list

    def update(frame):
        """
        各フレームで呼ばれる更新処理。
        frame: 0 ~ max_frames-1
        """
        start_time = frame * slide_sec
        end_time = start_time + window_sec
        print(start_time)

        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)

        for i, ch in enumerate(Ch_dict):
            ax_raw = axes[0, i]
            ax_psd = axes[1, i]

            signal_ch = signal[i, :]
            seg = signal_ch[start_idx:end_idx]

            # 時間軸
            t = np.linspace(0, window_sec, len(seg), endpoint=False)

            # Raw波形を更新
            line_raw = line_raw_list[i]
            line_raw.set_data(t, seg)

            # パワースペクトラムを計算して更新
            freqs, power = compute_power_spectrum(seg, fs)
            line_psd = line_psd_list[i]
            line_psd.set_data(freqs, power)

            # PSDの軸範囲を調整
            ax_psd.set_xlim(0, np.max(freqs))
            if len(power) > 0:
                ax_psd.set_ylim(0, np.max(power) * 1.1)
            else:
                ax_psd.set_ylim(0, 1)

            # タイトルをフレームごとに更新する例
            ax_raw.set_title(f"Raw (ch {ch}): {start_time:.2f}-{end_time:.2f}s")

        # 返り値として更新したアーティスト（Line2D等）をまとめて返す
        return line_raw_list + line_psd_list

    # --- 4) アニメーション作成 & 保存 or 表示 ---
    anim = FuncAnimation(fig, update, frames=max_frames,
                         init_func=init, blit=False, interval=100)

    if save_filename is not None:
        # ffmpeg が必要（インストールされていない場合は別途インストール）
        anim.save(save_filename, writer='ffmpeg', fps=5)
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
        save_filename=os.path.join(exp, "multi_channel_output1.mp4")
    )

