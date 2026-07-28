import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.signal import cheby1, butter, filtfilt
from neo.io import BlackrockIO
from tkinter import filedialog

############################
# 1. 前処理用の関数
############################
def cheby1_bandpass_filter(data, lowcut, highcut, sampling_rate, order=4, ripple=0.5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = cheby1(order, ripple, [low, high], btype='band')
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, sampling_rate, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def calculate_spectrum(signal, sampling_rate):
    """単純なFFTでパワースペクトルを計算し、周波数とパワーを返す。"""
    n = len(signal)
    signal = signal - np.mean(signal)  # DCオフセット除去
    freqs = np.fft.fftfreq(n, d=1.0/sampling_rate)
    fft_vals = np.fft.fft(signal)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    # 正の周波数成分だけ切り出し
    half = n // 2
    return freqs[:half], power_spectrum[:half]

def compute_db_heatmap_data(signal, sampling_rate, start_time, end_time,
                            freq_min, freq_max, time_bin=1.0, freq_bin=1.0):
    """
    dBスペクトログラム(heatmap)を作るために、
    指定区間[start_time, end_time]のデータを取り出して、
    time_bin区切りでFFT → dB変換 → 周波数方向にもfreq_bin区切りでまとめる。
    返り値は (heatmap_data_2d, extent) のタプル。
    """
    start_sample = int(start_time * sampling_rate)
    end_sample   = int(end_time * sampling_rate)
    segment = signal[start_sample:end_sample]

    # time_binごとのスペクトラムをまとめる
    segment_len = len(segment)
    samples_per_bin = int(time_bin * sampling_rate)
    n_bins_time = segment_len // samples_per_bin

    # freq方向のbin
    num_bins_freq = int(np.ceil((freq_max - freq_min) / freq_bin))
    freq_edges = np.linspace(freq_min, freq_max, num_bins_freq+1)

    heatmap_list = []
    for t_idx in range(n_bins_time):
        bin_data = segment[t_idx*samples_per_bin:(t_idx+1)*samples_per_bin]
        freqs, power = calculate_spectrum(bin_data, sampling_rate)
        power_db = 10.0 * np.log10(power + 1e-10)

        # freq_bin区切りで平均
        bin_indices = np.digitize(freqs, freq_edges)
        row_data = []
        for fbin in range(1, num_bins_freq+1):
            vals_in_bin = power_db[bin_indices == fbin]
            if len(vals_in_bin) == 0:
                row_data.append(np.nan)  # データがなければNaN
            else:
                row_data.append(np.mean(vals_in_bin))
        heatmap_list.append(row_data)

    heatmap_data = np.array(heatmap_list)

    # imshow用のextent: x軸=時間, y軸=周波数
    #   横軸: start_time -> end_time
    #   縦軸: freq_min -> freq_max
    # shape(行,列)= (n_bins_time, num_bins_freq)
    duration = end_time - start_time
    extent = [start_time, end_time, freq_min, freq_max]

    return heatmap_data, extent


############################
# 2. 描画(アニメーション)用のメイン関数
############################
def animate_eeg_analysis(ns3_file_path, freq_max=100, freq_min=0, video_out_path="output.mp4"):
    """
    ns3_file_path: .ns3ファイルのパス
    freq_max, freq_min: スペクトル表示やヒートマップの周波数範囲
    video_out_path: 出力動画ファイルパス
    """
    # --- 2.1 データ読み込み ---
    reader = BlackrockIO(filename=ns3_file_path)
    block = reader.read_block()
    raw_signals = [seg.analogsignals[0] for seg in block.segments]
    # 今回は最初のアナログ信号だけ使う想定(※ファイル仕様により適宜変えてください)
    ch2_signal = raw_signals[0][:, 1]
    sampling_rate = int(ch2_signal.sampling_rate.magnitude)
    signal_v = ch2_signal.magnitude.flatten()  # V_EEG

    ch1_signal = raw_signals[0][:, 0]          # M_EEG
    lfp = ch1_signal.magnitude.flatten()

    m1v1 = (ch1_signal - ch2_signal).magnitude.flatten()

    # EMGはch3 - ch4と想定
    ch3_signal = raw_signals[0][:, 2].magnitude.flatten()
    ch4_signal = raw_signals[0][:, 3].magnitude.flatten()
    emg = ch3_signal - ch4_signal

    total_duration = len(signal_v) / sampling_rate
    time_points = np.arange(0, total_duration, 1.0)  # 1秒刻みなどでアニメーションにする

    # --- 2.2 フィルタリングを事前にしておく (高速化のため) ---
    # 0-70Hz
    signal_v_0_70 = butter_lowpass_filter(signal_v, 70, sampling_rate)
    lfp_0_70 = butter_lowpass_filter(lfp, 70, sampling_rate)
    m1v1_0_70 = butter_lowpass_filter(m1v1, 70, sampling_rate)

    # 9-16Hz
    signal_v_9_16 = cheby1_bandpass_filter(signal_v, 9, 16, sampling_rate)
    lfp_9_16 = cheby1_bandpass_filter(lfp, 9, 16, sampling_rate)
    m1v1_9_16 = cheby1_bandpass_filter(m1v1, 9, 16, sampling_rate)

    # --- 2.3 FigureとAxesを作る ---
    fig = plt.figure(figsize=(30, 20))
    gs = gridspec.GridSpec(6, 3, height_ratios=[0.6, 0.4, 0.4, 0.4, 0.4, 0.4])
    plt.subplots_adjust(hspace=0.5)

    # 主要なサブプロット
    ax_spectrum_v    = fig.add_subplot(gs[0, 0])
    ax_spectrum_lfp  = fig.add_subplot(gs[0, 1])
    ax_spectrum_m1v1 = fig.add_subplot(gs[0, 2])

    ax_db_v    = fig.add_subplot(gs[1, 0])
    ax_db_lfp  = fig.add_subplot(gs[1, 1])
    ax_db_m1v1 = fig.add_subplot(gs[1, 2])

    ax_db_v_20    = fig.add_subplot(gs[2, 0])
    ax_db_lfp_20  = fig.add_subplot(gs[2, 1])
    ax_db_m1v1_20 = fig.add_subplot(gs[2, 2])

    ax_v_9_16    = fig.add_subplot(gs[3, 0])
    ax_lfp_9_16  = fig.add_subplot(gs[3, 1])
    ax_m1v1_9_16 = fig.add_subplot(gs[3, 2])

    ax_v_0_70    = fig.add_subplot(gs[4, 0])
    ax_lfp_0_70  = fig.add_subplot(gs[4, 1])
    ax_m1v1_0_70 = fig.add_subplot(gs[4, 2])

    ax_emg = fig.add_subplot(gs[5, :])

    # --- 2.4 プロットオブジェクト(Line2Dやimshow)を作る ---
    # 例: スペクトルプロット用の Line2D を3つ
    line_spectrum_v,   = ax_spectrum_v.plot([], [], 'r-', lw=1, label="V_EEG Spectrum")
    line_spectrum_lfp, = ax_spectrum_lfp.plot([], [], 'r-', lw=1, label="M_EEG Spectrum")
    line_spectrum_m1v1,= ax_spectrum_m1v1.plot([], [], 'r-', lw=1, label="M1-V1 Spectrum")

    # ヒートマップは imshow() オブジェクトを作り、のちに set_data() で更新
    im_db_v    = ax_db_v.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')
    im_db_lfp  = ax_db_lfp.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')
    im_db_m1v1 = ax_db_m1v1.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')

    im_db_v_20    = ax_db_v_20.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')
    im_db_lfp_20  = ax_db_lfp_20.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')
    im_db_m1v1_20 = ax_db_m1v1_20.imshow([[0]], aspect='auto', origin='lower', cmap='rainbow')

    # 0-70Hz や 9-16Hz のバンドパス波形を表示するためのライン
    line_v_9_16,    = ax_v_9_16.plot([], [], 'r-', lw=1)
    line_lfp_9_16,  = ax_lfp_9_16.plot([], [], 'r-', lw=1)
    line_m1v1_9_16, = ax_m1v1_9_16.plot([], [], 'r-', lw=1)

    line_v_0_70,    = ax_v_0_70.plot([], [], 'r-', lw=1)
    line_lfp_0_70,  = ax_lfp_0_70.plot([], [], 'r-', lw=1)
    line_m1v1_0_70, = ax_m1v1_0_70.plot([], [], 'r-', lw=1)

    # EMG
    line_emg, = ax_emg.plot([], [], 'r-', lw=1)

    # --- 各Axesに軸ラベルや固定の設定をしておく ---
    ax_spectrum_v.set_xlabel("Freq (Hz)");    ax_spectrum_v.set_ylabel("Power")
    ax_spectrum_lfp.set_xlabel("Freq (Hz)");  ax_spectrum_lfp.set_ylabel("Power")
    ax_spectrum_m1v1.set_xlabel("Freq (Hz)"); ax_spectrum_m1v1.set_ylabel("Power")

    ax_db_v.set_title("V_EEG Power dB")
    ax_db_lfp.set_title("M_EEG Power dB")
    ax_db_m1v1.set_title("M1-V1_EEG Power dB")

    # ... 必要に応じてさらに設定(ylimなど) ...

    # ---------------------------
    #   2.5 init関数
    # ---------------------------
    def init():
        # スペクトルラインは空にしておく
        line_spectrum_v.set_data([], [])
        line_spectrum_lfp.set_data([], [])
        line_spectrum_m1v1.set_data([], [])

        # imshow用ヒートマップを一旦1x1のゼロ配列に
        im_db_v.set_data([[0]])
        im_db_lfp.set_data([[0]])
        im_db_m1v1.set_data([[0]])

        im_db_v_20.set_data([[0]])
        im_db_lfp_20.set_data([[0]])
        im_db_m1v1_20.set_data([[0]])

        # バンドパス波形も空
        line_v_9_16.set_data([], [])
        line_lfp_9_16.set_data([], [])
        line_m1v1_9_16.set_data([], [])

        line_v_0_70.set_data([], [])
        line_lfp_0_70.set_data([], [])
        line_m1v1_0_70.set_data([], [])

        line_emg.set_data([], [])

        return (
            line_spectrum_v, line_spectrum_lfp, line_spectrum_m1v1,
            im_db_v, im_db_lfp, im_db_m1v1,
            im_db_v_20, im_db_lfp_20, im_db_m1v1_20,
            line_v_9_16, line_lfp_9_16, line_m1v1_9_16,
            line_v_0_70, line_lfp_0_70, line_m1v1_0_70,
            line_emg
        )

    # ---------------------------
    #   2.6 update関数
    # ---------------------------
    def update(frame_time):
        print(frame_time)
        """
        frame_time: アニメーションのフレーム(秒単位)。
                    time_pointsから渡される値を想定。
        """
        # 1) スペクトルを更新
        #   frame_timeの前後 ±1.5秒区間などでスペクトルを計算する例
        half_window = 1.5
        start_t = max(0, frame_time - half_window)
        end_t   = min(total_duration, frame_time + half_window)
        start_sample = int(start_t * sampling_rate)
        end_sample   = int(end_t   * sampling_rate)

        seg_v = signal_v[start_sample:end_sample]
        seg_lfp = lfp[start_sample:end_sample]
        seg_m1v1 = m1v1[start_sample:end_sample]

        freqs_v, power_v = calculate_spectrum(seg_v, sampling_rate)
        freqs_lfp, power_lfp = calculate_spectrum(seg_lfp, sampling_rate)
        freqs_m1v1, power_m1v1 = calculate_spectrum(seg_m1v1, sampling_rate)

        # スペクトルラインの更新
        line_spectrum_v.set_data(freqs_v, power_v)
        line_spectrum_lfp.set_data(freqs_lfp, power_lfp)
        line_spectrum_m1v1.set_data(freqs_m1v1, power_m1v1)

        # 軸の範囲などを調整
        ax_spectrum_v.set_xlim(freq_min, freq_max)
        ax_spectrum_lfp.set_xlim(freq_min, freq_max)
        ax_spectrum_m1v1.set_xlim(freq_min, freq_max)

        # 2) ヒートマップ更新
        #   - time_point<4sなら[0,8], それ以降なら[t-4, t+4] など
        #   - 例ではシンプルに毎回 [frame_time-4, frame_time+4] で計算
        if frame_time < 4:
            start_hm = 0
            end_hm   = 8
        else:
            start_hm = frame_time - 4
            end_hm   = frame_time + 4
            if end_hm > total_duration:
                end_hm = total_duration

        # v_eegヒートマップ (0-70Hz)
        hm_data_v, extent_v = compute_db_heatmap_data(signal_v, sampling_rate,
                                                      start_hm, end_hm,
                                                      freq_min, 70,
                                                      time_bin=1.0, freq_bin=1.0)
        im_db_v.set_data(hm_data_v.T)  # 転置して (freq, time) -> (行,列) = (freq, time)
        im_db_v.set_extent(extent_v)   # [xmin,xmax,ymin,ymax]

        # lfpヒートマップ (0-70Hz)
        hm_data_lfp, extent_lfp = compute_db_heatmap_data(lfp, sampling_rate,
                                                          start_hm, end_hm,
                                                          freq_min, 70,
                                                          time_bin=1.0, freq_bin=1.0)
        im_db_lfp.set_data(hm_data_lfp.T)
        im_db_lfp.set_extent(extent_lfp)

        # m1v1ヒートマップ (0-70Hz)
        hm_data_m1v1, extent_m1v1 = compute_db_heatmap_data(m1v1, sampling_rate,
                                                            start_hm, end_hm,
                                                            freq_min, 70,
                                                            time_bin=1.0, freq_bin=1.0)
        im_db_m1v1.set_data(hm_data_m1v1.T)
        im_db_m1v1.set_extent(extent_m1v1)

        # さらに 20Hzまでのヒートマップ
        hm_data_v_20, extent_v_20 = compute_db_heatmap_data(signal_v, sampling_rate,
                                                            start_hm, end_hm,
                                                            1, 20,
                                                            time_bin=1.0, freq_bin=1.0)
        im_db_v_20.set_data(hm_data_v_20.T)
        im_db_v_20.set_extent(extent_v_20)

        hm_data_lfp_20, extent_lfp_20 = compute_db_heatmap_data(lfp, sampling_rate,
                                                                start_hm, end_hm,
                                                                1, 20,
                                                                time_bin=1.0, freq_bin=1.0)
        im_db_lfp_20.set_data(hm_data_lfp_20.T)
        im_db_lfp_20.set_extent(extent_lfp_20)

        hm_data_m1v1_20, extent_m1v1_20 = compute_db_heatmap_data(m1v1, sampling_rate,
                                                                  start_hm, end_hm,
                                                                  1, 20,
                                                                  time_bin=1.0, freq_bin=1.0)
        im_db_m1v1_20.set_data(hm_data_m1v1_20.T)
        im_db_m1v1_20.set_extent(extent_m1v1_20)

        # 3) バンドパス波形 (9-16Hz, 0-70Hz) の更新
        #   - 可視範囲: [frame_time-4, frame_time+4] → サンプルに合わせてスライス
        start_bpf = int(start_hm*sampling_rate)
        end_bpf   = int(end_hm*sampling_rate)

        # 9-16Hz
        sig_v_9_16_seg = signal_v_9_16[start_bpf:end_bpf]
        sig_lfp_9_16_seg = lfp_9_16[start_bpf:end_bpf]
        sig_m1v1_9_16_seg = m1v1_9_16[start_bpf:end_bpf]
        t_axis_9_16 = np.linspace(start_hm, end_hm, len(sig_v_9_16_seg))

        line_v_9_16.set_data(t_axis_9_16, sig_v_9_16_seg)
        line_lfp_9_16.set_data(t_axis_9_16, sig_lfp_9_16_seg)
        line_m1v1_9_16.set_data(t_axis_9_16, sig_m1v1_9_16_seg)

        ax_v_9_16.set_xlim(start_hm, end_hm)
        ax_lfp_9_16.set_xlim(start_hm, end_hm)
        ax_m1v1_9_16.set_xlim(start_hm, end_hm)

        ax_v_9_16.set_ylim(-500, 500)
        ax_lfp_9_16.set_ylim(-500, 500)
        ax_m1v1_9_16.set_ylim(-500, 500)

        # 0-70Hz
        sig_v_0_70_seg = signal_v_0_70[start_bpf:end_bpf]
        sig_lfp_0_70_seg = lfp_0_70[start_bpf:end_bpf]
        sig_m1v1_0_70_seg = m1v1_0_70[start_bpf:end_bpf]
        t_axis_0_70 = np.linspace(start_hm, end_hm, len(sig_v_0_70_seg))

        line_v_0_70.set_data(t_axis_0_70, sig_v_0_70_seg)
        line_lfp_0_70.set_data(t_axis_0_70, sig_lfp_0_70_seg)
        line_m1v1_0_70.set_data(t_axis_0_70, sig_m1v1_0_70_seg)

        ax_v_0_70.set_xlim(start_hm, end_hm)
        ax_lfp_0_70.set_xlim(start_hm, end_hm)
        ax_m1v1_0_70.set_xlim(start_hm, end_hm)

        ax_v_0_70.set_ylim(-500, 500)
        ax_lfp_0_70.set_ylim(-500, 500)
        ax_m1v1_0_70.set_ylim(-500, 500)

        # 4) EMG
        emg_seg = emg[start_bpf:end_bpf]
        t_axis_emg = np.linspace(start_hm, end_hm, len(emg_seg))
        line_emg.set_data(t_axis_emg, emg_seg)
        ax_emg.set_xlim(start_hm, end_hm)
        ax_emg.set_ylim(-500, 500)

        return (
            line_spectrum_v, line_spectrum_lfp, line_spectrum_m1v1,
            im_db_v, im_db_lfp, im_db_m1v1,
            im_db_v_20, im_db_lfp_20, im_db_m1v1_20,
            line_v_9_16, line_lfp_9_16, line_m1v1_9_16,
            line_v_0_70, line_lfp_0_70, line_m1v1_0_70,
            line_emg
        )

    # --- 2.7 FuncAnimation でアニメーション生成 ---
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=time_points,   # 各フレームでの時間(秒)リスト
        init_func=init,
        blit=False,           # heatmapなどある場合はblit=Falseが無難
        interval=200,         # (ms) フレーム間の遅延
        repeat=False
    )

    # --- 2.8 動画として保存 ---
    anim.save(video_out_path, fps=5, dpi=100)
    plt.close(fig)
    print(f"Saved animation to: {video_out_path}")


############################
# 3. 実行例: メイン部
############################
if __name__ == "__main__":
    # 例: ユーザーが解析したい .ns3 ファイルパスを指定
    ns3_file = r"X:\Behavior\EEG\SYNCit-C_Pup\20241219_z206-5_Pup-Cript-2x-2p-frontal-2p-parietal_(8w_M1-V1_male)\02-O1-10min_002\02-O1-10min_002.ns3"
    out_mp4  = r"X:\Behavior\EEG\SYNCit-C_Pup\20241219_z206-5_Pup-Cript-2x-2p-frontal-2p-parietal_(8w_M1-V1_male)\02-O1-10min_002\output.mp4"

    animate_eeg_analysis(ns3_file, freq_max=100, freq_min=0, video_out_path=out_mp4)
