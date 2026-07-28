#結局解像あげようとするとかなり遅い


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
import multiprocessing as mp
from multiprocessing import Pool
import pyqtgraph as pg
print(pg.getConfigOption('imageDPI'))
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOption('imageDPI', 200)

from pyqtgraph.Qt import QtCore, QtGui,QtWidgets
import shutil
import subprocess



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

def compute_power_spectrum(signal, fs):
    signal -= np.mean(signal)
    fft_vals = np.fft.rfft(signal)  # 実数信号なら rfft の方が速い
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1 / fs)  # rfft に合わせて rfftfreq
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
    with Pool(processes=4) as p:
        results = p.map(_calc_psd_segment, tasks)
    p.close()
    p.terminate()
    p.join()
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


def make_multi_channel_movie_pyqtgraph(filename, Ch_dict,
                                       window_sec=8, slide_sec=1,
                                       save_filename=None):
    """
    pyqtgraph で複数チャネルの「Raw波形 + PSD」を (2行 × n列) に配置し、
    各フレームをキャプチャして FFmpeg で動画保存する。

    【流れ】
      1) 事前に全フレームのPSDを計算 (prepare_psd_in_parallel)
      2) pyqtgraph の GraphicsLayoutWidget 上に、チャンネル数ぶんの Raw + PSD プロットを用意
      3) forループで各フレームを可視化 → 画面キャプチャ → 連番PNGとして保存
      4) 連番PNGを ffmpeg で mp4 にまとめる
    """

    # ---------- (A) データ読み込みと PSD 計算 ----------
    signal, fs = load_ns3_data(filename, Ch_dict)
    psd_data = prepare_psd_in_parallel(signal, fs, window_sec, slide_sec)

    n_chan, num_samples = signal.shape
    total_sec = num_samples / fs
    max_frames = len(psd_data)
    if max_frames == 0:
        raise ValueError("データ長が短すぎてアニメーションを作成できません。")

    # PSDの全体最大
    all_power_vals = []
    for frame_info in psd_data:
        for (freqs, power) in frame_info:
            all_power_vals.append(np.max(power))
    max_power = max(all_power_vals) if len(all_power_vals) > 0 else 1

    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.quit()  # 既存の QApplication を終了
        # del app
        app.deletelater()
        app = None
    app = QtWidgets.QApplication([])

    win = pg.GraphicsLayoutWidget(title="Raw + PSD for each channel (pyqtgraph)")
    win.resize(3840, 2160)  # ウィンドウサイズはお好みで
    win.show()

    # PlotItem のリスト作成
    plots_raw = []
    plots_psd = []
    curves_raw = []
    curves_psd = []

    ch_keys = list(Ch_dict.keys())  # チャネル名リスト
    for i, ch_key in enumerate(ch_keys):
        # 行=0 (上段) に Raw
        p_raw = win.addPlot(row=0, col=i, title=f"Raw (ch {ch_key})")
        p_raw.setLabel('bottom', 'Time', units='s')
        p_raw.setLabel('left', 'Amplitude')
        p_raw.setXRange(0, window_sec)
        # p_raw.setYRange(global_min, global_max)
        p_raw.setYRange(-500, 500)
        curve_raw = p_raw.plot([], [], pen=pg.mkPen(color="b", width=0.2))

        # 行=1 (下段) に PSD
        p_psd = win.addPlot(row=1, col=i, title=f"PSD (ch {ch_key})")
        p_psd.setLabel('bottom', 'Freq', units='Hz')
        p_psd.setLabel('left', 'Power')
        p_psd.setXRange(0, 100)
        p_psd.setYRange(0, 2*10**6)
        curve_psd = p_psd.plot([], [], pen=pg.mkPen(color="r", width=0.2))

        plots_raw.append(p_raw)
        plots_psd.append(p_psd)
        curves_raw.append(curve_raw)
        curves_psd.append(curve_psd)

    # ---------- (C) 各フレームを更新＆スクリーンキャプチャ ----------
    # 連番画像を保存する一時フォルダ
    if save_filename is None:
        print("save_filename is None -> just show (no file saving).")
        temp_dir = None
    else:
        temp_dir = os.path.join(os.path.dirname(save_filename), "_frames")
        # if os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    # forループで全フレーム描画 → 画像として保存
    for frame in range(max_frames):
        # 波形切り出し
        start_time = frame * slide_sec
        end_time = start_time + window_sec
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)

        for i in range(n_chan):
            # --- Raw ---
            seg = signal[i, start_idx:end_idx]
            t = np.linspace(0, window_sec, len(seg), endpoint=False)
            curves_raw[i].setData(t, seg)

            # --- PSD ---
            freqs, power = psd_data[frame][i]
            valid = (freqs >= 0) & (freqs <= fs / 2)
            curves_psd[i].setData(freqs[valid], power[valid])

        # GUIイベントを処理して描画を更新させる
        app.processEvents()

        # 連番画像として保存したい場合
        if temp_dir is not None:
            # ウィンドウ全体をキャプチャ
            pixmap = win.grab()
            out_path = os.path.join(temp_dir, f"frame_{frame:05d}.png")
            # scaled_pixmap = pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2,
            #                               QtCore.Qt.KeepAspectRatio,
            #                               QtCore.Qt.SmoothTransformation)  # なめらかに拡大
            # scaled_pixmap.save(out_path, "PNG")

            # pixmap.save(out_path, "PNG")

    # ---------- (D) FFmpeg で mp4 化 ----------
    if temp_dir is not None:
        # ffmpeg コマンドで連番 PNG を mp4 にまとめる
        # 例: ffmpeg -framerate 10 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4
        cmd = [
            "ffmpeg",
            "-y",  # 上書き
            "-framerate", "1",  # 1秒あたり10フレーム
            "-i", os.path.join(temp_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            save_filename
        ]
        print("Running ffmpeg to create:", save_filename)
        try:
            subprocess.run(cmd, check=True, timeout=60)  # 60秒のタイムアウトを設定
        except subprocess.TimeoutExpired:
            print("FFmpeg がフリーズしたため強制終了します。")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg のエラー: {e}")

        print("Video saved:", save_filename)

        # 不要なら後処理で画像を削除
        # shutil.rmtree(temp_dir)

    # もしウィンドウを自動的に閉じたいなら
    # win.close()
    # app.quit()

    # 画面を開いたままにしたいなら以下を有効化
    if save_filename is None:
        # ユーザーが閉じるまでブロック
        QtGui.QApplication.instance().exec_()


# -----------------------------
# メイン処理
# -----------------------------
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # すでに設定されていた場合はスルー

    root = tk.Tk()
    root.withdraw()

    mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")
    if not mouse_dir:
        print("Canceled.")
        exit()

    dlc_type, Ch_dict = extract_params(mouse_dir)

    # フォルダ内のサブフォルダを探索して *.ns3 を処理
    for exp in glob.glob(os.path.join(mouse_dir, "[!_]*")):
        ns3_files = glob.glob(os.path.join(exp, "*.ns3"))
        if not ns3_files:
            continue
        file_name = ns3_files[0]
        print("Processing:", file_name)

        # 出力先
        out_mp4 = os.path.join(exp, "multi_channel_output3.mp4")

        make_multi_channel_movie_pyqtgraph(
            filename=file_name,
            Ch_dict=Ch_dict,
            window_sec=8,
            slide_sec=1,
            save_filename=out_mp4
        )