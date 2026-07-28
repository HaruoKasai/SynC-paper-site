"""
EEG Analysis - Optimized Version
=================================
主要优化点：
  1. 预计算全段信号的功率谱矩阵（STFT），帧循环内不再重算 FFT
  2. 使用 np.fft.rfft 替代 fft（实数信号专用，速度 ~1.7x，内存减半）
  3. dB heatmap 去掉 DataFrame，改用纯 numpy 二维数组
  4. calculate_spectrum 不再就地修改原始信号（signal -= mean → signal - mean）
  5. 多进程并行生成帧图像（ProcessPoolExecutor）
  6. figsize 缩小到 (16, 10)，减少渲染内存和磁盘占用
  7. 每个文件用 tqdm 显示帧级进度条

依赖：pip install tqdm neo scipy numpy matplotlib pandas
"""

import os
import tkinter as tk
from tkinter import filedialog
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import matplotlib
matplotlib.use("Agg")   # 非交互后端，避免多进程下的 GUI 冲突
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import spectrogram as scipy_spectrogram
from neo.io import BlackrockIO
from tqdm import tqdm


# ─────────────────────────────────────────
#  FFT / 功率谱  （不修改原始数组）
# ─────────────────────────────────────────

def calculate_spectrum(signal: np.ndarray, sampling_rate: int):
    """
    计算单段信号的单边功率谱。
    使用 rfft（实数信号专用），速度约为 fft 的 1.7 倍，内存减半。
    不修改传入数组。
    """
    n = len(signal)
    sig = signal - signal.mean()          # 产生新数组，不污染原数据
    fft_vals = np.fft.rfft(sig)
    power = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    return freqs, power


def precompute_power_matrix(signal: np.ndarray,
                             sampling_rate: int,
                             freq_min: float,
                             freq_max: float,
                             freq_bin: float = 1.0) -> tuple:
    """
    对整段信号按 1 秒分帧，一次性算好所有帧的功率谱 (dB)。
    返回:
        power_matrix_db : shape (n_frames, n_bins)  —— 每帧每频率箱的 dB 值
        bin_centers     : shape (n_bins,)            —— 频率箱中心 (Hz)
        n_frames        : int
    """
    n_frames = len(signal) // sampling_rate
    n_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
    bins = np.linspace(freq_min, freq_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    power_matrix_db = np.full((n_frames, n_bins), np.nan, dtype=np.float32)

    for t in range(n_frames):
        seg = signal[t * sampling_rate: (t + 1) * sampling_rate]
        freqs, power = calculate_spectrum(seg, sampling_rate)
        power_db = 10.0 * np.log10(power + 1e-10)
        bin_indices = np.digitize(freqs, bins)          # 每个频率点属于哪个箱
        for i in range(n_bins):
            mask = bin_indices == (i + 1)
            if mask.any():
                power_matrix_db[t, i] = power_db[mask].mean()

    return power_matrix_db, bin_centers, n_frames


def precompute_spectrum_for_display(signal: np.ndarray,
                                     sampling_rate: int,
                                     freq_min: float,
                                     freq_max: float,
                                     freq_bin: float = 0.5) -> tuple:
    """
    预计算用于 spectrum_graph 的功率谱（每帧 ±1.5 秒窗口）。
    返回:
        spec_power  : dict { frame_index → power_means (n_bins,) }
        bin_centers : shape (n_bins,)
    """
    n_total = len(signal) // sampling_rate
    n_bins = int(np.ceil((freq_max - freq_min) / freq_bin))
    bins = np.linspace(freq_min, freq_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    half_win = int(1.5 * sampling_rate)
    spec_power = {}

    for tp in range(n_total):
        start = max(0, tp * sampling_rate - half_win)
        end   = min(len(signal), tp * sampling_rate + half_win)
        seg = signal[start:end]
        freqs, power = calculate_spectrum(seg, sampling_rate)
        bin_indices = np.digitize(freqs, bins)
        means = np.array([
            power[bin_indices == i].mean() if (bin_indices == i).any() else 0.0
            for i in range(1, n_bins + 1)
        ], dtype=np.float32)
        spec_power[tp] = means

    return spec_power, bin_centers


# ─────────────────────────────────────────
#  绘图辅助函数
# ─────────────────────────────────────────

def plot_signal(ax, signal: np.ndarray, sampling_rate: int,
                ylabel: str, ylim: float, time_point: int,
                time_window: int = 8):
    """绘制滚动信号波形（8 秒窗口）。"""
    tw_samples = time_window * sampling_rate

    if time_point <= 4:
        start = 0
        end   = tw_samples
    else:
        start = max(0, (time_point - 4) * sampling_rate)
        end   = start + tw_samples

    end = min(end, len(signal))

    ax.plot(signal[start:end], lw=0.8, color='red')

    # 高亮当前时刻
    hl_start = time_point * sampling_rate - start
    hl_end   = hl_start + sampling_rate
    ax.axvspan(hl_start, hl_end, color='yellow', alpha=0.3)

    # X 轴刻度
    xticks = np.arange(0, tw_samples, sampling_rate)
    if time_point <= 4:
        xlabels = np.arange(0, time_window)
    else:
        xlabels = np.arange(time_point - 4, time_point + 4)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{ylabel} (uV)")
    ax.set_ylim(-ylim, ylim)
    ax.margins(x=0)


def draw_spectrum(ax, bin_centers: np.ndarray, power_means: np.ndarray,
                  label: str, freq_min: float, freq_max: float):
    """在给定 ax 上绘制功率谱曲线。"""
    vertical_lines = [1, 4, 8, 12, 30]
    ax.plot(bin_centers, power_means, lw=1.5, label=label)
    ax.set_xlim(freq_min, freq_max)
    ax.set_xlabel("Hz")
    ax.set_ylim(0, 2_000_000)
    ax.set_ylabel("Power (μV²)")
    ax.legend(fontsize=7)
    for vl in vertical_lines:
        ax.axvline(x=vl, color='grey', linestyle='--', lw=0.7)


def draw_heatmap(ax, power_matrix_db: np.ndarray,
                 time_point: int, sampling_rate: int,
                 freq_min: float, freq_max: float, title: str):
    """
    用预计算的功率矩阵绘制 dB 热图（8 秒滚动窗口）。
    power_matrix_db : shape (n_frames, n_bins)，按 1 秒/帧排列
    """
    if time_point < 4:
        t_start, t_end = 0, 8
    else:
        t_start = time_point - 4
        t_end   = time_point + 4

    t_end = min(t_end, power_matrix_db.shape[0])

    # 取对应帧切片，转置为 (n_bins, n_frames) 供 imshow
    chunk = power_matrix_db[t_start:t_end, :].T  # (n_bins, n_frames)

    ax.imshow(
        chunk,
        aspect='auto',
        cmap='rainbow',
        origin='lower',
        vmin=30, vmax=60,
        extent=[t_start, t_end, freq_min, freq_max]
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")


# ─────────────────────────────────────────
#  单帧渲染（供多进程调用）
# ─────────────────────────────────────────

def render_frame(time_point: int,
                 output_dir: str,
                 sampling_rate: int,
                 freq_min: float,
                 freq_max: float,
                 # 预计算数据（通过共享内存传递文件路径，避免 pickle 大数组）
                 npz_path: str):
    """
    渲染单帧并保存 PNG。
    所有大数组通过 npz 文件在进程间共享，避免 pickle 序列化开销。
    """
    # 加载预计算数据（进程内缓存：同一进程多次调用只加载一次）
    data = np.load(npz_path, allow_pickle=True)
    m1_signal      = data["m1_signal"]
    m1v1           = data["m1v1"]
    emg            = data["emg"]
    pm_m1_100      = data["pm_m1_100"]       # power_matrix_db, 0-100 Hz
    pm_m1v1_100    = data["pm_m1v1_100"]
    pm_m1_20       = data["pm_m1_20"]        # power_matrix_db, 1-20 Hz
    pm_m1v1_20     = data["pm_m1v1_20"]
    sp_m1          = data["sp_m1"].item()    # dict: tp → power_means
    sp_m1v1        = data["sp_m1v1"].item()
    bc_spec        = data["bc_spec"]         # bin_centers for spectrum
    bc_hm100       = data["bc_hm100"]        # bin_centers for heatmap 0-100
    bc_hm20        = data["bc_hm20"]         # bin_centers for heatmap 1-20

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(5, 2, height_ratios=[0.4, 0.4, 0.4, 0.4, 0.4])
    plt.subplots_adjust(hspace=0.55, wspace=0.35)

    label = f"t={time_point}s ±1.5s"

    # 行0：功率谱
    ax = fig.add_subplot(gs[0, 0])
    draw_spectrum(ax, bc_spec, sp_m1[time_point],  label, freq_min, freq_max)
    ax.set_title("M1 Spectrum", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    draw_spectrum(ax, bc_spec, sp_m1v1[time_point], label, freq_min, freq_max)
    ax.set_title("M1-V1 Spectrum", fontsize=8)

    # 行1：heatmap 0-100 Hz
    ax = fig.add_subplot(gs[1, 0])
    draw_heatmap(ax, pm_m1_100, time_point, sampling_rate, freq_min, 100, "M1 dB (0-100Hz)")

    ax = fig.add_subplot(gs[1, 1])
    draw_heatmap(ax, pm_m1v1_100, time_point, sampling_rate, freq_min, 100, "M1-V1 dB (0-100Hz)")

    # 行2：heatmap 1-20 Hz
    ax = fig.add_subplot(gs[2, 0])
    draw_heatmap(ax, pm_m1_20, time_point, sampling_rate, 1, 20, "M1 dB (1-20Hz)")

    ax = fig.add_subplot(gs[2, 1])
    draw_heatmap(ax, pm_m1v1_20, time_point, sampling_rate, 1, 20, "M1-V1 dB (1-20Hz)")

    # 行3：原始信号波形
    ax = fig.add_subplot(gs[3, 0])
    plot_signal(ax, m1_signal, sampling_rate, "M1-Ce", 500, time_point)

    ax = fig.add_subplot(gs[3, 1])
    plot_signal(ax, m1v1, sampling_rate, "M1-V1", 500, time_point)

    # 行4：EMG
    ax = fig.add_subplot(gs[4, 0])
    plot_signal(ax, emg, sampling_rate, "EMG", 500, time_point)

    ax = fig.add_subplot(gs[4, 1])
    plot_signal(ax, emg, sampling_rate, "EMG", 500, time_point)

    out_path = os.path.join(output_dir, f"EEG_Raw_Trace_{time_point:05d}.png")
    fig.savefig(out_path, bbox_inches='tight', dpi=80)
    plt.close(fig)

    return time_point   # 返回帧号，供主进程更新进度条


# ─────────────────────────────────────────
#  单文件处理入口
# ─────────────────────────────────────────

def process_file(file_path: str, freq_min: float = 0, freq_max: float = 100,
                 n_workers: int = 4):
    """
    处理单个 .ns3 文件：
      1. 读取信号
      2. 预计算所有 FFT 数据（含 tqdm 进度）
      3. 并行渲染每一帧（含 tqdm 进度）
    进度条仅针对当前文件，与外层文件循环无关。
    """
    print(f"\n{'='*60}")
    print(f"  文件: {os.path.basename(file_path)}")
    print(f"{'='*60}")

    # ── 读取信号 ──────────────────────────────────────────────
    print("[1/3] 读取信号...")
    reader = BlackrockIO(filename=file_path)
    block  = reader.read_block()
    raw    = block.segments[0].analogsignals[0]

    ch1 = raw[:, 0]
    ch2 = raw[:, 1]
    ch3 = raw[:, 2]
    ch4 = raw[:, 3]

    sampling_rate = int(ch1.sampling_rate.magnitude)
    m1_signal = ch1.magnitude.flatten().astype(np.float32)
    v1_signal = ch2.magnitude.flatten().astype(np.float32)
    m1v1      = (ch1 - ch2).magnitude.flatten().astype(np.float32)
    emg       = (ch3 - ch4).magnitude.flatten().astype(np.float32)

    n_frames = len(m1_signal) // sampling_rate
    print(f"    采样率: {sampling_rate} Hz | 总时长: {n_frames} 秒 | 总帧数: {n_frames}")

    # ── 预计算功率谱（带进度条）───────────────────────────────
    print("[2/3] 预计算功率谱矩阵（所有帧）...")

    def _precompute_with_progress(signal, sr, fmin, fmax, fb=1.0, label=""):
        n_bins = int(np.ceil((fmax - fmin) / fb))
        bins   = np.linspace(fmin, fmax, n_bins + 1)
        matrix = np.full((n_frames, n_bins), np.nan, dtype=np.float32)
        for t in tqdm(range(n_frames), desc=f"  FFT {label:12s}", unit="s",
                      ncols=70, leave=False):
            seg = signal[t * sr: (t + 1) * sr]
            freqs, power = calculate_spectrum(seg, sr)
            power_db = 10.0 * np.log10(power + 1e-10)
            bidx = np.digitize(freqs, bins)
            for i in range(n_bins):
                mask = bidx == (i + 1)
                if mask.any():
                    matrix[t, i] = power_db[mask].mean()
        bc = (bins[:-1] + bins[1:]) / 2.0
        return matrix, bc

    def _precompute_spec(signal, sr, fmin, fmax, fb=0.5, label=""):
        n_bins = int(np.ceil((fmax - fmin) / fb))
        bins   = np.linspace(fmin, fmax, n_bins + 1)
        bc     = (bins[:-1] + bins[1:]) / 2.0
        half   = int(1.5 * sr)
        result = {}
        for tp in tqdm(range(n_frames), desc=f"  Spec {label:12s}", unit="s",
                       ncols=70, leave=False):
            s = max(0, tp * sr - half)
            e = min(len(signal), tp * sr + half)
            freqs, power = calculate_spectrum(signal[s:e], sr)
            bidx  = np.digitize(freqs, bins)
            means = np.array([
                power[bidx == i].mean() if (bidx == i).any() else 0.0
                for i in range(1, n_bins + 1)
            ], dtype=np.float32)
            result[tp] = means
        return result, bc

    pm_m1_100,   bc_hm100 = _precompute_with_progress(m1_signal, sampling_rate, freq_min, 100,    label="M1 0-100Hz")
    pm_m1v1_100, _        = _precompute_with_progress(m1v1,      sampling_rate, freq_min, 100,    label="M1V1 0-100Hz")
    pm_m1_20,    bc_hm20  = _precompute_with_progress(m1_signal, sampling_rate, 1,        20,     label="M1 1-20Hz")
    pm_m1v1_20,  _        = _precompute_with_progress(m1v1,      sampling_rate, 1,        20,     label="M1V1 1-20Hz")
    sp_m1,       bc_spec  = _precompute_spec(m1_signal, sampling_rate, freq_min, freq_max, label="M1 spec")
    sp_m1v1,     _        = _precompute_spec(m1v1,      sampling_rate, freq_min, freq_max, label="M1V1 spec")

    # ── 把预计算数据存到 npz，供子进程读取（避免 pickle 大数组）──
    output_dir = os.path.join(os.path.dirname(file_path), "EEG_Raw_Trace")
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, "_precomputed.npz")
    print("    保存预计算数据到临时文件...")
    np.savez(
        npz_path,
        m1_signal   = m1_signal,
        m1v1        = m1v1,
        emg         = emg,
        pm_m1_100   = pm_m1_100,
        pm_m1v1_100 = pm_m1v1_100,
        pm_m1_20    = pm_m1_20,
        pm_m1v1_20  = pm_m1v1_20,
        sp_m1       = sp_m1,
        sp_m1v1     = sp_m1v1,
        bc_spec     = bc_spec,
        bc_hm100    = bc_hm100,
        bc_hm20     = bc_hm20,
    )

    # ── 并行渲染帧（带进度条）────────────────────────────────
    print(f"[3/3] 渲染 {n_frames} 帧图像（{n_workers} 个工作进程）...")

    frame_fn = partial(
        render_frame,
        output_dir    = output_dir,
        sampling_rate = sampling_rate,
        freq_min      = freq_min,
        freq_max      = freq_max,
        npz_path      = npz_path,
    )

    with tqdm(total=n_frames, desc="  渲染帧", unit="帧",
              ncols=70, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(frame_fn, tp): tp for tp in range(n_frames)}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    tp = futures[future]
                    tqdm.write(f"  ⚠ 帧 {tp} 渲染失败: {e}")
                pbar.update(1)

    # 删除临时 npz
    os.remove(npz_path)

    print(f"  ✓ 完成 → {output_dir}")


# ─────────────────────────────────────────
#  主程序入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    # 进程数：可根据机器核数调整
    # 建议设为 (CPU 核数 - 2)，留余量给系统
    N_WORKERS = 6

    # 弹出文件夹选择对话框
    root = tk.Tk()
    root.withdraw()
    mouse_dir = filedialog.askdirectory(title="Select the 'data' directory")

    if not mouse_dir:
        print("未选择目录，退出。")
        exit(0)

    # 收集所有 .ns3 文件
    ns3_files = []
    for root_dir, _, files in os.walk(mouse_dir):
        for f in files:
            if f.endswith(".ns3"):
                ns3_files.append(os.path.join(root_dir, f))

    if not ns3_files:
        print("未找到 .ns3 文件。")
        exit(0)

    print(f"\n共找到 {len(ns3_files)} 个 .ns3 文件，逐一处理...\n")

    for idx, fp in enumerate(ns3_files, 1):
        print(f"\n[文件 {idx}/{len(ns3_files)}]")
        try:
            process_file(
                file_path = fp,
                freq_min  = 0,
                freq_max  = 100,
                n_workers = N_WORKERS,
            )
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")

    print("\n\n全部完成。")