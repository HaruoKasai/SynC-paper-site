import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
from scipy.signal import medfilt, find_peaks  # ✅ 新增 find_peaks
import pandas as pd  # ✅ 新增：用于保存 CSV
from matplotlib.ticker import MultipleLocator

# -----------------------------
# 参数设置
# -----------------------------
fps = 30.0  # 成像帧率 (frames per second)
bin_sec = 0.25  # binning 时间 (秒)
bin_size = int(fps * bin_sec)  # 每个bin包含的帧数

phase1_frames = (0, 9999)
phase2_frames = (10000, 39999)
phase1_duration_min = 5
phase2_duration_min = 15

plot_mean = False  # 是否绘制所有ROI平均曲线
use_neu = False     # 是否使用 Fneu 背景校正
neu_ratio = 0.7    # 背景校正比例

# 🟦 新增参数：中值滤波设置
apply_median_filter = True   # 是否启用中值滤波
median_filter_sec = 0.1      # 滤波窗口大小（秒）

# 🟦 新增参数：F0 计算模式
f0_mode = "sliding"   # 可选: "global" 或 "sliding"
f0_window_sec = 8.0   # 滑动窗口总长度（秒）
f0_percentile = 8     # 百分位值

# 🟦 新增参数：峰值检测设置
peak_height = 0.5         # ΔF/F 峰值阈值
peak_distance_sec = 0.1   # 峰最小间隔（秒）

# -----------------------------
# 选择 NPY 文件
# -----------------------------
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="请选择 F.npy 文件", filetypes=[("NumPy Files", "*.npy")])
if not file_path:
    print("未选择文件。")
    exit()

# 自动匹配背景文件
neu_path = file_path.replace("F.npy", "Fneu.npy")
if use_neu and os.path.exists(neu_path):
    print(f"检测到背景文件：{neu_path}")
else:
    print("未检测到对应背景文件，将跳过背景校正。")
    use_neu = False

# -----------------------------
# 读取数据
# -----------------------------
F = np.load(file_path)  # shape = (cells, frames)
if use_neu:
    Fneu = np.load(neu_path)
    assert Fneu.shape == F.shape, "Fneu 与 F 的维度不匹配！"
else:
    Fneu = np.zeros_like(F)

print(f"数据维度: {F.shape}")
n_cells, n_frames = F.shape

# -----------------------------
# 选择要分析的 ROI 编号
# -----------------------------
roi_input = input(f"请输入要分析的 ROI 编号（0~{n_cells-1}，可用逗号分隔，如 0,1,5,7）：\n")
roi_indices = [int(i.strip()) for i in roi_input.split(',') if i.strip().isdigit()]
F = F[roi_indices, :]
Fneu = Fneu[roi_indices, :]
print(f"已选择 ROI: {roi_indices}")
n_cells = len(roi_indices)

# -----------------------------
# 背景校正
# -----------------------------
if use_neu:
    Fcorr = F - neu_ratio * Fneu
else:
    Fcorr = F.copy()

# ==========================================================
# 🟩 中值滤波（可选）
# ==========================================================
if apply_median_filter:
    kernel_size = int(median_filter_sec * fps)
    if kernel_size % 2 == 0:
        kernel_size += 1
    print(f"应用中值滤波：窗口 = {median_filter_sec}s ({kernel_size} 帧)")
    Fcorr = np.array([medfilt(trace, kernel_size=kernel_size) for trace in Fcorr])
else:
    print("未启用中值滤波。")

# ==========================================================
# 🟩 全局或滑动百分位 F0 的 ΔF/F 计算
# ==========================================================
def compute_dff(Fcorr, fps, mode="global", window_sec=8.0, perc=8):
    n_cells, n_frames = Fcorr.shape
    if mode == "global":
        F0 = np.percentile(Fcorr, perc, axis=1, keepdims=True)
        dff = (Fcorr - F0) / F0
        print(f"F0 计算方式：全局 {perc}th 百分位")
    elif mode == "sliding":
        half_win = int((window_sec / 2) * fps)
        F0 = np.zeros_like(Fcorr)
        for i in range(n_frames):
            start = max(0, i - half_win)
            end = min(n_frames, i + half_win)
            F0[:, i] = np.percentile(Fcorr[:, start:end], perc, axis=1)
        dff = (Fcorr - F0) / F0
        print(f"F0 计算方式：滑动 {perc}th 百分位（窗口 {window_sec:.1f} 秒）")
    else:
        raise ValueError("mode 必须是 'global' 或 'sliding'")
    return dff, F0

# 🟦 计算 ΔF/F
dff, F0 = compute_dff(Fcorr, fps, mode=f0_mode, window_sec=f0_window_sec, perc=f0_percentile)

# ==========================================================
# 🟩 Binning
# ==========================================================
def binning(data, bin_size):
    n_cells, n_frames = data.shape
    n_bins = n_frames // bin_size
    data = data[:, :n_bins * bin_size]
    data_binned = data.reshape(n_cells, n_bins, bin_size).mean(axis=2)
    return data_binned

dff_binned = binning(dff, bin_size)
F_binned = binning(F, bin_size)
Fcorr_binned = binning(Fcorr, bin_size)
Fneu_binned = binning(Fneu, bin_size)

n_cells, n_frames_binned = dff_binned.shape

# -----------------------------
# 构造时间轴（分钟）
# -----------------------------
x = np.arange(n_frames_binned)
time_min = np.zeros_like(x, dtype=float)
for i, xi in enumerate(x):
    frame_idx = xi * bin_size
    if phase1_frames[0] <= frame_idx < phase1_frames[1]:
        t = (frame_idx / (phase1_frames[1] - phase1_frames[0])) * phase1_duration_min - phase1_duration_min
    else:
        t = ((frame_idx - phase2_frames[0]) / (phase2_frames[1] - phase2_frames[0])) * phase2_duration_min
    time_min[i] = t

# -----------------------------
# 绘图函数
# -----------------------------
def plot_segment(t_range, save_pdf=True):
    t_min, t_max = t_range
    mask = (time_min >= t_min) & (time_min <= t_max)
    fig, ax = plt.subplots(figsize=(100, 5))

    for i in range(n_cells):
        ax.plot(time_min[mask], dff_binned[i, mask], linewidth=1.0, label=f'ROI {roi_indices[i]}')

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("ΔF/F")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right')

    # 设置主刻度（带标签）
    main_tick_interval = 1.0  # 1分钟为主刻度
    ax.set_xticks(np.arange(t_min, t_max + main_tick_interval, main_tick_interval))

    # 设置次刻度（不带标签，仅显示刻度线）
    minor_tick_interval = 0.1  # 0.1分钟为次刻度
    ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_interval))
    ax.tick_params(which='minor', length=5, color='gray')  # 调整次刻度线长度和颜色

    plt.title(f"{os.path.basename(file_path)} | {t_min}~{t_max} min (ΔF/F, {bin_sec:.1f}s binning)")
    plt.tight_layout()

    if save_pdf:
        roi_str = '_'.join([str(r) for r in roi_indices])
        pdf_name = f"{os.path.splitext(file_path)[0]}_ROI{roi_str}_{t_min}~{t_max}min_{bin_sec:.1f}s_bin_dFF.pdf"
        plt.savefig(pdf_name)
        print(f"已保存：{pdf_name}")
    plt.close()

"""
# ==========================================================
# 🟩 定义函数：检测指定时间段内的峰值并保存为 CSV
# ==========================================================
def detect_peaks_in_segment(t_range):
    t_min, t_max = t_range
    mask = (time_min >= t_min) & (time_min <= t_max)
    frame_indices = np.where(mask)[0]

    all_peaks = []
    peak_distance = int(peak_distance_sec * fps)

    for i, trace in enumerate(dff):
        peaks, props = find_peaks(trace, height=peak_height, distance=peak_distance)
        # ✅ 只保留当前时间段内的峰
        peaks_in_range = [p for p in peaks if p in frame_indices]
        for p in peaks_in_range:
            all_peaks.append({
                "ROI": roi_indices[i],
                "Phase": f"{t_min}~{t_max}min",
                "Peak_Frame": int(p),
                "Peak_Time(s)": p / fps,
                "Peak_Time(min)": time_min[p],
                "Peak_Value": trace[p]
            })

    if not all_peaks:
        print(f"⚠️ 时间段 {t_min}~{t_max} min 内未检测到峰。")
        return

    roi_str = '_'.join([str(r) for r in roi_indices])
    df_peaks = pd.DataFrame(all_peaks)
    csv_path = f"{os.path.splitext(file_path)[0]}_ROI{roi_str}_{t_min}~{t_max}min_peaks.csv"
    df_peaks.to_csv(csv_path, index=False)
    print(f"✅ 峰值检测完成（{t_min}~{t_max}min），共检测到 {len(df_peaks)} 个峰。已保存：{csv_path}")
"""

def detect_global_peaks():
    all_peaks = []
    peak_distance = int(peak_distance_sec * fps)

    for i, trace in enumerate(dff):
        peaks, props = find_peaks(trace, height=peak_height, distance=peak_distance)
        for p in peaks:
            # 判断峰所在阶段
            if phase1_frames[0] <= p < phase1_frames[1]:
                # phase1 倒序分钟
                t_min_val = (p / (phase1_frames[1] - phase1_frames[0])) * phase1_duration_min - phase1_duration_min
                phase = "Phase1"
            elif phase2_frames[0] <= p < phase2_frames[1]:
                # phase2 正序分钟
                t_min_val = ((p - phase2_frames[0]) / (phase2_frames[1] - phase2_frames[0])) * phase2_duration_min
                phase = "Phase2"
            else:
                t_min_val = np.nan
                phase = "OutOfPhase"

            all_peaks.append({
                "ROI": roi_indices[i],
                "Phase": phase,
                "Peak_Frame": int(p),
                "Peak_Time(s)": p / fps,
                "Peak_Time(min)": t_min_val,
                "Peak_Value": trace[p]
            })

    if not all_peaks:
        print("⚠️ 全局峰值未检测到。")
        return

    roi_str = '_'.join([str(r) for r in roi_indices])
    df_peaks = pd.DataFrame(all_peaks)
    csv_path = f"{os.path.splitext(file_path)[0]}_ROI{roi_str}_global_peaks.csv"
    df_peaks.to_csv(csv_path, index=False)
    print(f"✅ 全局峰值检测完成，共 {len(df_peaks)} 个峰。已保存：{csv_path}")

# -----------------------------
# 绘图执行
# -----------------------------
time_segments = [(-5, 0), (5, 10)]
for seg in time_segments:
    plot_segment(seg)
    # detect_peaks_in_segment(seg)
detect_global_peaks()
