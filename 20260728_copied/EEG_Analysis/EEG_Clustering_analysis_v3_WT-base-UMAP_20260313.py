"""
EEG/EMG/Velocity Clustering Analysis  v2
==========================================
跨组（WT vs 处理组）脑电行为状态无监督聚类分析

特征方案可选（启动时GUI选择）：
  A. EEG频谱 + EMG RMS
  B. EEG频谱 + EMG RMS + Velocity        <- 默认
  C. EEG频谱 + EMG RMS + Velocity + PAC/MI（较慢）

聚类方式：
  - 所有小鼠数据合并后联合做UMAP+K-means（跨组联合embedding）
  - Elbow法 + Silhouette Score 自动推荐最优K
  - 也可在GUI中强制指定K值

输出：
  - clustering_modeX_KY.pdf  （5页图）
  - clustering_modeX_KY.csv  （每个时间窗口的特征+cluster标签）

依赖：pip install h5py numpy scipy pandas matplotlib scikit-learn umap-learn
"""

import os
import sys
import glob
import traceback

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter, filtfilt, hilbert, welch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import tkinter as tk
import tkinter.messagebox
from tkinter import filedialog

# ===================================================================
#  可调参数区（也可在启动时GUI中修改）
# ===================================================================

FEATURE_MODE     = "B"      # "A" / "B" / "C"
WINDOW_SEC       = 8       # 特征提取窗口（秒）
OVERLAP_SEC      = 0        # 窗口重叠（秒）
FORCE_K          = None     # None=自动；填整数则强制使用该K
SUBCLUSTER_FORCE_K = None    # 子聚类强制K（None=自动）
ENABLE_SUBCLUSTER  = True    # 是否对全频段高活跃cluster做子聚类分析
USE_WT_BASIS       = True    # UMAP是否以WT组为基准fit
SPEC_MODE          = "dB"    # Spectrogram显示模式："dB" 或 "zscore"
K_RANGE          = range(2, 11)

# 时间范围（秒，相对于录制起点）
# "all"  : 使用全部数据
# "post" : 仅使用处理后数据（TREATMENT_ONSET_SEC 到结尾，最多到 POST_END_SEC）
TIME_RANGE          = "all"         # "all" 或 "post"
TREATMENT_ONSET_SEC = 3600.0        # 处理开始时间（秒），即第60分钟
POST_END_SEC        = 9000.0       # 处理后最多到第几秒（180min），不足则取到结尾

# EEG频段 (Hz)
EEG_BANDS = {
    "delta":      (0.5,  4),
    "theta":      (4,    8),
    "alpha":      (8,   13),
    "beta":       (13,  30),
    "low_gamma":  (30,  60),
    "high_gamma": (60, 100),
}

# EMG带通 (Hz)
EMG_BP_LOW  = 10.0
EMG_BP_HIGH = 300.0

# PAC组合（仅方案C）
PAC_PHASE_BANDS = [(4, 8), (8, 13)]
PAC_AMP_BANDS   = [(30, 80), (60, 100)]

# UMAP
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1

# 其他
FS_EEG_DEFAULT  = 2000.0
K_MEANS_N_INIT  = 20
K_MEANS_SEED    = 42
POINT_ALPHA     = 0.4
POINT_SIZE      = 4
DPI             = 150

GROUP_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
]

# ===================================================================
#  信号处理
# ===================================================================

def bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    lo, hi = max(low, 0.01), min(high, nyq * 0.99)
    if lo >= hi:
        return np.zeros_like(data)
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, data)


def band_power_log(data, fs, low, high):
    nperseg = min(len(data), int(fs * 2))
    f, pxx  = welch(data, fs=fs, nperseg=nperseg)
    idx     = (f >= low) & (f <= high)
    power   = float(np.trapz(pxx[idx], f[idx])) if idx.any() else 0.0
    return float(np.log10(power + 1e-12))


def emg_rms_log(sig, fs):
    hi  = min(EMG_BP_HIGH, fs / 2 * 0.99)
    flt = bandpass(sig, EMG_BP_LOW, hi, fs) if hi > EMG_BP_LOW + 5 else sig
    return float(np.log10(np.sqrt(np.mean(flt ** 2)) + 1e-12))


def compute_mi(phase, amplitude, n_bins=18):
    edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx     = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    amp_bin = np.array([amplitude[idx == k].mean() if (idx == k).any() else 0.0
                        for k in range(n_bins)])
    total   = amp_bin.sum()
    if total == 0:
        return 0.0
    p = amp_bin / total
    p = p[p > 0]
    return float((np.log(n_bins) + np.sum(p * np.log(p))) / np.log(n_bins))

# ===================================================================
#  特征构建
# ===================================================================

def build_feature_names(mode):
    names = [f"eeg_{b}" for b in EEG_BANDS] + ["emg_rms"]
    if mode in ("B", "C"):
        names += ["vel_mean", "vel_std"]
    if mode == "C":
        for plo, phi in PAC_PHASE_BANDS:
            for alo, ahi in PAC_AMP_BANDS:
                names.append(f"mi_ph{plo}-{phi}_amp{alo}-{ahi}")
    return names


def extract_features(eeg, emg, vel, fs_eeg, fs_emg, fs_vel, mode,
                     t_start_sec=0.0, t_end_sec=None):
    """
    t_start_sec / t_end_sec: 只提取该时间范围内的窗口（秒，相对录制起点）。
    t_end_sec=None 表示取到结尾。
    """
    step_pts = int((WINDOW_SEC - OVERLAP_SEC) * fs_eeg)
    win_pts  = int(WINDOW_SEC * fs_eeg)
    n        = len(eeg)

    # 裁剪到指定时间范围
    samp_start = int(t_start_sec * fs_eeg)
    samp_end   = int(t_end_sec   * fs_eeg) if t_end_sec is not None else n
    samp_start = max(0, min(samp_start, n))
    samp_end   = max(samp_start, min(samp_end, n))

    t_eeg = np.arange(n) / fs_eeg
    vel_r = np.interp(t_eeg, np.arange(len(vel)) / fs_vel, vel)
    emg_r = np.interp(t_eeg, np.arange(len(emg)) / fs_emg, emg)

    rows, t_cen = [], []
    start = samp_start
    while start + win_pts <= samp_end:
        end   = start + win_pts
        s_eeg = eeg  [start:end]
        s_emg = emg_r[start:end]
        s_vel = vel_r[start:end]

        feat = []
        for lo, hi in EEG_BANDS.values():
            feat.append(band_power_log(s_eeg, fs_eeg, lo, hi))
        feat.append(emg_rms_log(s_emg, fs_eeg))
        if mode in ("B", "C"):
            feat.append(float(np.mean(s_vel)))
            feat.append(float(np.std(s_vel)))
        if mode == "C":
            for plo, phi in PAC_PHASE_BANDS:
                ph_sig = np.angle(hilbert(bandpass(s_eeg, plo, phi, fs_eeg)))
                for alo, ahi in PAC_AMP_BANDS:
                    amp_sig = np.abs(hilbert(bandpass(s_eeg, alo, ahi, fs_eeg)))
                    feat.append(compute_mi(ph_sig, amp_sig))

        rows.append(feat)
        t_cen.append((start + win_pts / 2) / fs_eeg)
        start += step_pts

    return np.array(rows, dtype=np.float32), np.array(t_cen)

# ===================================================================
#  文件加载
# ===================================================================

def load_h5(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())

        def get(candidates):
            for k in candidates:
                if k in f:
                    d = f[k]
                    return (d[0, :] if d.ndim == 2 else d[:]).astype(np.float64)
            raise KeyError(f"Keys not found: {candidates}. Available: {keys}")

        eeg = get(["all_eeg", "eeg"])
        emg = get(["all_emg", "emg"])
        vel = get(["all_v", "velocity"])

        fs_eeg = (float(f["sampling_rate"][()]) if "sampling_rate" in f else
                  float(f["fs_eeg"][()])         if "fs_eeg"        in f else
                  FS_EEG_DEFAULT)
        fs_emg = float(f["fs_emg"][()]) if "fs_emg" in f else fs_eeg
        fs_vel = (float(f["fs_vel"][()]) if "fs_vel" in f else
                  len(vel) / (len(eeg) / fs_eeg))

    return eeg, emg, vel, fs_eeg, fs_emg, fs_vel


def discover_groups(root):
    groups = {}
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        files = sorted(glob.glob(
            os.path.join(entry.path, "**", "*.h5"), recursive=True))
        if files:
            groups[entry.name] = files
    return groups

# ===================================================================
#  K选择
# ===================================================================

def find_optimal_k(X_umap):
    inertias, silhouettes = [], []
    for k in K_RANGE:
        km     = KMeans(n_clusters=k, init="k-means++",
                        n_init=K_MEANS_N_INIT, random_state=K_MEANS_SEED)
        labels = km.fit_predict(X_umap)
        inertias.append(km.inertia_)
        sil = (silhouette_score(X_umap, labels,
                                sample_size=min(5000, len(X_umap)))
               if k > 1 else np.nan)
        silhouettes.append(sil)

    d2      = np.diff(np.diff(inertias))
    k_elbow = list(K_RANGE)[np.argmax(d2) + 1]
    sil_arr = np.array(silhouettes)
    sil_arr[np.isnan(sil_arr)] = -1
    k_sil   = list(K_RANGE)[np.argmax(sil_arr)]
    k_auto  = k_sil

    print(f"  Elbow K={k_elbow}  |  Silhouette K={k_sil}  ->  auto K={k_auto}")
    return list(K_RANGE), inertias, silhouettes, k_auto, k_elbow, k_sil

# ===================================================================
#  绘图
# ===================================================================

def page_k_selection(pdf, k_list, inertias, silhouettes,
                     k_auto, k_elbow, k_sil, forced_k, group_names, n_total):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    final_k   = forced_k if forced_k else k_auto
    note      = f"K={final_k} (forced)" if forced_k else f"K={k_auto} (auto: Silhouette)"
    fig.suptitle(
        f"Optimal K Selection — {note}\n"
        f"Groups: {', '.join(group_names)}  |  "
        f"Total windows: {n_total:,}  |  Window: {WINDOW_SEC}s  |  Mode: {FEATURE_MODE}",
        fontsize=11, fontweight="bold")

    ax = axes[0]
    ax.plot(k_list, inertias, "o-", color="#4E79A7", lw=2)
    ax.axvline(k_elbow, color="#4E79A7", ls=":", lw=1.5, label=f"Elbow K={k_elbow}")
    ax.axvline(final_k, color="red",     ls="--", lw=2,  label=f"Selected K={final_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Method", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    sil_pairs = [(k, s) for k, s in zip(k_list, silhouettes) if not np.isnan(s)]
    ks, ss    = zip(*sil_pairs)
    ax.plot(ks, ss, "s-", color="#E15759", lw=2)
    ax.axvline(k_sil,   color="#E15759", ls=":", lw=1.5, label=f"Best sil K={k_sil}")
    ax.axvline(final_k, color="red",     ls="--", lw=2,  label=f"Selected K={final_k}")
    ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_umap_overview(pdf, umap_2d, cluster_labels, group_labels,
                       group_names, final_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"UMAP Embedding  (K={final_k})\n"
        "Left: by cluster    Right: by group\n"
        "NOTE: All mice pooled into one shared embedding space",
        fontsize=11, fontweight="bold")

    cmap_k = plt.get_cmap("tab10")

    ax = axes[0]
    for k in range(final_k):
        mask = cluster_labels == k
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[cmap_k(k / max(final_k - 1, 1))],
                   s=POINT_SIZE, alpha=POINT_ALPHA,
                   label=f"Cluster {k}  (n={mask.sum():,})", rasterized=True)
    ax.set_title(f"Colored by Cluster (K={final_k})", fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for gi, gname in enumerate(group_names):
        mask = group_labels == gi
        ax.scatter(umap_2d[mask, 0], umap_2d[mask, 1],
                   c=[GROUP_PALETTE[gi % len(GROUP_PALETTE)]],
                   s=POINT_SIZE, alpha=POINT_ALPHA,
                   label=f"{gname}  (n={mask.sum():,})", rasterized=True)
    ax.set_title("Colored by Group", fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3, fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_cluster_characterization(pdf, X_scaled, cluster_labels,
                                   group_labels, group_names,
                                   feat_names, final_k):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Cluster Characterization", fontsize=12, fontweight="bold")

    # 左：特征热图
    ax = axes[0]
    means = np.array([X_scaled[cluster_labels == k].mean(axis=0)
                      for k in range(final_k)])
    im = ax.imshow(means, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(final_k))
    ax.set_yticklabels([f"Cluster {k}" for k in range(final_k)], fontsize=9)
    ax.set_title("Mean Feature Profile (z-score)\nRed=high, Blue=low",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="z-score", shrink=0.8)

    # 右：各cluster的组别占比堆叠柱
    ax = axes[1]
    n_groups = len(group_names)
    data     = np.zeros((final_k, n_groups))
    for k in range(final_k):
        mask  = cluster_labels == k
        total = mask.sum()
        if total == 0:
            continue
        for gi in range(n_groups):
            data[k, gi] = (group_labels[mask] == gi).sum() / total

    bottoms = np.zeros(final_k)
    x       = np.arange(final_k)
    for gi, gname in enumerate(group_names):
        bars = ax.bar(x, data[:, gi], bottom=bottoms,
                      color=GROUP_PALETTE[gi % len(GROUP_PALETTE)],
                      label=gname, edgecolor="white", lw=0.5)
        for k in range(final_k):
            pct = data[k, gi]
            if pct > 0.06:
                ax.text(x[k], bottoms[k] + pct / 2, f"{pct*100:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += data[:, gi]

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Cluster {k}\n(n={int((cluster_labels == k).sum()):,})"
         for k in range(final_k)], fontsize=9)
    ax.set_ylabel("Proportion of windows", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Group Composition per Cluster\n"
                 "(key result: are WT and treatment separated?)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_group_umap_comparison(pdf, umap_2d, cluster_labels, group_labels,
                               group_names, final_k):
    """P4: 各组UMAP并排，颜色按cluster，灰色背景=其他组。"""
    n_groups = len(group_names)
    cmap_k   = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 6), squeeze=False)
    fig.suptitle(
        f"Group UMAP Comparison  (K={final_k})\n"
        "Each panel: one group highlighted  |  Gray = other groups  |  Color = cluster",
        fontsize=12, fontweight="bold")
    for gi, gname in enumerate(group_names):
        ax   = axes[0][gi]
        mask = group_labels == gi
        ax.scatter(umap_2d[~mask, 0], umap_2d[~mask, 1],
                   c="lightgray", s=POINT_SIZE, alpha=0.15, rasterized=True)
        for k in range(final_k):
            mk = mask & (cluster_labels == k)
            if not mk.any(): continue
            ax.scatter(umap_2d[mk, 0], umap_2d[mk, 1],
                       c=[cmap_k(k / max(final_k-1, 1))],
                       s=POINT_SIZE+1, alpha=0.6,
                       label=f"C{k} (n={mk.sum():,})", rasterized=True)
        dist_str = "  ".join([f"C{k}:{(cluster_labels[mask]==k).sum()}"
                               for k in range(final_k) if (cluster_labels[mask]==k).any()])
        ax.set_title(f"{gname}  (n={mask.sum():,})\n{dist_str}", fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
        ax.legend(markerscale=2.5, fontsize=7.5, framealpha=0.75, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_cluster_time_course(pdf, t_centers, cluster_labels,
                              group_labels, group_names, final_k):
    """
    各组在不同cluster的时间占比曲线。
    意义：可以看出处理后某些cluster的占比是否发生变化。
    """
    cmap   = plt.get_cmap("tab10")
    n_grps = len(group_names)
    fig, axes = plt.subplots(n_grps, 1,
                              figsize=(13, 3.5 * n_grps),
                              squeeze=False, sharex=True)
    fig.suptitle(
        "Cluster Occupancy Over Time (per group, all mice pooled)\n"
        "Shows whether cluster composition changes after treatment",
        fontsize=12, fontweight="bold")

    for gi, gname in enumerate(group_names):
        ax   = axes[gi][0]
        mask = group_labels == gi
        t_g  = t_centers[mask] / 60   # -> minutes
        cl_g = cluster_labels[mask]

        order    = np.argsort(t_g)
        t_sorted = t_g [order]
        cl_sort  = cl_g[order]

        bin_size = max(1, len(t_sorted) // 80)
        half     = max(1, bin_size // 2)
        t_bins, occ = [], []
        for i in range(0, len(t_sorted) - bin_size, half):
            seg = cl_sort[i:i + bin_size]
            t_bins.append(t_sorted[i:i + bin_size].mean())
            occ.append([(seg == k).mean() for k in range(final_k)])

        if not t_bins:
            ax.set_title(f"{gname} — no data")
            continue

        t_bins = np.array(t_bins)
        occ    = np.array(occ)

        for k in range(final_k):
            ax.plot(t_bins, occ[:, k], lw=1.5,
                    color=cmap(k / max(final_k - 1, 1)),
                    label=f"Cluster {k}")

        ax.set_ylabel("Proportion", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"Group: {gname}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right", ncol=final_k, framealpha=0.7)

    axes[-1][0].set_xlabel("Time (min)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_per_mouse_umap(pdf, umap_2d, cluster_labels, group_labels,
                        mouse_ids, group_names, final_k):
    """P6: 每只鼠一个子图，按4列排列，超过12只自动分页。
    灰色背景=其他所有鼠，彩色=该鼠按cluster配色。"""
    unique_mice   = list(pd.unique(mouse_ids))
    n_mice        = len(unique_mice)
    cmap          = plt.get_cmap("tab10")
    ncols         = min(4, n_mice)
    mice_per_page = ncols * 3

    for page_start in range(0, n_mice, mice_per_page):
        batch  = unique_mice[page_start: page_start + mice_per_page]
        n_this = len(batch)
        nrows  = int(np.ceil(n_this / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(5.5 * ncols, 4.5 * nrows), squeeze=False)
        fig.suptitle(
            f"Per-Mouse UMAP  (K={final_k})  —  "
            f"Mice {page_start+1}–{min(page_start+mice_per_page, n_mice)} of {n_mice}\n"
            "Gray = all other mice  |  Colored = this mouse (by cluster)",
            fontsize=12, fontweight="bold")

        for idx, mname in enumerate(batch):
            r, c   = divmod(idx, ncols)
            ax     = axes[r][c]
            mask   = mouse_ids == mname
            m_umap = umap_2d[mask]
            m_clus = cluster_labels[mask]
            ax.scatter(umap_2d[~mask, 0], umap_2d[~mask, 1],
                       c="lightgray", s=1, alpha=0.10, rasterized=True)
            for k in range(final_k):
                mk = m_clus == k
                if not mk.any(): continue
                ax.scatter(m_umap[mk, 0], m_umap[mk, 1],
                           c=[cmap(k / max(final_k-1, 1))],
                           s=8, alpha=0.70, label=f"C{k}({mk.sum()})", rasterized=True)
            grp_name = group_names[int(group_labels[mask][0])]
            short    = mname.split("/")[-1]
            dist_str = "  ".join([f"C{k}:{(m_clus==k).sum()}"
                                   for k in range(final_k) if (m_clus==k).any()])
            ax.set_title(f"[{grp_name}] {short}\n{dist_str}", fontsize=8)
            ax.set_xlabel("UMAP-1", fontsize=7); ax.set_ylabel("UMAP-2", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(markerscale=2, fontsize=6, framealpha=0.7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.2)

        for extra in range(n_this, nrows * ncols):
            r, c = divmod(extra, ncols)
            axes[r][c].set_visible(False)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_cluster_spectrogram(pdf, cluster_labels, group_labels,
                              mouse_ids, group_names, t_centers,
                              groups, final_k,
                              freq_max=100):
    """
    每个 Cluster 的平均频谱图（spectrogram heatmap）。

    布局：行=组，列=Cluster（横向排列）。
    横轴=时间（窗口内相对时间，0~WINDOW_SEC秒）
    纵轴=频率（0~freq_max Hz，分辨率0.5Hz）
    颜色=dB 或 z-score（由全局 SPEC_MODE 控制）

    参数：
      nperseg = fs*1s  → 频率分辨率1Hz
      nfft    = fs*2s  → 零填充到0.5Hz间隔
      noverlap= nperseg//2 → 50%重叠，速度比nperseg-1快约240x，对heatmap足够
    dB模式下所有子图共享相同颜色轴（vmin/vmax由全局统计确定）。
    """
    from scipy.signal import spectrogram as sci_spectrogram

    mode = SPEC_MODE   # "dB" 或 "zscore"

    # 建立 mname -> h5路径
    mouse_to_h5 = {}
    for gname, files in groups.items():
        for fpath in files:
            mfolder = os.path.basename(os.path.dirname(fpath))
            mouse_to_h5[f"{gname}/{mfolder}"] = fpath

    # 从第一个可读h5获取fs
    fs_eeg = None
    for h5_path in mouse_to_h5.values():
        try:
            _, _, _, fs_eeg, _, _ = load_h5(h5_path)
            break
        except Exception:
            continue
    if fs_eeg is None:
        print("  [!] 无法读取fs_eeg，跳过spectrogram页")
        return

    # nperseg=1s，nfft=2s（→0.5Hz分辨率），noverlap=50%（速度比nperseg-1快~240x）
    nperseg_sec = 1.0
    nfft_sec    = 2.0
    nperseg  = int(nperseg_sec * fs_eeg)
    nfft     = int(nfft_sec    * fs_eeg)
    noverlap = nperseg // 2

    print(f"  [Spectrogram] mode={mode}, fs={fs_eeg:.0f}Hz, "
          f"nperseg={nperseg}({nperseg_sec}s), nfft={nfft}(→0.5Hz), "
          f"noverlap={noverlap}")

    n_groups   = len(group_names)
    spec_sum   = {k: {gi: None for gi in range(n_groups)} for k in range(final_k)}
    spec_count = {k: {gi: 0    for gi in range(n_groups)} for k in range(final_k)}
    f_axis = None
    t_axis = None

    for mname, h5_path in mouse_to_h5.items():
        try:
            eeg, _, _, fs_m, _, _ = load_h5(h5_path)
        except Exception:
            continue

        gi_arr = group_labels[mouse_ids == mname]
        if len(gi_arr) == 0:
            continue
        gi = int(gi_arr[0])

        m_idx      = np.where(mouse_ids == mname)[0]
        m_times    = t_centers[m_idx]
        m_clus     = cluster_labels[m_idx]

        nperseg_m  = int(nperseg_sec * fs_m)
        nfft_m     = int(nfft_sec    * fs_m)
        noverlap_m = nperseg_m // 2
        win_pts_m  = int(WINDOW_SEC * fs_m)

        for idx, t_cen, k in zip(m_idx, m_times, m_clus):
            half   = win_pts_m // 2
            samp_c = int(t_cen * fs_m)
            samp_s = samp_c - half
            samp_e = samp_s + win_pts_m
            if samp_s < 0 or samp_e > len(eeg):
                continue

            seg = eeg[samp_s:samp_e]
            f, t, Sxx = sci_spectrogram(seg, fs=fs_m,
                                         nperseg=nperseg_m,
                                         nfft=nfft_m,
                                         noverlap=noverlap_m,
                                         scaling='density')

            # 截取到 freq_max
            f_mask = f <= freq_max
            f_cut  = f[f_mask]
            Sxx_cut = Sxx[f_mask, :]

            # 转换为显示值
            if mode == "dB":
                val = 10 * np.log10(Sxx_cut + 1e-12)
            else:
                val = np.log10(Sxx_cut + 1e-12)

            if f_axis is None:
                f_axis = f_cut
                t_axis = t
                for kk in range(final_k):
                    for gg in range(n_groups):
                        spec_sum[kk][gg] = np.zeros_like(val)
            elif val.shape != spec_sum[k][gi].shape:
                continue

            spec_sum[k][gi]   += val
            spec_count[k][gi] += 1

    if f_axis is None:
        print("  [!] 未能计算任何spectrogram，跳过")
        return

    # 计算各(cluster, group)均值
    spec_mean = {k: {} for k in range(final_k)}
    for k in range(final_k):
        for gi in range(n_groups):
            cnt = spec_count[k][gi]
            spec_mean[k][gi] = spec_sum[k][gi] / cnt if cnt > 0 else None

    # 颜色轴：dB模式共享全局vmin/vmax；zscore做全局归一化
    all_valid = [spec_mean[k][gi].ravel()
                 for k in range(final_k)
                 for gi in range(n_groups)
                 if spec_mean[k][gi] is not None]
    all_vals = np.concatenate(all_valid)

    if mode == "dB":
        # 用全局 5th/95th 百分位数作为共享颜色轴，保证强度一致
        vmin_global = float(np.percentile(all_vals, 5))
        vmax_global = float(np.percentile(all_vals, 95))
        cmap        = "viridis"
        cbar_label  = "Power (dB)"

        def to_display(mat):
            return mat   # 已经是dB，直接用

        vmin_use, vmax_use = vmin_global, vmax_global

    else:  # zscore
        g_mean = all_vals.mean()
        g_std  = all_vals.std() + 1e-12
        cmap   = "RdBu_r"
        cbar_label = "z-score"
        vmin_use, vmax_use = -3.0, 3.0

        def to_display(mat):
            return (mat - g_mean) / g_std

    # ── 布局：行=组（heatmap） + 最后一行（组间叠加spectrum），列=Cluster ──
    from matplotlib.gridspec import GridSpec

    # 每组一行heatmap，最后加一行组间叠加spectrum
    n_rows      = n_groups + 1
    row_heights = [3] * n_groups + [1.5]   # heatmap行高3，spectrum行高1.5

    fig = plt.figure(figsize=(4.5 * final_k, 3.0 * n_groups + 2.0))
    gs  = GridSpec(n_rows, final_k,
                   figure=fig,
                   height_ratios=row_heights,
                   hspace=0.35,
                   wspace=0.35)

    fig.suptitle(
        f"Cluster Average Spectrogram  (0–{freq_max} Hz,  Δf=0.5 Hz)\n"
        f"Rows: groups (heatmap)  |  Bottom row: mean spectrum overlay (all groups)  |  {cbar_label}",
        fontsize=12, fontweight="bold")

    # 预计算所有 1D spectrum（时间平均）
    spec_1d = {}
    for gi in range(n_groups):
        spec_1d[gi] = {}
        for k in range(final_k):
            mat = spec_mean[k][gi]
            if mat is not None:
                spec_1d[gi][k] = to_display(mat).mean(axis=1)
            else:
                spec_1d[gi][k] = None

    # 全局 spectrum y轴范围（所有组、所有cluster）
    all_sp_vals = [spec_1d[gi][k]
                   for gi in range(n_groups)
                   for k in range(final_k)
                   if spec_1d[gi][k] is not None]
    if all_sp_vals:
        sp_all = np.concatenate(all_sp_vals)
        sp_ylim_global = (float(np.percentile(sp_all, 1)),
                          float(np.percentile(sp_all, 99)))
    else:
        sp_ylim_global = (vmin_use, vmax_use)

    # 组的颜色（用于折线）
    grp_colors = [plt.get_cmap("Set1")(i / max(n_groups - 1, 1))
                  for i in range(n_groups)]

    for gi, gname in enumerate(group_names):
        for k in range(final_k):
            ax_h = fig.add_subplot(gs[gi, k])
            mat  = spec_mean[k][gi]

            if mat is None:
                ax_h.text(0.5, 0.5, "No data", ha="center", va="center",
                          transform=ax_h.transAxes, fontsize=9)
                ax_h.axis("off")
                ax_h.set_title(f"{gname} — C{k} (no data)", fontsize=9)
                continue

            disp = to_display(mat)
            n    = spec_count[k][gi]

            im = ax_h.pcolormesh(t_axis, f_axis, disp,
                                 cmap=cmap, vmin=vmin_use, vmax=vmax_use,
                                 shading="auto", rasterized=True)
            plt.colorbar(im, ax=ax_h, label=cbar_label, shrink=0.90, pad=0.02)

            for band_name, (blo, bhi) in EEG_BANDS.items():
                if bhi <= freq_max:
                    ax_h.axhline(bhi, color="white", lw=0.6, ls="--", alpha=0.55)
                    ax_h.text(t_axis[-1] * 0.98, (blo + bhi) / 2, band_name,
                              color="white", fontsize=6,
                              va="center", ha="right", alpha=0.80)

            ax_h.set_title(f"{gname} — C{k}  (n={n:,})", fontsize=9)
            ax_h.set_ylabel("Freq (Hz)", fontsize=8)
            ax_h.tick_params(labelsize=7)
            if gi < n_groups - 1:
                ax_h.tick_params(axis="x", labelbottom=False)
            else:
                ax_h.set_xlabel("Time in window (s)", fontsize=8)

    # ── 最后一行：组间叠加 spectrum ───────────────────────────────────
    for k in range(final_k):
        ax_s = fig.add_subplot(gs[n_groups, k])

        for gi, gname in enumerate(group_names):
            sp = spec_1d[gi][k]
            if sp is None:
                continue
            ax_s.plot(f_axis, sp, color=grp_colors[gi],
                      lw=1.4, label=gname, alpha=0.85)

        for _, (blo, bhi) in EEG_BANDS.items():
            if bhi <= freq_max:
                ax_s.axvline(bhi, color="gray", lw=0.5, ls="--", alpha=0.45)

        ax_s.set_xlim(0, freq_max)
        ax_s.set_ylim(sp_ylim_global)
        ax_s.set_xlabel("Frequency (Hz)", fontsize=8)
        ax_s.set_ylabel(cbar_label,       fontsize=7)
        ax_s.tick_params(labelsize=6)
        ax_s.yaxis.set_major_locator(plt.MaxNLocator(3))
        if k == 0:
            ax_s.legend(fontsize=7, framealpha=0.8, loc="upper right")
        ax_s.set_title(f"C{k} — all groups", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def page_markov_matrix(pdf, cluster_labels, group_labels,
                       mouse_ids, group_names, t_centers, final_k):
    """
    P7: 各组Markov转移概率矩阵热图（组内所有鼠合并统计）。
    trans[i,j] = P(next=j | current=i)，按行归一化。
    只统计同一只鼠内相邻时间步的转换，不跨鼠。
    """
    n_groups  = len(group_names)
    fig, axes = plt.subplots(1, n_groups,
                              figsize=(5 * n_groups, 4.8), squeeze=False)
    fig.suptitle(
        "Markov Transition Probability Matrix  (row → col)\n"
        "P(next cluster | current cluster)  |  intra-mouse transitions only",
        fontsize=12, fontweight="bold")

    for gi, gname in enumerate(group_names):
        ax       = axes[0][gi]
        grp_mice = [m for m in pd.unique(mouse_ids)
                    if group_labels[mouse_ids == m][0] == gi]
        trans    = np.zeros((final_k, final_k), dtype=np.float64)

        for mname in grp_mice:
            m_mask = mouse_ids == mname
            seq    = cluster_labels[m_mask][np.argsort(t_centers[m_mask])]
            for t in range(len(seq) - 1):
                trans[seq[t], seq[t+1]] += 1

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob = trans / row_sums

        im = ax.imshow(prob, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Transition probability")

        fs = 8 if final_k <= 7 else 6
        for i in range(final_k):
            for j in range(final_k):
                v = prob[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=fs, color="white" if v > 0.55 else "black")

        ticks = range(final_k)
        labels = [f"C{k}" for k in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Next cluster", fontsize=9)
        ax.set_ylabel("Current cluster", fontsize=9)
        ax.set_title(f"{gname}", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def page_umap_time_series(pdf, umap_2d, cluster_labels, group_labels,
                           mouse_ids, group_names, t_centers, final_k,
                           bin_min=10):
    """
    P8: UMAP时间序列图（游程压缩版）。
    布局：行=组，列=时间窗口（每格 bin_min 分钟，每页最多6列）。

    每个子图中：
      - 灰色背景 = 全体UMAP（所有时间）
      - 圆环节点：颜色=cluster，大小=该cluster连续持续的时间长度
        （将连续相同cluster的窗口压缩为一个节点，位置=重心）
      - 连线+箭头：颜色=鼠，按时间顺序连接各节点，表示状态切换顺序
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch

    def run_length_encode(pts, clus, t_vals, window_sec):
        """
        游程编码：将连续相同cluster的点合并为一个节点。
        返回列表 [(cx, cy, cluster_id, duration_sec), ...]
        """
        nodes = []
        i = 0
        while i < len(clus):
            k = clus[i]
            j = i
            while j < len(clus) and clus[j] == k:
                j += 1
            # 该段：i到j-1，cluster=k
            seg_pts  = pts[i:j]
            cx, cy   = seg_pts.mean(axis=0)
            duration = (j - i) * window_sec   # 秒
            nodes.append((cx, cy, k, duration))
            i = j
        return nodes

    bin_sec   = bin_min * 60.0
    t_min_all = t_centers.min()
    t_max_all = t_centers.max()
    bin_edges = np.arange(t_min_all, t_max_all + bin_sec, bin_sec)
    n_bins    = len(bin_edges) - 1
    if n_bins == 0:
        return

    n_groups    = len(group_names)
    unique_mice = list(pd.unique(mouse_ids))
    cmap_k      = plt.get_cmap("tab10")
    # 用Set2+Dark2混合，与tab10(cluster颜色)视觉差异最大
    _set2  = [plt.get_cmap("Set2")(i)  for i in range(8)]
    _dark2 = [plt.get_cmap("Dark2")(i) for i in range(8)]
    _mouse_palette = _set2 + _dark2   # 16色循环
    mouse_color = {m: _mouse_palette[i % len(_mouse_palette)]
                   for i, m in enumerate(unique_mice)}

    # 节点大小映射：持续时间（秒）→ scatter marker size
    # 1个窗口(WINDOW_SEC) → s=40，线性缩放，最小20最大600
    def dur_to_size(dur):
        s = 40 * (dur / WINDOW_SEC)
        return np.clip(s, 20, 600)

    max_cols_per_page = 6
    for page_start in range(0, n_bins, max_cols_per_page):
        page_bins = range(page_start, min(page_start + max_cols_per_page, n_bins))
        n_cols    = len(page_bins)

        fig, axes = plt.subplots(n_groups, n_cols,
                                  figsize=(4.2 * n_cols, 4.8 * n_groups),
                                  squeeze=False)

        t_label_start = bin_edges[page_start] / 60
        t_label_end   = bin_edges[min(page_start + max_cols_per_page, n_bins)] / 60
        fig.suptitle(
            f"UMAP Time Series  (bin={bin_min} min)  |  "
            f"{t_label_start:.0f}–{t_label_end:.0f} min\n"
            "Ring = cluster stay (size ∝ duration)  |  "
            "Line+arrow = state transition (color = mouse)",
            fontsize=12, fontweight="bold")

        for col_idx, bn in enumerate(page_bins):
            t0       = bin_edges[bn]
            t1       = bin_edges[bn + 1]
            bin_mask = (t_centers >= t0) & (t_centers < t1)

            for gi, gname in enumerate(group_names):
                ax       = axes[gi][col_idx]
                grp_mask = group_labels == gi

                # 灰色背景
                ax.scatter(umap_2d[:, 0], umap_2d[:, 1],
                           c="lightgray", s=1, alpha=0.08, rasterized=True)

                grp_mice = [m for m in unique_mice
                            if group_labels[mouse_ids == m][0] == gi]

                for mname in grp_mice:
                    m_mask = (mouse_ids == mname) & bin_mask
                    if not m_mask.any():
                        continue

                    order  = np.argsort(t_centers[m_mask])
                    pts    = umap_2d[m_mask][order]
                    clus   = cluster_labels[m_mask][order]
                    t_vals = t_centers[m_mask][order]
                    mcolor = mouse_color[mname]

                    # 游程编码 → 节点列表
                    nodes = run_length_encode(pts, clus, t_vals, WINDOW_SEC)
                    if not nodes:
                        continue

                    nxs = np.array([n[0] for n in nodes])
                    nys = np.array([n[1] for n in nodes])
                    nks = np.array([n[2] for n in nodes])
                    nds = np.array([n[3] for n in nodes])

                    # ── 连线（鼠颜色）+ 方向箭头 ──
                    if len(nodes) >= 2:
                        ax.plot(nxs, nys, color=mcolor, lw=1.0,
                                alpha=0.55, zorder=2, solid_capstyle="round")

                        # 每段线中点加箭头
                        arrow_step = max(1, len(nodes) // 5)
                        for ai in range(0, len(nodes) - 1, arrow_step):
                            x0, y0 = nxs[ai],     nys[ai]
                            x1, y1 = nxs[ai + 1], nys[ai + 1]
                            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                            dx, dy = x1 - x0, y1 - y0
                            seg_len = np.hypot(dx, dy)
                            if seg_len < 1e-6:
                                continue
                            scale = 0.15
                            ax.annotate("",
                                xy=(mx + dx * scale, my + dy * scale),
                                xytext=(mx - dx * scale, my - dy * scale),
                                arrowprops=dict(
                                    arrowstyle="->,head_width=0.20,head_length=0.14",
                                    color=mcolor, lw=0.9, alpha=0.80),
                                zorder=3)

                    # ── 圆环节点（cluster颜色，大小=持续时间）──
                    for k in range(final_k):
                        mk = nks == k
                        if not mk.any():
                            continue
                        kcolor = cmap_k(k / max(final_k - 1, 1))
                        sizes  = dur_to_size(nds[mk])
                        # 外圈（鼠颜色边框）
                        ax.scatter(nxs[mk], nys[mk],
                                   s=sizes * 1.5, c=[mcolor],
                                   alpha=0.35, zorder=4,
                                   edgecolors="none", rasterized=True)
                        # 内圈（cluster颜色，半透明填充）
                        ax.scatter(nxs[mk], nys[mk],
                                   s=sizes, c=[kcolor],
                                   alpha=0.60, zorder=5,
                                   edgecolors=mcolor, linewidths=0.8,
                                   rasterized=True)

                t0_min = t0 / 60; t1_min = t1 / 60
                n_pts  = (grp_mask & bin_mask).sum()
                ax.set_title(f"{gname}\n{t0_min:.0f}–{t1_min:.0f} min  (n={n_pts})",
                             fontsize=8, fontweight="bold")
                ax.set_xlabel("UMAP-1", fontsize=7)
                ax.set_ylabel("UMAP-2", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.2)

                # 图例：鼠颜色（第一列）+ cluster颜色（最后一列）
                if col_idx == 0:
                    line_handles = [
                        Line2D([0],[0], color=mouse_color[m], lw=2,
                               label=m.split("/")[-1])
                        for m in grp_mice
                    ]
                    ax.legend(handles=line_handles, fontsize=5.5,
                              framealpha=0.8, loc="upper right",
                              ncol=1, title="Mouse (line)")

                if col_idx == len(page_bins) - 1:
                    clus_handles = [
                        Line2D([0],[0], marker="o", color="w",
                               markerfacecolor=cmap_k(k / max(final_k-1, 1)),
                               markersize=6, label=f"C{k}")
                        for k in range(final_k)
                    ]
                    # 大小图例
                    size_handles = [
                        Line2D([0],[0], marker="o", color="w",
                               markerfacecolor="gray", alpha=0.6,
                               markersize=np.sqrt(dur_to_size(d*WINDOW_SEC)/np.pi)*0.8,
                               label=f"{d} windows")
                        for d in [1, 5, 10]
                    ]
                    ax.legend(handles=clus_handles + size_handles,
                              fontsize=5.5, framealpha=0.8,
                              loc="upper right", ncol=1,
                              title="Cluster / Duration")

        plt.tight_layout(rect=[0, 0, 1, 0.91])
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


# ===================================================================
#  子聚类分析（全频段高活跃 Cluster 的进一步聚类）
# ===================================================================

def find_high_activity_cluster(X_scaled, cluster_labels, feat_names, final_k):
    """
    在所有cluster中，找出 delta/theta/alpha/beta 四个频段z-score均值之和最高的cluster。
    返回 target_cluster_id (int)。
    """
    band_names = ["eeg_delta", "eeg_theta", "eeg_alpha", "eeg_beta"]
    band_idx   = [feat_names.index(b) for b in band_names if b in feat_names]
    if not band_idx:
        # 找不到频段特征时fallback：用所有EEG特征
        band_idx = [i for i, n in enumerate(feat_names) if n.startswith("eeg_")]

    scores = []
    for k in range(final_k):
        mask = cluster_labels == k
        if not mask.any():
            scores.append(-np.inf)
            continue
        mean_z = X_scaled[mask][:, band_idx].mean()
        scores.append(mean_z)
    return int(np.argmax(scores))


def run_subcluster(X_scaled, umap_2d, cluster_labels, target_k):
    """
    对target_k对应的窗口做子聚类：
    1. 子集上重新UMAP（以WT子集为基准fit，其他transform）
    2. 使用 SUBCLUSTER_FORCE_K（UI中用户指定，必填）做KMeans
    返回 (sub_umap_2d, sub_labels, sub_final_k)
    """
    mask  = cluster_labels == target_k
    X_sub = X_scaled[mask]

    try:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=min(30, len(X_sub)-1),
                            min_dist=0.1, random_state=42, verbose=False)
        # 子集内找WT对应行（通过外部 group_labels 和 mouse_ids 无法直接访问，
        # 故此处用 _sub_wt_mask 全局传递；若未设置则fallback到全fit）
        wt_sub_mask = globals().get("_sub_wt_mask", None)
        if wt_sub_mask is not None and wt_sub_mask.sum() > 1:
            reducer.fit(X_sub[wt_sub_mask])
            sub_umap = reducer.transform(X_sub)
        else:
            sub_umap = reducer.fit_transform(X_sub)
    except ImportError:
        pca = PCA(n_components=2, random_state=42)
        wt_sub_mask = globals().get("_sub_wt_mask", None)
        if wt_sub_mask is not None and wt_sub_mask.sum() > 1:
            pca.fit(X_sub[wt_sub_mask])
            sub_umap = pca.transform(X_sub)
        else:
            sub_umap = pca.fit_transform(X_sub)

    km = KMeans(n_clusters=SUBCLUSTER_FORCE_K, init="k-means++",
                n_init=K_MEANS_N_INIT, random_state=K_MEANS_SEED)
    sub_labels = km.fit_predict(sub_umap)

    return sub_umap, sub_labels, SUBCLUSTER_FORCE_K


def sub_page_umap_scatter(pdf, sub_umap, sub_labels, sub_final_k, target_k, n_sub):
    """子聚类PDF P1：子集UMAP散点（按sub-cluster配色），替代K选择图。"""
    cmap_k = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle(
        f"Sub-Cluster UMAP  (source: Cluster {target_k},  n={n_sub:,},  K={sub_final_k})\n"
        f"K={sub_final_k} (user-specified)",
        fontsize=12, fontweight="bold")
    for k in range(sub_final_k):
        mk = sub_labels == k
        ax.scatter(sub_umap[mk, 0], sub_umap[mk, 1],
                   c=[cmap_k(k / max(sub_final_k-1, 1))],
                   s=POINT_SIZE+2, alpha=0.65,
                   label=f"Sub-C{k} (n={mk.sum():,})", rasterized=True)
    ax.set_xlabel("UMAP-1", fontsize=10); ax.set_ylabel("UMAP-2", fontsize=10)
    ax.legend(markerscale=2.5, fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def sub_page_umap_overview(pdf, sub_umap, sub_labels, group_labels_sub,
                            group_names, sub_final_k, target_k):
    """子聚类PDF P2：UMAP总览（按子cluster + 按组）。"""
    cmap_k = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Sub-Cluster UMAP Overview  (source: Cluster {target_k},  K={sub_final_k})",
        fontsize=12, fontweight="bold")

    # 左：按子cluster
    for k in range(sub_final_k):
        mk = sub_labels == k
        axes[0].scatter(sub_umap[mk, 0], sub_umap[mk, 1],
                        c=[cmap_k(k / max(sub_final_k-1, 1))],
                        s=POINT_SIZE+1, alpha=0.6,
                        label=f"Sub-C{k} (n={mk.sum():,})", rasterized=True)
    axes[0].set_title("Colored by sub-cluster"); axes[0].grid(True, alpha=0.2)
    axes[0].legend(markerscale=2.5, fontsize=8, framealpha=0.75)
    axes[0].set_xlabel("UMAP-1"); axes[0].set_ylabel("UMAP-2")

    # 右：按组
    grp_colors = plt.get_cmap("Set1")
    for gi, gname in enumerate(group_names):
        gm = group_labels_sub == gi
        axes[1].scatter(sub_umap[gm, 0], sub_umap[gm, 1],
                        c=[grp_colors(gi / max(len(group_names)-1, 1))],
                        s=POINT_SIZE, alpha=0.55,
                        label=f"{gname} (n={gm.sum():,})", rasterized=True)
    axes[1].set_title("Colored by group"); axes[1].grid(True, alpha=0.2)
    axes[1].legend(markerscale=2.5, fontsize=8, framealpha=0.75)
    axes[1].set_xlabel("UMAP-1"); axes[1].set_ylabel("UMAP-2")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def sub_page_group_comparison(pdf, sub_umap, sub_labels, group_labels_sub,
                               group_names, sub_final_k, target_k):
    """子聚类PDF P3：组间UMAP对比（每组一个子图，按子cluster配色）。"""
    n_groups = len(group_names)
    cmap_k   = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, n_groups, figsize=(7*n_groups, 6), squeeze=False)
    fig.suptitle(
        f"Sub-Cluster Group Comparison  (source: Cluster {target_k},  K={sub_final_k})\n"
        "Each panel: one group highlighted  |  Gray = other groups  |  Color = sub-cluster",
        fontsize=12, fontweight="bold")

    for gi, gname in enumerate(group_names):
        ax   = axes[0][gi]
        mask = group_labels_sub == gi
        ax.scatter(sub_umap[~mask, 0], sub_umap[~mask, 1],
                   c="lightgray", s=POINT_SIZE, alpha=0.15, rasterized=True)
        for k in range(sub_final_k):
            mk = mask & (sub_labels == k)
            if not mk.any(): continue
            ax.scatter(sub_umap[mk, 0], sub_umap[mk, 1],
                       c=[cmap_k(k / max(sub_final_k-1, 1))],
                       s=POINT_SIZE+1, alpha=0.65,
                       label=f"Sub-C{k} (n={mk.sum():,})", rasterized=True)
        dist_str = "  ".join([f"SC{k}:{(sub_labels[mask]==k).sum()}"
                               for k in range(sub_final_k) if (sub_labels[mask]==k).any()])
        ax.set_title(f"{gname}  (n={mask.sum():,})\n{dist_str}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
        ax.legend(markerscale=2.5, fontsize=7.5, framealpha=0.75, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def sub_page_markov_matrix(pdf, sub_labels, group_labels_sub,
                            mouse_ids_sub, group_names, t_centers_sub,
                            sub_final_k, target_k):
    """子聚类PDF P6：组间Markov转移概率矩阵热图。"""
    n_groups  = len(group_names)
    fig, axes = plt.subplots(1, n_groups,
                              figsize=(5 * n_groups, 4.8), squeeze=False)
    fig.suptitle(
        f"Sub-Cluster Markov Transition Matrix  (source: Cluster {target_k},  K={sub_final_k})\n"
        "P(next sub-cluster | current sub-cluster)  |  intra-mouse transitions only",
        fontsize=12, fontweight="bold")

    for gi, gname in enumerate(group_names):
        ax       = axes[0][gi]
        grp_mice = [m for m in pd.unique(mouse_ids_sub)
                    if group_labels_sub[mouse_ids_sub == m][0] == gi]
        trans    = np.zeros((sub_final_k, sub_final_k), dtype=np.float64)

        for mname in grp_mice:
            m_mask = mouse_ids_sub == mname
            seq    = sub_labels[m_mask][np.argsort(t_centers_sub[m_mask])]
            for t in range(len(seq) - 1):
                trans[seq[t], seq[t+1]] += 1

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob = trans / row_sums

        im = ax.imshow(prob, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label="Transition probability")

        fs = 8 if sub_final_k <= 7 else 6
        for i in range(sub_final_k):
            for j in range(sub_final_k):
                v = prob[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=fs, color="white" if v > 0.55 else "black")

        ticks  = range(sub_final_k)
        labels = [f"SC{k}" for k in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Next sub-cluster", fontsize=9)
        ax.set_ylabel("Current sub-cluster", fontsize=9)
        ax.set_title(f"{gname}", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def sub_page_characterization(pdf, X_scaled, sub_labels, group_labels_sub,
                               group_names, feat_names, sub_final_k, target_k):
    """子聚类PDF P4：特征热图 + 组成比例（复用主分析风格）。"""
    page_cluster_characterization(pdf, X_scaled, sub_labels,
                                   group_labels_sub, group_names,
                                   feat_names, sub_final_k)


def sub_page_manual_validation(pdf, sub_umap, sub_labels, group_labels_sub,
                                mouse_ids_sub, group_names, t_centers_sub,
                                groups, sub_final_k, target_k):
    """
    子聚类PDF P5: Manual event验证。
    逻辑与主分析相同，但只在target_k的子集上操作。
    """
    mouse_to_h5 = {}
    for gname, files in groups.items():
        for fpath in files:
            mfolder = os.path.basename(os.path.dirname(fpath))
            mouse_to_h5[f"{gname}/{mfolder}"] = fpath

    n_windows   = len(t_centers_sub)
    event_label = [set() for _ in range(n_windows)]

    for i, mname in enumerate(mouse_ids_sub):
        h5_path = mouse_to_h5.get(mname)
        if h5_path is None: continue
        df_ev = load_manual_events(h5_path)
        if df_ev is None: continue
        t = t_centers_sub[i]
        half_win = WINDOW_SEC / 2.0
        for _, row in df_ev.iterrows():
            t0  = float(row["start_time"]) + TREATMENT_ONSET_SEC
            t1  = float(row["end_time"])   + TREATMENT_ONSET_SEC
            evt = str(row["event_name"]).strip()
            # 要求整个窗口完全在event区间内，避免边界污染
            if (t - half_win >= t0) and (t + half_win <= t1):
                event_label[i].add(evt)

    all_events = sorted({e for s in event_label for e in s})
    if not all_events:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No manual_event.csv found or no windows matched.",
                ha="center", va="center", fontsize=12)
        ax.axis("off")
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        return

    n_groups = len(group_names)
    cmap_k   = plt.get_cmap("tab10")

    for evt_name in all_events:
        evt_mask = np.array([evt_name in s for s in event_label])
        fig, axes = plt.subplots(1, n_groups,
                                  figsize=(7*n_groups, 6), squeeze=False)
        fig.suptitle(
            f'Sub-Cluster Manual Event Validation:  "{evt_name}"  '
            f"(source: Cluster {target_k},  annotated={evt_mask.sum()})\n"
            "Highlighted = annotated windows  |  Color = sub-cluster  |  Gray = others",
            fontsize=12, fontweight="bold")

        for gi, gname in enumerate(group_names):
            ax        = axes[0][gi]
            grp_mask  = group_labels_sub == gi
            highlight = grp_mask & evt_mask
            n_hi      = highlight.sum()
            ax.scatter(sub_umap[~highlight, 0], sub_umap[~highlight, 1],
                       c="lightgray", s=POINT_SIZE, alpha=0.15, rasterized=True)
            for k in range(sub_final_k):
                mk = highlight & (sub_labels == k)
                if not mk.any(): continue
                ax.scatter(sub_umap[mk, 0], sub_umap[mk, 1],
                           c=[cmap_k(k / max(sub_final_k-1, 1))],
                           s=18, alpha=0.85, edgecolors="black", linewidths=0.4,
                           label=f"SC{k} ({mk.sum()})", rasterized=True, zorder=3)
            pct = 100 * n_hi / grp_mask.sum() if grp_mask.sum() > 0 else 0
            ax.set_title(f"{gname}\nannotated: {n_hi} ({pct:.1f}%)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
            ax.legend(markerscale=2, fontsize=7.5, framealpha=0.75,
                      loc="best", ncol=2, title=f'Sub-Cluster ("{evt_name}")')
            ax.grid(True, alpha=0.2)

        plt.tight_layout(rect=[0, 0, 1, 0.88])
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)


def generate_subcluster_pdf(root_folder, X_scaled, X_raw, cluster_labels,
                             group_labels, mouse_ids, group_names,
                             t_centers, feat_names, final_k, groups,
                             time_suffix=""):
    """
    对全频段高活跃cluster做子聚类，生成独立PDF（5页）：
      P1: 子集UMAP散点（K=用户指定）
      P2: UMAP总览（按sub-cluster + 按组）
      P3: 组间对比
      P4: 特征热图+组成比例
      P5: Manual event验证
    """
    print("\n[子聚类] 识别全频段高活跃Cluster...")
    target_k = find_high_activity_cluster(X_scaled, cluster_labels, feat_names, final_k)
    n_sub    = (cluster_labels == target_k).sum()
    print(f"  → Cluster {target_k} 被选中（delta/theta/alpha/beta均值最高，n={n_sub:,}）")
    print(f"  → 子聚类 K={SUBCLUSTER_FORCE_K}（用户指定）")

    print("[子聚类] 运行子聚类（UMAP + KMeans）...")
    mask_sub         = cluster_labels == target_k
    X_sub            = X_scaled[mask_sub]
    group_labels_sub = group_labels[mask_sub]
    mouse_ids_sub    = mouse_ids[mask_sub]
    t_centers_sub    = t_centers[mask_sub]

    # 在子集内标记WT行，供run_subcluster中UMAP fit使用
    wt_gi = next((i for i, n in enumerate(group_names) if n.upper() == "WT"), None)
    import builtins
    if USE_WT_BASIS and wt_gi is not None:
        globals()["_sub_wt_mask"] = (group_labels_sub == wt_gi)
        print(f"  → 子聚类UMAP以WT子集 (n={globals()['_sub_wt_mask'].sum():,}) 为基准")
    else:
        globals()["_sub_wt_mask"] = None
        if not USE_WT_BASIS:
            print("  → 子聚类UMAP使用全体子集 fit（WT基准未启用）")

    sub_umap, sub_labels, sub_final_k = run_subcluster(
        X_scaled, None, cluster_labels, target_k)

    globals().pop("_sub_wt_mask", None)   # 用完清理

    sub_pdf_path = os.path.join(
        root_folder,
        f"subcluster_C{target_k}_K{sub_final_k}{time_suffix}.pdf")

    print(f"[子聚类] 生成PDF → {sub_pdf_path}")
    with PdfPages(sub_pdf_path) as pdf:
        sub_page_umap_scatter(pdf, sub_umap, sub_labels,
                              sub_final_k, target_k, n_sub)
        sub_page_umap_overview(pdf, sub_umap, sub_labels, group_labels_sub,
                               group_names, sub_final_k, target_k)
        sub_page_group_comparison(pdf, sub_umap, sub_labels, group_labels_sub,
                                   group_names, sub_final_k, target_k)
        sub_page_characterization(pdf, X_sub, sub_labels, group_labels_sub,
                                   group_names, feat_names, sub_final_k, target_k)
        sub_page_manual_validation(pdf, sub_umap, sub_labels, group_labels_sub,
                                    mouse_ids_sub, group_names, t_centers_sub,
                                    groups, sub_final_k, target_k)
        sub_page_markov_matrix(pdf, sub_labels, group_labels_sub,
                                mouse_ids_sub, group_names, t_centers_sub,
                                sub_final_k, target_k)

    print(f"[OK] 子聚类PDF: {sub_pdf_path}")
    return sub_pdf_path, target_k, sub_final_k


def load_manual_events(h5_path):
    """加载同目录 manual_event.csv，列：start_time, end_time, event_name（秒）。"""
    csv_path = os.path.join(os.path.dirname(h5_path), "manual_event.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception:
        return None


def page_manual_event_validation(pdf, umap_2d, cluster_labels,
                                  group_labels, mouse_ids, group_names,
                                  t_centers, groups):
    """
    P9: Manual event验证图。
    对每种 event_name 生成一页，各组UMAP并排：
    灰色背景=全体，彩色高亮（黑色轮廓）=手动标注时间段内的窗口，按cluster配色。
    """
    mouse_to_h5 = {}
    for gname, files in groups.items():
        for fpath in files:
            mfolder = os.path.basename(os.path.dirname(fpath))
            mouse_to_h5[f"{gname}/{mfolder}"] = fpath

    n_windows   = len(t_centers)
    event_label = [set() for _ in range(n_windows)]
    half_win = WINDOW_SEC / 2.0   # 窗口半宽（秒）

    for mname, h5_path in mouse_to_h5.items():
        df_ev = load_manual_events(h5_path)
        if df_ev is None: continue
        m_idx   = np.where(mouse_ids == mname)[0]
        m_times = t_centers[m_idx]
        for _, row in df_ev.iterrows():
            # manual_event 时间以"处理开始"为0点
            # t_centers 以"录制起点"为0点，两者差 TREATMENT_ONSET_SEC
            t0  = float(row["start_time"]) + TREATMENT_ONSET_SEC
            t1  = float(row["end_time"])   + TREATMENT_ONSET_SEC
            evt = str(row["event_name"]).strip()
            # 要求整个窗口[t_center-half_win, t_center+half_win]完全在[t0, t1]内
            # 避免窗口边界跨越event边界带来的信号污染
            hits = m_idx[(m_times - half_win >= t0) & (m_times + half_win <= t1)]
            for idx in hits:
                event_label[idx].add(evt)

    all_events = sorted({e for s in event_label for e in s})
    if not all_events:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5,
                "No manual_event.csv found or no windows matched.\n"
                "Ensure manual_event.csv is in each mouse data folder.",
                ha="center", va="center", fontsize=11)
        ax.axis("off")
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)
        return

    n_groups = len(group_names)
    cmap_k   = plt.get_cmap("tab10")
    final_k  = int(cluster_labels.max()) + 1

    for evt_name in all_events:
        evt_mask = np.array([evt_name in s for s in event_label])
        fig, axes = plt.subplots(1, n_groups,
                                  figsize=(7 * n_groups, 6), squeeze=False)
        fig.suptitle(
            f'Manual Event Validation:  "{evt_name}"  '
            f"(annotated windows = {evt_mask.sum()})\n"
            "Highlighted (black outline) = manually annotated  |  "
            "Color = auto cluster  |  Gray = all others",
            fontsize=12, fontweight="bold")
        for gi, gname in enumerate(group_names):
            ax        = axes[0][gi]
            grp_mask  = group_labels == gi
            highlight = grp_mask & evt_mask
            n_hi      = highlight.sum()
            ax.scatter(umap_2d[~highlight, 0], umap_2d[~highlight, 1],
                       c="lightgray", s=POINT_SIZE, alpha=0.15, rasterized=True)
            for k in range(final_k):
                mk = highlight & (cluster_labels == k)
                if not mk.any(): continue
                ax.scatter(umap_2d[mk, 0], umap_2d[mk, 1],
                           c=[cmap_k(k / max(final_k-1, 1))],
                           s=18, alpha=0.85, edgecolors="black", linewidths=0.4,
                           label=f"C{k} ({mk.sum()})", rasterized=True, zorder=3)
            pct = 100 * n_hi / grp_mask.sum() if grp_mask.sum() > 0 else 0
            ax.set_title(f"{gname}\nannotated: {n_hi} windows ({pct:.1f}% of group)",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=9); ax.set_ylabel("UMAP-2", fontsize=9)
            ax.legend(markerscale=2, fontsize=7.5, framealpha=0.75,
                      loc="best", ncol=2, title=f'Cluster ("{evt_name}")')
            ax.grid(True, alpha=0.2)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        pdf.savefig(fig, dpi=DPI, bbox_inches="tight"); plt.close(fig)

def page_manual_event_timeseries(pdf, cluster_labels, group_labels,
                                  mouse_ids, group_names, t_centers, groups):
    """
    Per-mouse manual event × cluster 时序图。
    每只有 manual event 的鼠生成一页。
    横轴统一为分析时间范围（TIME_RANGE决定），相对treatment onset（分钟）。
    两条轨道（共享时间轴）：
      上轨：manual event 色块（每种 event_name 一种颜色）
      下轨：cluster 散点（仅 manual event 覆盖的时间窗口，按 cluster 配色）
    """
    mouse_to_h5 = {}
    for gname, files in groups.items():
        for fpath in files:
            mfolder = os.path.basename(os.path.dirname(fpath))
            mouse_to_h5[f"{gname}/{mfolder}"] = fpath

    half_win = WINDOW_SEC / 2.0
    final_k  = int(cluster_labels.max()) + 1
    cmap_k   = plt.get_cmap("tab10")

    # ── 统一横轴范围（分钟，相对 treatment onset）────────────────────
    if TIME_RANGE == "post":
        xmin_min = 0.0                                          # onset = 0
        xmax_min = (POST_END_SEC - TREATMENT_ONSET_SEC) / 60.0
    else:  # "all"
        xmin_min = -TREATMENT_ONSET_SEC / 60.0                 # 录制起点（负值）
        # 全体鼠中最长录制
        all_tc_min = (t_centers - TREATMENT_ONSET_SEC) / 60.0
        xmax_min = float(all_tc_min.max()) + 1.0

    # 收集所有 event_name，分配颜色
    all_evt_names = set()
    mouse_events  = {}
    for mname, h5_path in mouse_to_h5.items():
        df_ev = load_manual_events(h5_path)
        mouse_events[mname] = df_ev
        if df_ev is not None:
            all_evt_names.update(df_ev["event_name"].astype(str).str.strip().unique())

    all_evt_names = sorted(all_evt_names)
    if not all_evt_names:
        return

    evt_cmap   = plt.get_cmap("Set2")
    evt_colors = {e: evt_cmap(i % 8) for i, e in enumerate(all_evt_names)}

    mice_by_group = {gi: [] for gi in range(len(group_names))}
    for mname in mouse_to_h5:
        if mouse_events[mname] is None:
            continue
        gi_arr = group_labels[mouse_ids == mname]
        if len(gi_arr) == 0:
            continue
        mice_by_group[int(gi_arr[0])].append(mname)

    for gi, gname in enumerate(group_names):
        for mname in sorted(mice_by_group[gi]):
            df_ev    = mouse_events[mname]
            m_idx    = np.where(mouse_ids == mname)[0]
            m_times  = t_centers[m_idx]
            m_clus   = cluster_labels[m_idx]

            in_event       = np.zeros(len(m_idx), dtype=bool)
            events_present = []

            for _, row in df_ev.iterrows():
                t0_abs = float(row["start_time"]) + TREATMENT_ONSET_SEC
                t1_abs = float(row["end_time"])   + TREATMENT_ONSET_SEC
                evt    = str(row["event_name"]).strip()
                hit    = (m_times - half_win >= t0_abs) & (m_times + half_win <= t1_abs)
                in_event |= hit
                events_present.append((t0_abs, t1_abs, evt))

            if not in_event.any():
                continue

            # 转为分钟（相对 treatment onset）
            t_plot = (m_times - TREATMENT_ONSET_SEC) / 60.0

            fig, (ax_ev, ax_cl) = plt.subplots(
                2, 1, figsize=(14, 3.8),
                gridspec_kw={"height_ratios": [1, 2], "hspace": 0.06},
                sharex=True)

            short_name = mname.split("/")[-1]
            fig.suptitle(
                f"Manual Event × Cluster Timeseries  —  {gname} / {short_name}\n"
                f"Top: manual events  |  Bottom: cluster (windows inside events only)",
                fontsize=11, fontweight="bold")

            # ── 上轨：event 色块 ──────────────────────────────────────
            ax_ev.set_ylim(0, 1)
            ax_ev.set_yticks([])
            ax_ev.set_ylabel("Event", fontsize=9, labelpad=4)

            legend_handles = {}
            for t0_abs, t1_abs, evt in events_present:
                t0_m = (t0_abs - TREATMENT_ONSET_SEC) / 60.0
                t1_m = (t1_abs - TREATMENT_ONSET_SEC) / 60.0
                col  = evt_colors[evt]
                rect = plt.Rectangle((t0_m, 0.05), t1_m - t0_m, 0.90,
                                     color=col, alpha=0.75, zorder=2)
                ax_ev.add_patch(rect)
                if t1_m - t0_m > 0.3:
                    ax_ev.text((t0_m + t1_m) / 2, 0.50, evt,
                               ha="center", va="center",
                               fontsize=7, fontweight="bold",
                               color="white", zorder=3, clip_on=True)
                if evt not in legend_handles:
                    legend_handles[evt] = plt.Rectangle(
                        (0, 0), 1, 1, color=col, alpha=0.8)

            ax_ev.legend(legend_handles.values(), legend_handles.keys(),
                         loc="upper right", fontsize=7.5,
                         framealpha=0.85, ncol=min(len(legend_handles), 5))
            ax_ev.set_xlim(xmin_min, xmax_min)
            ax_ev.grid(False)
            ax_ev.spines[["top", "right", "left", "bottom"]].set_visible(False)
            ax_ev.axhline(0, color="lightgray", lw=0.5)

            # treatment onset 参考线
            if TIME_RANGE == "all":
                ax_ev.axvline(0, color="red", lw=0.8, ls="--", alpha=0.6,
                              label="onset")

            # ── 下轨：cluster 散点 ────────────────────────────────────
            ax_cl.set_ylabel("Cluster", fontsize=9)
            ax_cl.set_yticks(range(final_k))
            ax_cl.set_yticklabels([f"C{k}" for k in range(final_k)], fontsize=8)
            ax_cl.set_ylim(-0.6, final_k - 0.4)
            ax_cl.set_xlim(xmin_min, xmax_min)
            ax_cl.grid(axis="x", alpha=0.2)
            ax_cl.grid(axis="y", alpha=0.12)

            for t0_abs, t1_abs, evt in events_present:
                t0_m = (t0_abs - TREATMENT_ONSET_SEC) / 60.0
                t1_m = (t1_abs - TREATMENT_ONSET_SEC) / 60.0
                ax_cl.axvspan(t0_m, t1_m, color=evt_colors[evt],
                              alpha=0.08, zorder=0)

            if TIME_RANGE == "all":
                ax_cl.axvline(0, color="red", lw=0.8, ls="--", alpha=0.6)

            for k in range(final_k):
                mask_k = in_event & (m_clus == k)
                if not mask_k.any():
                    continue
                ax_cl.scatter(t_plot[mask_k],
                              np.full(mask_k.sum(), k),
                              c=[cmap_k(k / max(final_k - 1, 1))],
                              s=22, alpha=0.85, zorder=3,
                              label=f"C{k} (n={mask_k.sum()})",
                              edgecolors="white", linewidths=0.3)

            ax_cl.legend(loc="upper right", fontsize=7, framealpha=0.8,
                         ncol=min(final_k, 6), markerscale=1.4)
            ax_cl.set_xlabel(
                "Time relative to treatment onset (min)", fontsize=9)

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            pdf.savefig(fig, dpi=DPI, bbox_inches="tight")
            plt.close(fig)



def gui_config(root_tk):
    global FEATURE_MODE, WINDOW_SEC, FORCE_K, TIME_RANGE
    global ENABLE_SUBCLUSTER, SUBCLUSTER_FORCE_K, USE_WT_BASIS, SPEC_MODE

    win = tk.Toplevel(root_tk)
    win.title("Clustering 配置")
    win.resizable(False, False)

    # --- 特征方案 ---
    tk.Label(win, text="特征方案", font=("Arial", 11, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(14, 4), padx=20)
    mode_var = tk.StringVar(value=FEATURE_MODE)
    options = [
        ("A", "A：EEG频谱 + EMG RMS"),
        ("B", "B：EEG频谱 + EMG RMS + Velocity  (推荐)"),
        ("C", "C：EEG频谱 + EMG RMS + Velocity + PAC/MI  (慢)"),
    ]
    for i, (val, label) in enumerate(options):
        tk.Radiobutton(win, text=label, variable=mode_var,
                       value=val, anchor="w").grid(
            row=i + 1, column=0, columnspan=2, sticky="w", padx=28)

    # --- 窗口长度 ---
    tk.Label(win, text="窗口长度（秒）", font=("Arial", 10)).grid(
        row=5, column=0, pady=(14, 4), padx=20, sticky="e")
    win_var = tk.IntVar(value=WINDOW_SEC)
    tk.Entry(win, textvariable=win_var, width=8).grid(
        row=5, column=1, sticky="w")

    # --- 时间范围 ---
    tk.Label(win, text="分析时间范围", font=("Arial", 10, "bold")).grid(
        row=6, column=0, columnspan=2, pady=(12, 2), padx=20)
    time_var = tk.StringVar(value=TIME_RANGE)
    tk.Radiobutton(win, text="全部数据（完整录制）",
                   variable=time_var, value="all", anchor="w").grid(
        row=7, column=0, columnspan=2, sticky="w", padx=28)
    tk.Radiobutton(win,
                   text=f"仅处理后数据（{int(TREATMENT_ONSET_SEC)}s 至结尾，最多{int(POST_END_SEC)}s）",
                   variable=time_var, value="post", anchor="w").grid(
        row=8, column=0, columnspan=2, sticky="w", padx=28)

    # --- 强制K ---
    tk.Label(win, text="强制K值（留空=自动）", font=("Arial", 10)).grid(
        row=9, column=0, pady=(12, 4), padx=20, sticky="e")
    k_var = tk.StringVar(value="" if FORCE_K is None else str(FORCE_K))
    tk.Entry(win, textvariable=k_var, width=8).grid(
        row=9, column=1, sticky="w")

    # --- 分割线 + UMAP基准 + Spectrogram模式 ---
    tk.Label(win, text="─" * 42, fg="gray").grid(
        row=10, column=0, columnspan=2, pady=(8, 4))

    wt_var = tk.BooleanVar(value=USE_WT_BASIS)
    tk.Checkbutton(win, text="以 WT 组为基准制作 UMAP 空间（其他组 transform）",
                   variable=wt_var, anchor="w").grid(
        row=11, column=0, columnspan=2, sticky="w", padx=28)

    tk.Label(win, text="Spectrogram 颜色模式", font=("Arial", 10)).grid(
        row=12, column=0, pady=(8, 4), padx=20, sticky="e")
    spec_var = tk.StringVar(value=SPEC_MODE)
    spec_frame = tk.Frame(win)
    spec_frame.grid(row=12, column=1, sticky="w", pady=(8, 4))
    tk.Radiobutton(spec_frame, text="dB（共享颜色轴）",
                   variable=spec_var, value="dB").pack(side="left")
    tk.Radiobutton(spec_frame, text="z-score",
                   variable=spec_var, value="zscore").pack(side="left", padx=(12, 0))

    # --- 子聚类分析开关 ---
    tk.Label(win, text="─" * 42, fg="gray").grid(
        row=13, column=0, columnspan=2, pady=(8, 2))
    tk.Label(win, text="子聚类分析（全频段高活跃 Cluster）",
             font=("Arial", 10, "bold")).grid(
        row=14, column=0, columnspan=2, pady=(2, 4), padx=20)

    sub_var = tk.BooleanVar(value=ENABLE_SUBCLUSTER)
    tk.Checkbutton(win, text="启用子聚类分析（生成独立PDF）",
                   variable=sub_var, anchor="w").grid(
        row=15, column=0, columnspan=2, sticky="w", padx=28)

    tk.Label(win, text="子聚类强制K（留空=自动）", font=("Arial", 10)).grid(
        row=16, column=0, pady=(6, 4), padx=20, sticky="e")
    sub_k_var = tk.StringVar(
        value="" if SUBCLUSTER_FORCE_K is None else str(SUBCLUSTER_FORCE_K))
    tk.Entry(win, textvariable=sub_k_var, width=8).grid(
        row=16, column=1, sticky="w")

    confirmed = [False]

    def on_ok():
        global FEATURE_MODE, WINDOW_SEC, FORCE_K, TIME_RANGE
        global ENABLE_SUBCLUSTER, SUBCLUSTER_FORCE_K, USE_WT_BASIS, SPEC_MODE
        FEATURE_MODE      = mode_var.get()
        TIME_RANGE        = time_var.get()
        ENABLE_SUBCLUSTER = sub_var.get()
        USE_WT_BASIS      = wt_var.get()
        SPEC_MODE         = spec_var.get()
        try:
            WINDOW_SEC = int(win_var.get())
        except ValueError:
            pass
        k_str   = k_var.get().strip()
        FORCE_K = int(k_str) if k_str.isdigit() else None
        sk_str  = sub_k_var.get().strip()
        if ENABLE_SUBCLUSTER and not sk_str.isdigit():
            tk.messagebox.showwarning(
                "子聚类K值未填写",
                "已启用子聚类分析，请在「子聚类强制K」中填写一个整数后再开始。")
            return
        SUBCLUSTER_FORCE_K = int(sk_str) if sk_str.isdigit() else None
        confirmed[0] = True
        win.destroy()

    tk.Button(win, text="  开始分析  ", command=on_ok,
              bg="#4E79A7", fg="white",
              font=("Arial", 11, "bold"),
              padx=10, pady=5).grid(
        row=17, column=0, columnspan=2, pady=16)

    win.grab_set()
    root_tk.wait_window(win)
    return confirmed[0]

# ===================================================================
#  主流程
# ===================================================================

def main():
    # GUI：选文件夹 + 配置
    root_tk = tk.Tk()
    root_tk.withdraw()

    root_folder = filedialog.askdirectory(
        title="选择根文件夹（子文件夹自动识别为组别，如 WT/ Treatment/）")
    if not root_folder:
        print("未选择文件夹，退出。")
        sys.exit(0)

    if not gui_config(root_tk):
        print("用户取消，退出。")
        sys.exit(0)

    root_tk.destroy()

    feat_names = build_feature_names(FEATURE_MODE)
    print("=" * 65)
    print(f"  特征方案 : {FEATURE_MODE}  ({len(feat_names)} 维特征)")
    print(f"  窗口长度 : {WINDOW_SEC}s")
    print(f"  强制K    : {FORCE_K if FORCE_K else '否，自动选择'}")
    print(f"  时间范围 : {'全部数据' if TIME_RANGE == 'all' else f'处理后 {int(TREATMENT_ONSET_SEC)}s–{int(POST_END_SEC)}s'}")
    print("=" * 65)

    # 发现组别
    groups = discover_groups(root_folder)
    if not groups:
        print(f"未找到含.h5文件的子文件夹：{root_folder}")
        sys.exit(0)

    group_names = list(groups.keys())
    print(f"\n发现 {len(group_names)} 个组：")
    for gn, fs in groups.items():
        print(f"  [{gn}]  {len(fs)} 个文件")

    # 提取特征（所有小鼠，跨组联合）
    all_feat, all_gid, all_mid, all_tc = [], [], [], []
    for gi, (gname, files) in enumerate(groups.items()):
        for fpath in files:
            mouse_folder = os.path.basename(os.path.dirname(fpath))
            mname = f"{gname}/{mouse_folder}"
            print(f"  [{gname}] {os.path.basename(fpath)} ...",
                  end=" ", flush=True)
            try:
                eeg, emg, vel, fs_eeg, fs_emg, fs_vel = load_h5(fpath)
                # 确定时间裁剪范围
                total_dur = len(eeg) / fs_eeg
                if TIME_RANGE == "post":
                    t_start = TREATMENT_ONSET_SEC
                    t_end   = min(POST_END_SEC, total_dur)
                    if t_start >= total_dur:
                        print(f"SKIP (录制时长{total_dur:.0f}s < 处理起点{TREATMENT_ONSET_SEC}s)")
                        continue
                else:
                    t_start = 0.0
                    t_end   = total_dur

                feats, tc = extract_features(
                    eeg, emg, vel, fs_eeg, fs_emg, fs_vel, FEATURE_MODE,
                    t_start_sec=t_start, t_end_sec=t_end)
                n = len(feats)
                all_feat.append(feats)
                all_gid.extend( [gi]    * n)
                all_mid.extend( [mname] * n)
                all_tc .extend(tc.tolist())
                range_str = f"{t_start/60:.0f}-{t_end/60:.0f}min"
                print(f"{n} windows  ({range_str})")
            except Exception as e:
                print(f"FAILED -- {e}")
                traceback.print_exc()

    if not all_feat:
        print("无有效数据，退出。")
        sys.exit(1)

    X_raw        = np.vstack(all_feat)
    group_labels = np.array(all_gid,  dtype=np.int32)
    mouse_ids    = np.array(all_mid)
    t_centers    = np.array(all_tc)

    print(f"\n总窗口: {len(X_raw):,}  |  特征维度: {X_raw.shape[1]}")

    # 标准化
    X_scaled = StandardScaler().fit_transform(X_raw)

    # UMAP降维：以WT组为基准fit，其他组transform到同一空间
    print("\n[UMAP] 降维中（以WT组为基准空间）...")
    wt_gi = next((i for i, n in enumerate(group_names) if n.upper() == "WT"), None)
    if USE_WT_BASIS and wt_gi is None:
        print("  [!] 未找到名为'WT'的组，将使用全体数据fit UMAP（回退行为）")
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            n_components=2,
            random_state=42,
            verbose=False)
        if USE_WT_BASIS and wt_gi is not None:
            wt_mask   = group_labels == wt_gi
            n_wt      = wt_mask.sum()
            print(f"  → 以 WT 组 (n={n_wt:,}) 为基准 fit UMAP...")
            reducer.fit(X_scaled[wt_mask])
            print(f"  → transform 全体数据到 WT 空间...")
            umap_2d = reducer.transform(X_scaled)
        else:
            if not USE_WT_BASIS:
                print("  → 使用全体数据 fit UMAP（WT基准未启用）")
            umap_2d = reducer.fit_transform(X_scaled)
        print("  UMAP 完成")
    except ImportError:
        print("  [!] umap-learn 未安装，使用 PCA 代替")
        pca     = PCA(n_components=2, random_state=42)
        if USE_WT_BASIS and wt_gi is not None:
            wt_mask = group_labels == wt_gi
            pca.fit(X_scaled[wt_mask])
            umap_2d = pca.transform(X_scaled)
        else:
            umap_2d = pca.fit_transform(X_scaled)
        print(f"  PCA 解释方差: {pca.explained_variance_ratio_}")

    # K选择
    print("\n[K-means] 计算最优K...")
    k_list, inertias, silhouettes, k_auto, k_elbow, k_sil = find_optimal_k(umap_2d)
    final_k = FORCE_K if FORCE_K else k_auto
    print(f"  最终使用 K={final_k} {'(强制指定)' if FORCE_K else '(Silhouette推荐)'}")

    # 最终聚类
    km = KMeans(n_clusters=final_k, init="k-means++",
                n_init=K_MEANS_N_INIT, random_state=K_MEANS_SEED)
    cluster_labels = km.fit_predict(umap_2d)

    print(f"\n各Cluster分布：")
    for k in range(final_k):
        n_k = (cluster_labels == k).sum()
        pct = 100 * n_k / len(cluster_labels)
        # 各组占比
        grp_str = "  ".join([
            f"{group_names[gi]}:{(group_labels[cluster_labels == k] == gi).sum()}"
            for gi in range(len(group_names))])
        print(f"  Cluster {k}: {n_k:,} ({pct:.1f}%)  [{grp_str}]")

    # 保存CSV
    df_out = pd.DataFrame(X_raw, columns=feat_names)
    df_out.insert(0, "mouse",      mouse_ids)
    df_out.insert(1, "group",      [group_names[g] for g in group_labels])
    df_out.insert(2, "t_center_s", t_centers)
    df_out.insert(3, "umap_1",     umap_2d[:, 0])
    df_out.insert(4, "umap_2",     umap_2d[:, 1])
    df_out.insert(5, "cluster",    cluster_labels)

    time_suffix = "" if TIME_RANGE == "all" else "_post"
    csv_path = os.path.join(
        root_folder, f"clustering_mode{FEATURE_MODE}_K{final_k}{time_suffix}.csv")
    df_out.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[OK] CSV: {csv_path}")

    # 生成PDF（5页）
    pdf_path = os.path.join(
        root_folder, f"clustering_mode{FEATURE_MODE}_K{final_k}{time_suffix}.pdf")
    print("[绘图] 生成PDF（5页）...")
    with PdfPages(pdf_path) as pdf:
        # P1: K选择（Elbow + Silhouette）
        page_k_selection(pdf, k_list, inertias, silhouettes,
                         k_auto, k_elbow, k_sil, FORCE_K,
                         group_names, len(X_raw))
        # P2: UMAP总览
        page_umap_overview(pdf, umap_2d, cluster_labels,
                           group_labels, group_names, final_k)
        # P3: 特征热图 + 组成比例
        page_cluster_characterization(pdf, X_scaled, cluster_labels,
                                       group_labels, group_names,
                                       feat_names, final_k)
        # P3b: 各Cluster平均Spectrogram热图
        print("[绘图] 生成Cluster Spectrogram热图页...")
        page_cluster_spectrogram(pdf, cluster_labels, group_labels,
                                  mouse_ids, group_names, t_centers,
                                  groups, final_k)
        # P4: 组间UMAP对比
        page_group_umap_comparison(pdf, umap_2d, cluster_labels,
                                   group_labels, group_names, final_k)
        # P5: cluster时间占比曲线
        page_cluster_time_course(pdf, t_centers, cluster_labels,
                                  group_labels, group_names, final_k)
        # P6: 每只鼠UMAP
        print("[绘图] 生成每只鼠UMAP页...")
        page_per_mouse_umap(pdf, umap_2d, cluster_labels,
                            group_labels, mouse_ids, group_names, final_k)
        # P7: Markov转移矩阵
        print("[绘图] 生成Markov矩阵页...")
        page_markov_matrix(pdf, cluster_labels, group_labels,
                           mouse_ids, group_names, t_centers, final_k)
        # P8: UMAP时间序列
        print("[绘图] 生成UMAP时间序列页...")
        page_umap_time_series(pdf, umap_2d, cluster_labels, group_labels,
                              mouse_ids, group_names, t_centers, final_k, bin_min=10)
        # P9: manual event验证
        print("[绘图] 生成manual event验证页...")
        page_manual_event_validation(pdf, umap_2d, cluster_labels,
                                      group_labels, mouse_ids, group_names,
                                      t_centers, groups)
        # P10: per-mouse manual event × cluster 时序图
        print("[绘图] 生成per-mouse event×cluster时序图...")
        page_manual_event_timeseries(pdf, cluster_labels, group_labels,
                                      mouse_ids, group_names, t_centers, groups)


    print(f"[OK] PDF: {pdf_path}")
    print(f"\n完成！\n  {csv_path}\n  {pdf_path}")

    # ── 子聚类分析（可选）──────────────────────────────────────────
    if ENABLE_SUBCLUSTER:
        generate_subcluster_pdf(
            root_folder, X_scaled, X_raw, cluster_labels,
            group_labels, mouse_ids, group_names,
            t_centers, feat_names, final_k, groups,
            time_suffix=time_suffix)
    else:
        print("\n[子聚类] 已跳过（UI中未启用）")


if __name__ == "__main__":
    main()