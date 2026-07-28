import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk, filedialog
import os
from scipy.stats import sem
from matplotlib.ticker import MultipleLocator

# -----------------------------
# 1. 选择 CSV 文件
# -----------------------------
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="请选择CSV文件",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not file_path:
    print("未选择文件，已退出。")
    exit()

# -----------------------------
# 2. 读取数据
# -----------------------------
df = pd.read_csv(file_path)

required_cols = {"ROI", "Phase", "F/F0"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"文件中必须包含列: {required_cols}")

# -----------------------------
# 3. 设置 bin 参数
# -----------------------------
bin_min = 0.5
bin_max = 4.0
bin_width = 0.25
bins = np.arange(bin_min, bin_max + bin_width, bin_width)
bin_centers = (bins[:-1] + bins[1:]) / 2

# -----------------------------
# 4. Phase 顺序： 保持文件中出现的顺序（不排序）
# -----------------------------
phases_all = df["Phase"].astype(str).values
# unique preserving order
phases = []
for p in phases_all:
    if p not in phases:
        phases.append(p)
# 只处理前两个 phase（如只需比较特定两相可修改此处）
if len(phases) < 2:
    raise ValueError("文件中需要至少两个不同的 Phase 可供比较。")
ph1, ph2 = phases[0], phases[1]
phases_to_plot = [ph1, ph2]

# -----------------------------
# 5. 创建 3x2 grid
# -----------------------------
plt.style.use("default")
fig, axes = plt.subplots(3, 2, figsize=(11, 12))
axes = axes.flatten()
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# Convenient function to tidy axis
def tidy_ax(ax, x_label=None, y_label=None):
    if x_label:
        ax.set_xlabel(x_label, fontsize=10)
    if y_label:
        ax.set_ylabel(y_label, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

# -----------------------------
# A. 第一行：总体概率分布（合并所有 ROI 的数据）
#    axes[0] = ph1 probability, axes[1] = ph2 probability
# -----------------------------
for idx, phase in enumerate(phases_to_plot):
    subset = df[df["Phase"] == phase]
    counts, _ = np.histogram(subset["F/F0"], bins=bins)
    probs = counts / counts.sum() * 100  # percent

    ax = axes[idx]
    ax.bar(bin_centers, probs, width=bin_width * 0.85, color=colors[idx], alpha=0.85)
    ax.set_title(f"{phase} — Total Probability", fontsize=12)
    tidy_ax(ax, x_label="ΔF/F0", y_label="Probability (%)")
    ax.set_xlim(bin_min - 0.01, bin_max + 0.01)
    # optional fixed ylim (comment/uncomment as needed)
    ax.set_ylim(0, 50)

# -----------------------------
# B. 第二行：基于 ROI 的次数分布（每个 ROI 做直方 -> 求 mean ± SEM）
#    axes[2] = ph1 ROI-counts mean±SEM, axes[3] = ph2 ROI-counts mean±SEM
# -----------------------------
for idx, phase in enumerate(phases_to_plot):
    subset = df[df["Phase"] == phase]
    # group by ROI: for each ROI compute histogram counts across bins
    grouped = subset.groupby("ROI")["F/F0"].apply(lambda x: np.histogram(x, bins=bins)[0])
    # Some ROIs might be empty for this phase -> drop them
    grouped = grouped[grouped.apply(lambda v: np.sum(v) > 0)]
    if len(grouped) == 0:
        raise ValueError(f"No ROI contains events for phase {phase}.")
    counts_all = np.vstack(grouped.values)  # shape: (nROIs, nBins)
    mean_counts = counts_all.mean(axis=0)
    sem_counts = counts_all.std(axis=0, ddof=1) / np.sqrt(counts_all.shape[0])

    ax = axes[2 + idx]
    ax.bar(bin_centers, mean_counts, width=bin_width * 0.85,
           color=colors[idx], alpha=0.85, yerr=sem_counts, capsize=3)
    ax.set_title(f"{phase} — ROI-averaged Counts (Mean ± SEM)", fontsize=12)
    tidy_ax(ax, x_label="ΔF/F0", y_label="Mean counts per ROI")
    ax.set_xlim(bin_min - 0.01, bin_max + 0.01)
    # optional: set ylim or leave auto
    ax.set_ylim(0, None)

# -----------------------------
# C. 第三行左：每个 ROI 的平均 F/F0 比较（bar: mean±SEM across ROIs; gray lines: per-ROI）
#    axes[4] = comparison of mean ΔF/F0 per ROI
# -----------------------------
# compute per-ROI mean F/F0 for each phase
mean_df = df.groupby(["ROI", "Phase"])["F/F0"].mean().unstack()
# keep only ROIs that have both phases
mean_df_pair = mean_df.dropna(subset=phases_to_plot)
if mean_df_pair.shape[0] == 0:
    raise ValueError("没有ROI同时在两个Phase中都有数据。")

means = mean_df_pair[phases_to_plot].mean()
errors = mean_df_pair[phases_to_plot].apply(lambda col: sem(col.dropna()))

ax_cmp = axes[4]
x = np.arange(len(phases_to_plot))
# bars
ax_cmp.bar(x, means.values, yerr=errors.values, color=[colors[0], colors[1]],
          alpha=0.9, capsize=5)
# per-ROI connecting lines
for _, row in mean_df_pair.iterrows():
    ax_cmp.plot(x, row[phases_to_plot].values, color="gray", alpha=0.5, linewidth=0.9)

ax_cmp.set_xticks(x)
ax_cmp.set_xticklabels(phases_to_plot)
ax_cmp.set_title("ROI-wise Mean ΔF/F0 (per ROI) — Mean ± SEM", fontsize=12)
tidy_ax(ax_cmp, x_label="", y_label="Mean ΔF/F0 per ROI")

# -----------------------------
# D. 第三行右：每个 ROI 的“总次数”比较（每个 ROI 在该 phase 中事件数：bar mean±SEM; lines：每ROI的变化）
#    axes[5]
# -----------------------------
# For each ROI, count how many events it has in each phase
count_df = df.groupby(["ROI", "Phase"])["F/F0"].count().unstack()
count_df_pair = count_df.dropna(subset=phases_to_plot)
if count_df_pair.shape[0] == 0:
    raise ValueError("没有ROI同时在两个Phase中都有计数数据。")

count_means = count_df_pair[phases_to_plot].mean()
count_errors = count_df_pair[phases_to_plot].apply(lambda col: sem(col.dropna()))
print(count_df_pair)
ax_cnt = axes[5]
ax_cnt.bar(x, count_means.values, yerr=count_errors.values, color=[colors[0], colors[1]],
          alpha=0.9, capsize=5)
for _, row in count_df_pair.iterrows():
    ax_cnt.plot(x, row[phases_to_plot].values, color="gray", alpha=0.5, linewidth=0.9)

ax_cnt.set_xticks(x)
ax_cnt.set_xticklabels(phases_to_plot)
ax_cnt.set_title("ROI-wise Total Counts — Mean ± SEM", fontsize=12)
tidy_ax(ax_cnt, x_label="", y_label="Total counts per ROI")

# -----------------------------
# Final layout and save
# -----------------------------
plt.tight_layout()
save_path = os.path.splitext(file_path)[0] + ".pdf"
plt.savefig(save_path, bbox_inches="tight", transparent=True)
plt.close()
print(f"✅ 图已保存到：{save_path}")
