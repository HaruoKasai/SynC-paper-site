import numpy as np
import os
import sys
import pandas as pd
import glob
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import pathlib
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.stats import kruskal
from itertools import combinations

# ディレクトリ選択
tk_root = tk.Tk()
tk_root.withdraw()
dir_path = filedialog.askdirectory()

# ディレクトリ内のCSVファイルを取得（_で始まるファイルを除外）
file_paths = {
    os.path.splitext(f)[0]: os.path.join(dir_path, f)
    for f in os.listdir(dir_path) if f.endswith(".csv") and not f.startswith("_")
}

# 各CSVを読み込み
dataframes = {name: pd.read_csv(path, index_col=0) for name, path in file_paths.items()}

# 不要な列を削除し、試行回数ごとの平均と標準誤差を計算
mean_values = {}
std_err_values = {}
n_values ={}
for key, df in dataframes.items():
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    mean_values[key] = df.mean(axis=1)
    std_err_values[key] = df.sem(axis=1)
    n_values[key] = df.shape[1]

# 色の設定（chimerin_ACは赤、他は寒色系）
#colors = {"chimerin_AC": "red", "chimerin_dead_AC": "blue", "chimerin_Veh": "cyan", "No_virus": "navy"}
colors = {"chimerin_AC_rpm": "red", "chimerin_dead_AC_rpm": "blue", "chimerin_Veh_rpm": "cyan", "No_virus_rpm": "navy"}

# 折れ線グラフの作成（標準誤差付き）
plt.figure(figsize=(8, 6))
for key, mean_series in mean_values.items():
    color = colors.get(key, "blue")  # chimerin_ACは赤、それ以外は青系
    plt.errorbar(mean_series.index, mean_series.values, yerr=std_err_values[key], fmt='-o', label=f"{key} (N={n_values[key]})", color=color, capsize=5)

    # strip plotの追加（各試行ごとに重ねる）
    # df = dataframes[key]
    # for i in range(df.shape[0]):  # 各試行（Trialごと）
    #     sns.stripplot(x=[i] * df.shape[1], y=df.iloc[i, :], color=color, jitter=True, alpha=0.5)

plt.xlabel("trial")
#plt.ylabel("spend time (s)")
plt.ylabel("performance (r.p.m)")
plt.title("SynC rotarod")
plt.xticks(ticks=range(len(mean_series)), labels=["trial 1", "trial 2", "trial 3", "trial 4"])
plt.legend()


# PDFに保存
#plot_filename = os.path.join(dir_path, "_group_average_plot.pdf")
plot_filename = os.path.join(dir_path, "_group_r.p.m.average_plot.pdf")
plt.savefig(plot_filename)
plt.show()

# Kruskal-Wallis検定の実施
kruskal_result = kruskal(
    *[df.values.flatten() for df in dataframes.values()]
)
print("Kruskal-Wallis検定結果:")
print(f"統計量: {kruskal_result.statistic}, p値: {kruskal_result.pvalue}")

# 群ごとのデータリスト
group_data = {
    name: df.values.flatten() for name, df in dataframes.items()
}

# 群の組み合わせごとにKruskal-Wallis検定を実施
pairwise_results = {}
for (group1, group2) in combinations(group_data.keys(), 2):
    stat, p_value = kruskal(group_data[group1], group_data[group2])
    pairwise_results[(group1, group2)] = p_value

# 結果をデータフレーム化
pairwise_df = pd.DataFrame(pairwise_results.items(), columns=["Group Pair", "p-value"])

# Bonferroni補正（多重比較補正）
pairwise_df["Adjusted p-value"] = pairwise_df["p-value"] * len(pairwise_results)
pairwise_df["Significant"] = pairwise_df["Adjusted p-value"] < 0.05

# CSVに保存
kruskal_csv_filename = os.path.join(dir_path, "_kruskal_wallis_results.csv")
pairwise_df.to_csv(kruskal_csv_filename, index=False)

# 結果を表示
print("Pairwise Kruskal-Wallis Test Results saved to:", kruskal_csv_filename)
print(pairwise_df)


