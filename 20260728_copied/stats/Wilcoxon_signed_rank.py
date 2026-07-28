import pandas as pd
import numpy as np
from scipy import stats
import itertools
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import os
import matplotlib.pyplot as plt
from scipy.stats import sem
from datetime import datetime
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


graph = True

# ===== GUI準備 =====
root = tk.Tk()
root.withdraw()

# ===== 1) 入力CSVの選択 =====
file_path = tkinter.filedialog.askopenfilename(
    initialdir=r"\\Synology\zhou\SynC_Fig\_stats\_raw",
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)
if not file_path:
    raise SystemExit("ファイルが選択されませんでした。")

df = pd.read_csv(file_path)
print(df.head())

# ===== 2) 数値列のみを対象に =====
all_cols = list(df.columns)
num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
if len(num_cols) < 2:
    raise ValueError("Wilcoxon 検定には少なくとも2つの数値列が必要です。")

# 入力ファイル内の列順を尊重
target_cols = [c for c in all_cols if c in num_cols]

# ===== 3) Wilcoxon 符号付順位検定（全ペア、両側） =====
pairs = []
w_stats = []
p_vals = []
n_used = []

for a, b in itertools.combinations(target_cols, 2):
    # ペアごとにNaNを含む行を落とす（対応の取れたデータにする）
    pair_df = df[[a, b]].dropna(how="any")
    x = pair_df[a].to_numpy()
    y = pair_df[b].to_numpy()


    if graph:
        fig = plt.figure(figsize=(11.69, 8.27))
        gs = gridspec.GridSpec(1, 1)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        ax = fig.add_subplot(gs[0, 0])

        # ==== 平均とSEM ====
        means = [x.mean(), y.mean()]
        errors = [sem(x), sem(y)]
        ax.bar([0, 1], means, yerr=errors, color='none', edgecolor="black", alpha=0.8, width=0.4,
               capsize=2, ecolor='black', error_kw=dict(alpha=0.7, lw=0.8)
               )
        for xi, yi in zip(x, y):
            ax.plot([0, 1], [xi, yi], color='gray', linewidth=0.1, alpha=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([a, b])

        plt.tight_layout()
        plt.legend(fontsize=1, labelspacing=0.1)
        pdf_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), os.path.basename(file_path)[:-8] + ".pdf")
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, dpi=300)
        plt.close(fig)


    if len(x) < 1:
        W = np.nan
        p = np.nan
        n = 0
    else:
        # zero_method="wilcox": 差が0のペアを除外
        # alternative="two-sided", mode="auto"（SciPyが正確検定/近似を自動選択）
        W, p = stats.wilcoxon(x, y, zero_method="wilcox",
                              alternative="two-sided", mode="auto")
        n = len(x)

    pairs.append(f"{a} vs {b}")
    w_stats.append(W)
    p_vals.append(p)
    n_used.append(n)

wilcoxon_df = pd.DataFrame({
    "test": "Wilcoxon signed-rank (two-sided)",
    "comparison": pairs,
    "n_rows_used": n_used,
    "W_statistic": w_stats,
    "p_value": p_vals,
})

# ===== 4) 保存先（file_path[:-8] + ".csv"） =====
save_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), os.path.basename(file_path)[:-8] + ".csv")

# ===== 5) 出力 =====
with open(save_path, "w", encoding="utf-8", newline="") as f:
    # 元データ
    df.to_csv(f, index=False)
    # 仕切り（必要ならコメントアウト外してメタ情報を付ける）
    # f.write("\n\n#STATS_BLOCK_START\n")
    # f.write(f"#Source_File,{file_path}\n")
    # f.write(f"#Saved_At,{datetime.now().isoformat(timespec='seconds')}\n")
    # f.write("#Test_Block:Wilcoxon_Signed_Rank_All_Pairs\n")
    wilcoxon_df.to_csv(f, index=False)

# tkinter.messagebox.showinfo("完了", f"保存しました：\n{save_path}")
print("Done:", save_path)
