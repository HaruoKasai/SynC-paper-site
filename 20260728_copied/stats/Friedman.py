import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import shutil
import os
import tkinter.filedialog
import tkinter.messagebox

# ===== ユーザー設定 =====
file_path = tkinter.filedialog.askopenfilename(initialdir=r"\\Synology\zhou\\SynC_Fig\_stats\_raw")  # 対象CSVのパス
alpha = 0.05                                  # 有意水準（表示用）

# ===== 1) 入力読み込み =====
df = pd.read_csv(file_path)
print(df)

# 数値列のみを条件として扱う（必要なら明示的に列名を指定してもOK）
cond_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
if len(cond_cols) < 3:
    raise ValueError("Friedman検定には3条件以上が必要です。数値列が3列以上あるか確認してください。")

# ===== 2) Friedman検定（行=被験体、列=条件） =====
# 全条件で欠損のない行のみを使用
friedman_mat = df[cond_cols].dropna(axis=0, how='any').to_numpy()
if friedman_mat.shape[0] < 2:
    raise ValueError("Friedman検定に十分なサンプル行がありません（欠損を除いた後の行数が足りません）。")

stat, p_friedman = friedmanchisquare(*[friedman_mat[:, j] for j in range(friedman_mat.shape[1])])
friedman_n = friedman_mat.shape[0]

friedman_summary = pd.DataFrame({
    "test": ["Friedman"],
    "k_conditions": [len(cond_cols)],
    "n_rows_used": [friedman_n],
    "statistic": [stat],
    "p_value": [p_friedman],
    "alpha": [alpha]
})

# ===== 3) 事後比較：全ペアのWilcoxon（Holm補正） =====
pairs = []
stats_ = []
pvals = []
ns = []

for i in range(len(cond_cols)):
    for j in range(i+1, len(cond_cols)):
        a_name, b_name = cond_cols[i], cond_cols[j]
        # 当該ペアのどちらかがNaNの行を除外
        pair_df = df[[a_name, b_name]].dropna(axis=0, how='any')
        a = pair_df[a_name].to_numpy()
        b = pair_df[b_name].to_numpy()
        if len(a) < 2:
            # Wilcoxonに必要なサンプル不足
            pairs.append(f"{a_name} vs {b_name}")
            stats_.append(np.nan)
            pvals.append(np.nan)
            ns.append(len(a))
            continue
        # 両側検定（事前方向性なし）
        w_stat, p_raw = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided", mode="auto")
        pairs.append(f"{a_name} vs {b_name}")
        stats_.append(w_stat)
        pvals.append(p_raw)
        ns.append(len(a))

# 多重比較補正（Holm）
# NaNを除いたものだけ補正し、元の順番に戻す
pvals_series = pd.Series(pvals, dtype="float64")
mask_valid = pvals_series.notna()
p_adj = pd.Series(np.nan, index=pvals_series.index, dtype="float64")
if mask_valid.any():
    rej, p_corr, _, _ = multipletests(pvals_series[mask_valid].values, method="holm")
    p_adj.loc[mask_valid] = p_corr
    reject = pd.Series(False, index=pvals_series.index)
    reject.loc[mask_valid] = rej
else:
    reject = pd.Series(False, index=pvals_series.index)

posthoc_df = pd.DataFrame({
    "test": "Wilcoxon (paired, two-sided)",
    "comparison": pairs,
    "n_rows_used": ns,
    "W_statistic": stats_,
    "p_raw": pvals,
    "p_adj_holm": p_adj,
    "reject_at_alpha": reject.astype(bool),
})

save_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), os.path.basename(file_path)[:-8] + ".csv")
# CSVはシートを持てないので、元データの下に空行とマーカー行を挟んで統計結果を追記
with open(save_path, "w", encoding="utf-8", newline="") as f:
    # 元データ
    df.to_csv(f, index=False)
    # 仕切り
    f.write("\n\n#STATS_BLOCK_START\n")
    # Friedman結果
    f.write("#Friedman_test\n")
    friedman_summary.to_csv(f, index=False)
    # Posthoc結果
    f.write("\n#Posthoc_pairwise_Wilcoxon_Holm\n")
    posthoc_df.to_csv(f, index=False)

