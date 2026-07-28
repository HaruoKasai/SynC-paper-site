import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存（Illustrator互換）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import wilcoxon

# ================= 設定 =================
# 解析フォルダ
DATA_DIR = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH"
OUTPUT_DIR = DATA_DIR  # 出力先（同じフォルダに出す場合）

# 対象ファイル名の条件
MUST_INCLUDE = ["M1-Ce", "-20-20s", "end"] #, "-180-180s"
ANY_OF = ["Ce_gamma", "emg_rms", "delta", "Ce_velocity"]


# 時間ウィンドウ: ①と②（平均どうしを結ぶ2点の傾き）
COND1 = {"name": "cond1", "winA": (-20, -10), "winB": (10, 20)}  # ①
COND2 = {"name": "cond2", "winA": ( -4,   0), "winB": ( 0,  4)}  # ②
CONDITIONS = [COND1, COND2]

MIN_POINTS_PER_WIN = 1  # 各ウィンドウ内の最低必要点数
# =======================================

def pick_files(data_dir, must_include, any_of):
    files = [f for f in glob.glob(os.path.join(data_dir, "*.csv"))
             if not os.path.basename(f).startswith("slope_")]
    picked = []
    for f in files:
        base = os.path.basename(f)
        if all(k in base for k in must_include) and any(k in base for k in any_of):
            picked.append(f)
    return sorted(picked)

def detect_time_col(df):
    # time列名の自動判定（"time" を含む列があればそれ、なければ先頭列）
    candidates = [c for c in df.columns if "time" in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def window_mean(time, values, t1, t2):
    mask = (time > t1) & (time < t2)
    if np.sum(mask) < MIN_POINTS_PER_WIN:
        return np.nan, np.nan
    return np.nanmean(values[mask]), np.nanmean(time[mask])

def slope_from_windows(time, values, winA, winB):
    yA, tA = window_mean(time, values, *winA)
    yB, tB = window_mean(time, values, *winB)
    if np.isnan(yA) or np.isnan(yB) or np.isnan(tA) or np.isnan(tB) or tA == tB:
        return np.nan
    return (yB - yA) / (tB - tA)

def analyze_file(path):
    base = os.path.basename(path)
    df_wide = pd.read_csv(path)

    time_col = detect_time_col(df_wide)
    time = df_wide[time_col].values
    subjects = [c for c in df_wide.columns if c != time_col]

    rows = []
    for subj in subjects:
        vals = df_wide[subj].values
        row = {"subject": subj}
        for cond in CONDITIONS:
            row[f"{cond['name']}_slope"] = slope_from_windows(time, vals, cond["winA"], cond["winB"])
        rows.append(row)

    res = pd.DataFrame(rows)

    # Wilcoxon（対応あり）
    c1, c2 = CONDITIONS[0]["name"], CONDITIONS[1]["name"]
    valid = res.dropna(subset=[f"{c1}_slope", f"{c2}_slope"])
    if len(valid) >= 1:
        try:
            stat, pval = wilcoxon(valid[f"{c1}_slope"], valid[f"{c2}_slope"], zero_method="zsplit")
        except TypeError:
            stat, pval = wilcoxon(valid[f"{c1}_slope"], valid[f"{c2}_slope"])
    else:
        stat, pval = np.nan, np.nan

    res["wilcoxon_stat"] = stat
    res["wilcoxon_p"] = pval
    res["n_used"] = len(valid)

    # 出力CSV
    out_csv = os.path.join(OUTPUT_DIR, f"slope_{base}")
    res.to_csv(out_csv, index=False)

    # 図（棒＋個体線）
    plt.figure(figsize=(6, 5))
    mean1, mean2 = valid[f"{c1}_slope"].mean(), valid[f"{c2}_slope"].mean()
    sem1, sem2 = valid[f"{c1}_slope"].sem(),  valid[f"{c2}_slope"].sem()
    plt.bar([0, 1], [mean1, mean2], yerr=[sem1, sem2], capsize=5)
    for _, r in valid.iterrows():
        plt.plot([0, 1], [r[f"{c1}_slope"], r[f"{c2}_slope"]], alpha=0.6)
    plt.xticks([0, 1], [f"{c1} {CONDITIONS[0]['winA']}→{CONDITIONS[0]['winB']}",
                        f"{c2} {CONDITIONS[1]['winA']}→{CONDITIONS[1]['winB']}"])
    plt.ylabel("Slope (Δmean_value / Δmean_time)")
    plt.title(f"{base}\nWilcoxon: n={len(valid)}, stat={stat:.3g}, p={pval:.3g}")
    plt.tight_layout()

    out_pdf = os.path.join(OUTPUT_DIR, f"slope_{base.replace('.csv', '.pdf')}")
    plt.savefig(out_pdf)
    plt.close()

    return out_csv, out_pdf

def main():
    targets = pick_files(DATA_DIR, MUST_INCLUDE, ANY_OF)
    if not targets:
        print("対象CSVが見つかりませんでした。条件やDATA_DIRを確認してください。")
        return
    for f in targets:
        print(f"Analyzing: {os.path.basename(f)}")
        out_csv, out_pdf = analyze_file(f)
        print(f" -> {out_csv}\n -> {out_pdf}")

if __name__ == "__main__":
    main()
