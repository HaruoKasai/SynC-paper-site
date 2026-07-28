import matplotlib as mpl
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42  # TrueTypeフォントで保存（Illustrator互換）

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import wilcoxon
import matplotlib.gridspec as gridspec

# ================= 設定 =================
# 解析フォルダ
DATA_DIR = r"X:\Behavior\Openfield_EEG\_Group_Analysis_EEG-EMG_PETH"
OUTPUT_DIR = DATA_DIR  # 出力先（同じフォルダに出す場合）

# 対象ファイル名の条件
# MUST_INCLUDE = ["M1-Ce", "-20-20s"]  # , "-180-180s" "start"
# ANY_OF_1 = ["Ce_gamma", "emg_rms", "Ce_velocity"] #"delta",
# ANY_OF_2 = ["Ctrl_NREM", "SynC_StateC"]
#
# # 時間ウィンドウ: ①と②（平均どうしを結ぶ2点の傾き）
# COND1 = {"name": "cond1", "winA": (-12, -8), "winB": (8, 12)}  # ①
# COND2 = {"name": "cond2", "winA": ( -2,   0), "winB": ( 0,  6)}  # ②
# CONDITIONS = [COND1, COND2]
# AUC_WINDOW = (-20, 0)

# MUST_INCLUDE = ["M1-Ce", "-20-20s"]  # , "-180-180s" "start"
# ANY_OF_1 = ["delta"] #"delta",
# ANY_OF_2 = ["Ctrl_NREM", "SynC_StateC"]
# # 時間ウィンドウ: ①と②（平均どうしを結ぶ2点の傾き）
# COND1 = {"name": "cond1", "winA": (-20, 0), "winB": ( 0,20)}  # ①
# COND2 = {"name": "cond2", "winA": ( -20,   0), "winB": (0,20)}  # ②
# CONDITIONS = [COND1, COND2]
# AUC_WINDOW = (0,20)

# MUST_INCLUDE = ["M1-Ce", "-90-90s"]  # , "-180-180s" "start"
# ANY_OF_1 = ["Ce_gamma", "emg_rms", "Ce_velocity"] #"delta",
# ANY_OF_2 = ["Ctrl_NREM", "SynC_StateC"]
# # 時間ウィンドウ: ①と②（平均どうしを結ぶ2点の傾き）
# COND1 = {"name": "cond1", "winA": (-75, -15), "winB": (15, 75)}  # ①
# COND2 = {"name": "cond2", "winA": ( -2,   0), "winB": ( 0,  6)}  # ②
# CONDITIONS = [COND1, COND2]
# AUC_WINDOW = (-90, 0)

# MUST_INCLUDE = ["M1-Ce", "-20-20s", "end"]  # , "-180-180s" "start"
# ANY_OF_1 = ["Ce_gamma", "emg_rms", "Ce_velocity"] #"delta",
# ANY_OF_2 = ["Ctrl_NREM", "SynC_StateC"]
# COND1 = {"name": "cond1", "winA": (-12, -8), "winB": (8, 12)}  # ①
# COND2 = {"name": "cond2", "winA": ( -6,   0), "winB": ( 0,  2)}  # ②
# CONDITIONS = [COND1, COND2]
# AUC_WINDOW = (0,20)

MUST_INCLUDE = ["M1-Ce", "-20-20s", "end"]  # , "-180-180s" "start"
ANY_OF_1 = ["delta"] #"delta",
ANY_OF_2 = ["Ctrl_NREM", "SynC_StateC"]
COND1 = {"name": "cond1", "winA": (-12, -8), "winB": (8, 12)}  # ①
COND2 = {"name": "cond2", "winA": ( -6,   0), "winB": ( 0,  2)}  # ②
CONDITIONS = [COND1, COND2]
AUC_WINDOW = (-20,0)


MIN_POINTS_PER_WIN = 1  # 各ウィンドウ内の最低必要点数
# =======================================

def pick_files(data_dir, must_include, any_of1, any_of2):
    files = [f for f in glob.glob(os.path.join(data_dir, "*.csv"))
             if not os.path.basename(f).startswith("slope_")]
    picked = []
    for f in files:
        base = os.path.basename(f)
        if all(k in base for k in must_include) and any(k in base for k in any_of1) and any(k in base for k in any_of2):
            picked.append(f)
    return sorted(picked)

def detect_time_col(df):
    # time列名の自動判定（"time" を含む列があればそれ、なければ先頭列）
    candidates = [c for c in df.columns if "time" in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def compute_auc(time, values, win):
    """台形公式でAUCを計算（NaNは線形補間して処理）"""
    t1, t2 = win
    mask = (time >= t1) & (time <= t2)

    t = time[mask]
    v = values[mask]

    if len(t) < 2:
        return np.nan

    # 線形補間で NaN を埋める
    if np.isnan(v).any():
        not_nan = ~np.isnan(v)
        if not_nan.sum() < 2:
            # 補間できる点が足りなければ NaN
            return np.nan
        v = np.interp(t, t[not_nan], v[not_nan])

    return np.trapz(v, t)

def window_mean(time, values, t1, t2):
    mask = (time > t1) & (time < t2)
    t = time[mask]
    v = values[mask]
    if np.sum(mask) < MIN_POINTS_PER_WIN:
        return np.nan, np.nan
    return np.nanmean(v), np.nanmean(t)

def slope_from_windows(time, values, winA, winB):
    yA, tA = window_mean(time, values, *winA)
    yB, tB = window_mean(time, values, *winB)
    if np.isnan(yA) or np.isnan(yB) or np.isnan(tA) or np.isnan(tB) or tA == tB:
        return np.nan
    return (yB - yA) / (tB - tA)

def safe_ratio(num, den):
    # cond2 / cond1 の安全な計算（0やNaNはNaNに）
    if np.isnan(num) or np.isnan(den) or den == 0:
        return np.nan
    return num / den

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

        # 追加: 比率（cond2 / cond1）
        # row["ratio_c2_over_c1"] = safe_ratio(row.get("cond2_slope"), row.get("cond1_slope"))
        row["ratio_c2_over_c1"] = row.get("cond2_slope") / row.get("cond1_slope")
        # row["ratio_c2_over_c1"] = (row.get("cond2_slope") / row.get("cond1_slope") -1)*100
        # row["ratio_c2_over_c1"] = row.get("cond2_slope") - row.get("cond1_slope")
        # row["ratio_c2_over_c1"] =(row.get("cond2_slope")-row.get("cond1_slope")) / (row.get("cond2_slope") + row.get("cond1_slope"))
        # row["ratio_c2_over_c1"] = (row.get("cond2_slope") - row.get("cond1_slope")) / abs(row.get("cond1_slope"))
        # row["ratio_c2_over_c1"] = (row.get("cond2_slope") - row.get("cond1_slope")) / (abs(row.get("cond1_slope")) + abs(row.get("cond2_slope")))

        row["AUC"] = compute_auc(time, vals, AUC_WINDOW)

        print("AUC", row["AUC"] )

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

    # 出力CSV（元の + 比率列も含む）
    out_csv = os.path.join(OUTPUT_DIR, f"slope_{base}")
    res.to_csv(out_csv, index=False)

    # 図（棒＋個体線：各条件スロープ）
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

    # ===== 追加: 比率の棒グラフ（平均±SEM）＋各個体散布 =====
    ratio_valid = res.dropna(subset=["ratio_c2_over_c1"])
    auc_valid = res.dropna(subset=["AUC"])
    print(auc_valid)
    out_ratio_pdf = os.path.join(OUTPUT_DIR, f"slope_ratio_{base.replace('.csv', '.pdf')}")
    if len(ratio_valid) > 0:
        ratio_mean = ratio_valid["ratio_c2_over_c1"].mean()
        ratio_sem  = ratio_valid["ratio_c2_over_c1"].sem()

        auc_mean = auc_valid["AUC"].mean()
        auc_sem = auc_valid["AUC"].sem()

        fig = plt.figure(figsize=(4.5, 5))
        gs = gridspec.GridSpec(1,2)
        ax0,ax1 = [fig.add_subplot(gs[0, i]) for i in range(2)]

        ax0.bar([0], [ratio_mean], yerr=[ratio_sem], capsize=5)
        # 個体値の散布（横ジッター）
        x = np.zeros(len(ratio_valid))
        jitter = (np.random.rand(len(ratio_valid)) - 0.5) * 0.1
        ax0.scatter(x + jitter, ratio_valid["ratio_c2_over_c1"],
                    alpha=0.7, marker="o", edgecolors="none")

        ax1.bar([0], [auc_mean], yerr=[auc_sem], capsize=5)
        # 個体値の散布（横ジッター）
        x1 = np.zeros(len(auc_valid))
        jitter = (np.random.rand(len(auc_valid)) - 0.5) * 0.1
        ax1.scatter(x1 + jitter, auc_valid["AUC"],
                    alpha=0.7, marker="o", edgecolors="none")



        # plt.xticks([0], ["cond2/cond1"])
        # plt.ylabel("Slope ratio (cond2 / cond1)")
        # plt.title(f"{base}\nRatio: n={len(ratio_valid)}, mean={ratio_mean:.3g}, SEM={ratio_sem:.3g}")
        # plt.axhline(1.0, linestyle="--", linewidth=1)  # 1をガイドとして
        plt.tight_layout()
        plt.savefig(out_ratio_pdf)
        plt.close()
    else:
        out_ratio_pdf = "NA"

    # 追加: 比率だけの簡易CSVも保存（必要に応じて）
    out_ratio_csv = os.path.join(OUTPUT_DIR, f"slope_ratio_only_{base}")
    # res[["subject", "ratio_c2_over_c1"]].to_csv(out_ratio_csv, index=False)

    return out_csv, out_pdf, out_ratio_csv, out_ratio_pdf

def main():
    targets = pick_files(DATA_DIR, MUST_INCLUDE, ANY_OF_1, ANY_OF_2)
    if not targets:
        print("対象CSVが見つかりませんでした。条件やDATA_DIRを確認してください。")
        return
    for f in targets:
        print(f"Analyzing: {os.path.basename(f)}")
        out_csv, out_pdf, out_ratio_csv, out_ratio_pdf = analyze_file(f)
        print(f" -> {out_csv}\n -> {out_pdf}\n -> {out_ratio_csv}\n -> {out_ratio_pdf}")

if __name__ == "__main__":
    main()
