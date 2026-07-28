import os
import re
import glob
import sys
import pathlib

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib as mpl

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../Lib')
sys.path.append(str(current_dir) + '/../IALib')

from ImageJRoiReader import *

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "Arial"


# =============================================================================
# Path
# =============================================================================
# dir_name = r"N:\SynC_invitro_test\EDF1c_fluctuation\Control\zstack"
# ref_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\Control"
# output_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\Control\fluctuation"

dir_name = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1\zstack"
ref_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1"
output_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1\fluctuation"

os.makedirs(output_dir, exist_ok=True)


# =============================================================================
# Time-point settings
#
# [start, end): endは含まない
# 例：[[0, 3], [3, 10]]
# before: ΔV(tp0→1), ΔV(tp1→2), ΔV(tp2→3)
# after : ΔV(tp3→4) ～ ΔV(tp9→10)
# =============================================================================
tp_list = [
    [[0, 3], [3, 10]],
    [[0, 3], [3, 6]],
    [[0, 3], [4, 7]],
    [[0, 3], [5, 8]],
    [[0, 3], [6, 9]]
]

vol_bin = 8
t_list = ["before", "after"]
color_list = ["k", "red"]


# =============================================================================
# Helper functions
# =============================================================================
def safe_mannwhitneyu(before_values, after_values):
    """Pool内のbefore vs afterのΔV分布差。"""
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    before_values = before_values[np.isfinite(before_values)]
    after_values = after_values[np.isfinite(after_values)]

    if len(before_values) == 0 or len(after_values) == 0:
        return np.nan, np.nan

    result = scipy.stats.mannwhitneyu(
        before_values,
        after_values,
        alternative="two-sided"
    )

    return result.statistic, result.pvalue


def safe_brown_forsythe(before_values, after_values):
    """
    Brown-Forsythe test:
    median-centered Levene test.
    """
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    before_values = before_values[np.isfinite(before_values)]
    after_values = after_values[np.isfinite(after_values)]

    if len(before_values) < 2 or len(after_values) < 2:
        return np.nan, np.nan

    result = scipy.stats.levene(
        before_values,
        after_values,
        center="median"
    )

    return result.statistic, result.pvalue


def safe_wilcoxon(before_values, after_values):
    """
    対応ありWilcoxon signed-rank test。
    """
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    valid = np.isfinite(before_values) & np.isfinite(after_values)

    before_values = before_values[valid]
    after_values = after_values[valid]

    if len(before_values) < 2:
        return np.nan, np.nan, len(before_values)

    differences = before_values - after_values

    if np.all(differences == 0):
        return np.nan, np.nan, len(before_values)

    try:
        result = scipy.stats.wilcoxon(
            before_values,
            after_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        return result.statistic, result.pvalue, len(before_values)

    except ValueError:
        return np.nan, np.nan, len(before_values)


def make_paired_plot(
    before_values,
    after_values,
    y_label,
    title,
    output_path,
    wilcoxon_p,
    n_pairs
):
    """
    左：平均 ± SEMの棒グラフと個別点
    右：spineごとの対応線
    """
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    valid = np.isfinite(before_values) & np.isfinite(after_values)

    before_values = before_values[valid]
    after_values = after_values[valid]

    before_mean = np.mean(before_values)
    after_mean = np.mean(after_values)

    if len(before_values) > 1:
        before_sem = scipy.stats.sem(before_values)
        after_sem = scipy.stats.sem(after_values)
    else:
        before_sem = np.nan
        after_sem = np.nan

    fig_pair, axes_pair = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(5.2, 3.5),
        sharey=True
    )

    # -------------------------------------------------------------------------
    # 左：平均 ± SEM
    # -------------------------------------------------------------------------
    axes_pair[0].bar(
        [0, 1],
        [before_mean, after_mean],
        yerr=[before_sem, after_sem],
        color=["white", "white"],
        edgecolor=["black", "red"],
        linewidth=1.2,
        width=0.65,
        capsize=4,
        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2
        }
    )

    # 同じ乱数系列で毎回同じjitter位置にする
    rng = np.random.default_rng(0)

    jitter_before = rng.normal(
        loc=0,
        scale=0.025,
        size=len(before_values)
    )

    jitter_after = rng.normal(
        loc=0,
        scale=0.025,
        size=len(after_values)
    )

    axes_pair[0].scatter(
        np.zeros(len(before_values)) + jitter_before,
        before_values,
        s=10,
        color="black",
        alpha=0.6,
        zorder=3
    )

    axes_pair[0].scatter(
        np.ones(len(after_values)) + jitter_after,
        after_values,
        s=10,
        color="red",
        alpha=0.6,
        zorder=3
    )

    axes_pair[0].set_xticks([0, 1])
    axes_pair[0].set_xticklabels(["Before", "After"])
    axes_pair[0].set_ylabel(y_label)

    # -------------------------------------------------------------------------
    # 右：対応線
    # -------------------------------------------------------------------------
    for before_value, after_value in zip(before_values, after_values):
        axes_pair[1].plot(
            [0, 1],
            [before_value, after_value],
            color="gray",
            linewidth=0.6,
            alpha=0.45,
            zorder=1
        )

    axes_pair[1].scatter(
        np.zeros(len(before_values)),
        before_values,
        s=10,
        color="black",
        alpha=0.7,
        zorder=2
    )

    axes_pair[1].scatter(
        np.ones(len(after_values)),
        after_values,
        s=10,
        color="red",
        alpha=0.7,
        zorder=2
    )

    # 平均値を太線で重ねる
    axes_pair[1].plot(
        [0, 1],
        [before_mean, after_mean],
        color="black",
        linewidth=2.0,
        zorder=3
    )

    axes_pair[1].scatter(
        [0],
        [before_mean],
        s=35,
        color="black",
        zorder=4
    )

    axes_pair[1].scatter(
        [1],
        [after_mean],
        s=35,
        color="red",
        zorder=4
    )

    axes_pair[1].set_xticks([0, 1])
    axes_pair[1].set_xticklabels(["Before", "After"])

    # -------------------------------------------------------------------------
    # 共通設定
    # -------------------------------------------------------------------------
    all_values = np.concatenate([before_values, after_values])

    y_min = np.min(all_values)
    y_max = np.max(all_values)
    y_range = y_max - y_min

    if y_range == 0:
        y_range = 0.1

    y_lower = y_min - y_range * 0.12
    y_upper = y_max + y_range * 0.25

    for ax in axes_pair:
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(y_lower, y_upper)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # ΔV plotだけy=0基準線を表示。
    # SD / log SDでは不要なので、タイトルから判定する。
    if "ΔV" in y_label:
        for ax in axes_pair:
            ax.axhline(
                y=0,
                color="gray",
                ls="dotted",
                lw=0.8,
                zorder=0
            )

    if np.isfinite(wilcoxon_p):
        p_text = f"Wilcoxon p = {wilcoxon_p:.3g}\nn = {n_pairs}"
    else:
        p_text = f"Wilcoxon p = n/a\nn = {n_pairs}"

    axes_pair[1].text(
        0.5,
        y_upper - (y_upper - y_lower) * 0.03,
        p_text,
        ha="center",
        va="top",
        fontsize=9
    )

    fig_pair.suptitle(title, y=1.02, fontsize=10)
    fig_pair.tight_layout()

    fig_pair.savefig(
        output_path,
        dpi=300,
        transparent=True
    )

    plt.close(fig_pair)


# =============================================================================
# Main analysis
# =============================================================================
for b, tp in enumerate(tp_list):

    print("\n========================================================")
    print(f"Analysis {b + 1}/{len(tp_list)}: tp = {tp}")
    print("========================================================")

    # 各行 = spine、各列 = time point
    df_abs_all = pd.DataFrame(
        columns=range(tp[1][1]),
        dtype=float
    )

    df_change_all = pd.DataFrame(
        columns=range(tp[1][1]),
        dtype=float
    )

    img_files = glob.glob(os.path.join(dir_name, "*_binary_sum.tif*"))

    for img in img_files:

        exp_name = os.path.basename(img)[:5]
        ser_name = re.findall(r"series.", os.path.basename(img))[0]

        print(exp_name, ser_name)

        roi_list = glob.glob(
            os.path.join(
                ref_dir,
                "ROI_by_dend",
                "*" + exp_name + "*" + ser_name + "*ROIspine.zip"
            )
        )

        au_csv_list = glob.glob(
            os.path.join(
                ref_dir,
                "timeseries_by_dend",
                "*" + exp_name + "*" + ser_name + "*c1_AU.csv"
            )
        )

        for r, spine_roi in enumerate(roi_list):

            print("####################", spine_roi)

            df = ImageJRoiReader(img, spine_roi)

            df["V_t2"] = (
                df["area_in_pixel"]
                * df["mean_0"]
                * 0.3
                * 0.069
                * 0.069
            )

            df_au = pd.read_csv(au_csv_list[r])

            # -------------------------------------------------------------
            # 全time pointで0のspineを除外
            # -------------------------------------------------------------
            zero_cols = [
                col for col in df_au.columns[:-2]
                if (df_au[col] == 0).all()
            ]

            cols_as_str = df_au.columns.astype(str)

            zero_col_indices = [
                cols_as_str.get_loc(str(c))
                if str(c) in cols_as_str else i
                for i, c in enumerate(df_au.columns)
                if c in zero_cols
            ]

            rows_to_drop = [i - 1 for i in zero_col_indices]

            df_au = df_au.drop(columns=zero_cols)

            df = df.reset_index(drop=True)
            df = df.drop(index=rows_to_drop, errors="ignore")

            print("df.shape =", df.shape)
            print("df_au.shape =", df_au.shape)

            # -------------------------------------------------------------
            # Absolute spine volume time series
            # -------------------------------------------------------------
            df_abs = df_au.iloc[:, 1:-3].div(
                df_au.iloc[2][1:-3],
                axis=1
            )

            df_abs.loc["V_abs_t2"] = df["V_t2"].values

            df_abs = df_abs.T
            df_abs = df_abs * df_abs["V_abs_t2"].values.reshape(-1, 1)
            df_abs = df_abs.drop("V_abs_t2", axis=1)
            df_abs = df_abs.apply(pd.to_numeric, errors="coerce")

            df_abs.to_csv(
                os.path.join(
                    output_dir,
                    exp_name + "_" + ser_name + "_abs.csv"
                ),
                index=True
            )

            # -------------------------------------------------------------
            # Absolute volume time-series plot
            # -------------------------------------------------------------
            df_plot = df_abs.T.copy()

            average = df_abs.mean(axis="index").values.reshape(-1, 1)
            sem = df_abs.sem(axis="index").values.reshape(-1, 1)

            df_plot["min"] = df_au["min"]
            df_plot["average"] = average
            df_plot["sem"] = sem

            fig, axes = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(6, 3)
            )

            ax = df_plot.plot(
                x="min",
                ax=axes[0],
                legend=False,
                linewidth=0.5,
                cmap="bone"
            )

            ax.axvspan(0, 10, color="blue", alpha=0.3, linewidth=0)
            ax.axhline(y=0, color="gray", ls="dotted", lw=1)

            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            ax.plot(
                df_plot["min"],
                df_plot["average"],
                color="red",
                linewidth=1.5
            )

            x = df_plot["min"].values
            y1 = (average - sem).reshape(average.shape[0])
            y2 = (average + sem).reshape(average.shape[0])

            ax.fill_between(
                x,
                y1,
                y2,
                color="red",
                alpha=0.3,
                linewidth=0
            )

            ax.set_xlim(-144, 496)
            ax.set_xticks(np.arange(-120, 490, 120))

            fig.tight_layout()

            output_dir2 = os.path.join(output_dir, "Vol_abs_ts_graph")
            os.makedirs(output_dir2, exist_ok=True)

            fig.savefig(
                os.path.join(
                    output_dir2,
                    exp_name + "_" + ser_name + ".pdf"
                ),
                dpi=300,
                transparent=True
            )

            plt.close(fig)

            # -------------------------------------------------------------
            # ΔV = V(t+1) - V(t)
            # -------------------------------------------------------------
            df_change = df_abs.shift(-1, axis=1) - df_abs
            df_change = df_change.apply(pd.to_numeric, errors="coerce")

            df_abs_all = pd.concat(
                [df_abs_all, df_abs.iloc[:, :tp[1][1]]],
                ignore_index=True
            )

            df_change_all = pd.concat(
                [df_change_all, df_change.iloc[:, :tp[1][1]]],
                ignore_index=True
            )

    # =========================================================================
    # Volume pool calculation
    # =========================================================================
    output = pd.DataFrame(
        columns=[
            "time",
            "bin",
            "n",
            "vol_mean",
            "mu",
            "se",
            "sigma",
            "CI"
        ]
    )

    bin_change_values = {
        "before": {},
        "after": {}
    }

    for t in range(len(tp)):

        epoch_name = t_list[t]

        abs_values = (
            df_abs_all
            .iloc[:, tp[t][0]:tp[t][1]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )

        change_values = (
            df_change_all
            .iloc[:, tp[t][0]:tp[t][1]]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )

        concat = np.concatenate((abs_values, change_values), axis=1)
        concat = concat[np.isfinite(concat).all(axis=1)]
        concat = concat[np.argsort(concat[:, 0])]

        n_total = concat.shape[0]
        n_per_bin = int(n_total / vol_bin)

        print(
            epoch_name,
            "n_total =",
            n_total,
            "n_per_bin =",
            n_per_bin
        )

        for i in range(vol_bin):

            # 余りは元コードと同様に使わない
            concat_bin = concat[
                n_per_bin * i:n_per_bin * (i + 1)
            ]

            volume_bin_values = concat_bin[:, 0]
            change_bin_values = concat_bin[:, 1]

            bin_change_values[epoch_name][i] = change_bin_values.copy()

            vol_mean = np.mean(volume_bin_values)
            mu = np.mean(change_bin_values)
            sigma = np.std(change_bin_values, ddof=0)

            n_bin = len(change_bin_values)
            se = sigma / np.sqrt(n_bin)

            alpha = 0.05
            df_ci = n_bin - 1

            if df_ci > 0:
                t_value = scipy.stats.t.ppf(1 - alpha / 2, df_ci)
                CI = t_value * se
            else:
                CI = np.nan

            new_row = pd.DataFrame([{
                "time": epoch_name,
                "bin": i + 1,
                "n": n_bin,
                "vol_mean": vol_mean,
                "mu": mu,
                "se": se,
                "sigma": sigma,
                "CI": CI
            }])

            output = pd.concat(
                [output, new_row],
                ignore_index=True
            )

    # =========================================================================
    # ① PoolごとのMann–Whitney U test
    # ② PoolごとのBrown–Forsythe test
    # =========================================================================
    stats_rows = []

    for i in range(vol_bin):

        before_values = bin_change_values["before"][i]
        after_values = bin_change_values["after"][i]

        mw_u, mw_p = safe_mannwhitneyu(
            before_values,
            after_values
        )

        bf_stat, bf_p = safe_brown_forsythe(
            before_values,
            after_values
        )

        stats_rows.append({
            "analysis": "poolwise_independent",
            "bin": i + 1,
            "test": "Mann-Whitney U",
            "comparison": "before_vs_after_deltaV",
            "n_before": len(before_values),
            "n_after": len(after_values),
            "before_mean_value": np.mean(before_values),
            "after_mean_value": np.mean(after_values),
            "before_SD_value": np.std(before_values, ddof=0),
            "after_SD_value": np.std(after_values, ddof=0),
            "statistic": mw_u,
            "p_value_raw": mw_p,
            "note": (
                "All pooled deltaV values treated as independent; "
                "two-sided; no multiple-comparison correction."
            )
        })

        stats_rows.append({
            "analysis": "poolwise_independent",
            "bin": i + 1,
            "test": "Brown-Forsythe",
            "comparison": "before_vs_after_deltaV_variance",
            "n_before": len(before_values),
            "n_after": len(after_values),
            "before_mean_value": np.mean(before_values),
            "after_mean_value": np.mean(after_values),
            "before_SD_value": np.std(before_values, ddof=0),
            "after_SD_value": np.std(after_values, ddof=0),
            "statistic": bf_stat,
            "p_value_raw": bf_p,
            "note": (
                "Levene test centered at median; tests difference in variance; "
                "no multiple-comparison correction."
            )
        })

    # =========================================================================
    # ③ Pool分けなし：spine単位の対応ありWilcoxon解析
    #
    # A. 各spineの平均ΔV
    # B. 各spineのΔVのSD（sample SD; ddof=1）
    # C. 各spineのlog(SD)
    # =========================================================================
    before_change_by_spine = (
        df_change_all
        .iloc[:, tp[0][0]:tp[0][1]]
        .apply(pd.to_numeric, errors="coerce")
    )

    after_change_by_spine = (
        df_change_all
        .iloc[:, tp[1][0]:tp[1][1]]
        .apply(pd.to_numeric, errors="coerce")
    )

    # A. spineごとの平均ΔV
    before_spine_mean_deltaV = (
        before_change_by_spine
        .mean(axis=1, skipna=True)
        .to_numpy(dtype=float)
    )

    after_spine_mean_deltaV = (
        after_change_by_spine
        .mean(axis=1, skipna=True)
        .to_numpy(dtype=float)
    )

    # B. spineごとのΔVのSD
    # ddof=1 = sample SD
    before_spine_sd_deltaV = (
        before_change_by_spine
        .std(axis=1, ddof=1, skipna=True)
        .to_numpy(dtype=float)
    )

    after_spine_sd_deltaV = (
        after_change_by_spine
        .std(axis=1, ddof=1, skipna=True)
        .to_numpy(dtype=float)
    )

    # C. SDが0のspineを除外してlog(SD)を作る
    valid_positive_sd = (
        np.isfinite(before_spine_sd_deltaV)
        & np.isfinite(after_spine_sd_deltaV)
        & (before_spine_sd_deltaV > 0)
        & (after_spine_sd_deltaV > 0)
    )

    before_spine_log_sd_deltaV = np.log(
        before_spine_sd_deltaV[valid_positive_sd]
    )

    after_spine_log_sd_deltaV = np.log(
        after_spine_sd_deltaV[valid_positive_sd]
    )

    # -------------------------------------------------------------------------
    # A. Mean ΔV: Wilcoxon
    # -------------------------------------------------------------------------
    wilcoxon_mean_stat, wilcoxon_mean_p, n_pairs_mean = safe_wilcoxon(
        before_spine_mean_deltaV,
        after_spine_mean_deltaV
    )

    valid_mean_pairs = (
        np.isfinite(before_spine_mean_deltaV)
        & np.isfinite(after_spine_mean_deltaV)
    )

    before_spine_mean_valid = before_spine_mean_deltaV[valid_mean_pairs]
    after_spine_mean_valid = after_spine_mean_deltaV[valid_mean_pairs]

    stats_rows.append({
        "analysis": "spinewise_paired",
        "bin": "all_spines",
        "test": "Wilcoxon signed-rank",
        "comparison": "spine_mean_deltaV_before_vs_after",
        "n_before": n_pairs_mean,
        "n_after": n_pairs_mean,
        "before_mean_value": np.mean(before_spine_mean_valid),
        "after_mean_value": np.mean(after_spine_mean_valid),
        "before_SD_value": np.std(before_spine_mean_valid, ddof=1),
        "after_SD_value": np.std(after_spine_mean_valid, ddof=1),
        "statistic": wilcoxon_mean_stat,
        "p_value_raw": wilcoxon_mean_p,
        "note": (
            "Paired test. Each spine contributes one mean deltaV "
            "for before and one mean deltaV for after."
        )
    })

    # -------------------------------------------------------------------------
    # B. SD of ΔV: Wilcoxon
    # -------------------------------------------------------------------------
    wilcoxon_sd_stat, wilcoxon_sd_p, n_pairs_sd = safe_wilcoxon(
        before_spine_sd_deltaV,
        after_spine_sd_deltaV
    )

    valid_sd_pairs = (
        np.isfinite(before_spine_sd_deltaV)
        & np.isfinite(after_spine_sd_deltaV)
    )

    before_spine_sd_valid = before_spine_sd_deltaV[valid_sd_pairs]
    after_spine_sd_valid = after_spine_sd_deltaV[valid_sd_pairs]

    stats_rows.append({
        "analysis": "spinewise_paired",
        "bin": "all_spines",
        "test": "Wilcoxon signed-rank",
        "comparison": "spine_SD_deltaV_before_vs_after",
        "n_before": n_pairs_sd,
        "n_after": n_pairs_sd,
        "before_mean_value": np.mean(before_spine_sd_valid),
        "after_mean_value": np.mean(after_spine_sd_valid),
        "before_SD_value": np.std(before_spine_sd_valid, ddof=1),
        "after_SD_value": np.std(after_spine_sd_valid, ddof=1),
        "statistic": wilcoxon_sd_stat,
        "p_value_raw": wilcoxon_sd_p,
        "note": (
            "Paired test of within-spine sample SD of deltaV (ddof=1). "
            "Interpret primarily when before and after use equal numbers "
            "of time intervals."
        )
    })

    # -------------------------------------------------------------------------
    # C. log(SD of ΔV): Wilcoxon
    # -------------------------------------------------------------------------
    wilcoxon_log_sd_stat, wilcoxon_log_sd_p, n_pairs_log_sd = safe_wilcoxon(
        before_spine_log_sd_deltaV,
        after_spine_log_sd_deltaV
    )

    stats_rows.append({
        "analysis": "spinewise_paired",
        "bin": "all_spines",
        "test": "Wilcoxon signed-rank",
        "comparison": "spine_log_SD_deltaV_before_vs_after",
        "n_before": n_pairs_log_sd,
        "n_after": n_pairs_log_sd,
        "before_mean_value": np.mean(before_spine_log_sd_deltaV),
        "after_mean_value": np.mean(after_spine_log_sd_deltaV),
        "before_SD_value": np.std(before_spine_log_sd_deltaV, ddof=1),
        "after_SD_value": np.std(after_spine_log_sd_deltaV, ddof=1),
        "statistic": wilcoxon_log_sd_stat,
        "p_value_raw": wilcoxon_log_sd_p,
        "note": (
            "Paired test of natural-log-transformed within-spine sample SD "
            "of deltaV. Spines with SD <= 0 are excluded."
        )
    })

    stats_output = pd.DataFrame(stats_rows)

    # =========================================================================
    # Save CSV
    # =========================================================================
    tp_name = (
        "tp"
        + str(tp[0][0]) + "-" + str(tp[0][1])
        + "-" + str(tp[1][0]) + "-" + str(tp[1][1])
    )

    output.to_csv(
        os.path.join(output_dir, "summary_" + tp_name + ".csv"),
        index=False
    )

    stats_output.to_csv(
        os.path.join(output_dir, "stats_" + tp_name + ".csv"),
        index=False
    )

    print("\nSaved:")
    print(os.path.join(output_dir, "summary_" + tp_name + ".csv"))
    print(os.path.join(output_dir, "stats_" + tp_name + ".csv"))

    # =========================================================================
    # Paired plots:
    # 1. spineごとの平均ΔV
    # 2. spineごとのΔVのSD
    # 3. spineごとのlog(ΔVのSD)
    # =========================================================================
    make_paired_plot(
        before_values=before_spine_mean_valid,
        after_values=after_spine_mean_valid,
        y_label="Mean ΔV per spine",
        title="Spine-wise mean ΔV",
        output_path=os.path.join(
            output_dir,
            "spinewise_mean_deltaV_" + tp_name + ".pdf"
        ),
        wilcoxon_p=wilcoxon_mean_p,
        n_pairs=n_pairs_mean
    )

    make_paired_plot(
        before_values=before_spine_sd_valid,
        after_values=after_spine_sd_valid,
        y_label="SD of ΔV per spine",
        title="Spine-wise fluctuation magnitude",
        output_path=os.path.join(
            output_dir,
            "spinewise_SD_deltaV_" + tp_name + ".pdf"
        ),
        wilcoxon_p=wilcoxon_sd_p,
        n_pairs=n_pairs_sd
    )

    make_paired_plot(
        before_values=before_spine_log_sd_deltaV,
        after_values=after_spine_log_sd_deltaV,
        y_label="log(SD of ΔV) per spine",
        title="Spine-wise log fluctuation magnitude",
        output_path=os.path.join(
            output_dir,
            "spinewise_log_SD_deltaV_" + tp_name + ".pdf"
        ),
        wilcoxon_p=wilcoxon_log_sd_p,
        n_pairs=n_pairs_log_sd
    )

    # =========================================================================
    # Volume-pool plot
    # =========================================================================
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(9, 6)
    )

    for t in range(len(tp)):

        data = output.loc[output["time"] == t_list[t]]

        # Mean ΔV
        axes[0][t].scatter(
            data["vol_mean"],
            data["mu"],
            s=5,
            color=color_list[t]
        )

        axes[0][t].errorbar(
            data["vol_mean"],
            data["mu"],
            yerr=data["se"],
            fmt="none",
            ecolor=color_list[t],
            capsize=3
        )

        axes[0][t].set_xlim(0, 0.7)
        axes[0][t].set_ylim(-0.4, 0.4)
        axes[0][t].set_yticks(np.arange(-0.4, 0.4, 0.1))
        axes[0][t].axhline(y=0, color="k", lw=1)

        # SD of ΔV
        axes[1][t].scatter(
            data["vol_mean"],
            data["sigma"],
            s=5,
            color=color_list[t]
        )

        axes[1][t].errorbar(
            data["vol_mean"],
            data["sigma"],
            yerr=data["CI"],
            fmt="none",
            ecolor=color_list[t],
            capsize=3
        )

        axes[1][t].set_xlim(0, 0.7)
        axes[1][t].set_ylim(0, 0.3)
        axes[1][t].set_yticks(np.arange(0, 0.3, 0.05))

        # Overlaid mean ΔV
        axes[0][2].scatter(
            data["vol_mean"],
            data["mu"],
            s=5,
            color=color_list[t]
        )

        axes[0][2].errorbar(
            data["vol_mean"],
            data["mu"],
            yerr=data["se"],
            fmt="none",
            ecolor=color_list[t],
            capsize=3
        )

        axes[0][2].set_xlim(0, 0.7)
        axes[0][2].set_ylim(-0.4, 0.4)
        axes[0][2].set_yticks(np.arange(-0.4, 0.4, 0.1))
        axes[0][2].axhline(y=0, color="k", lw=1)

        # Overlaid SD
        axes[1][2].scatter(
            data["vol_mean"],
            data["sigma"],
            s=5,
            color=color_list[t]
        )

        axes[1][2].errorbar(
            data["vol_mean"],
            data["sigma"],
            yerr=data["CI"],
            fmt="none",
            ecolor=color_list[t],
            capsize=3
        )

        axes[1][2].set_xlim(0, 0.7)
        axes[1][2].set_ylim(0, 0.3)
        axes[1][2].set_yticks(np.arange(0, 0.3, 0.05))

        for i in range(2):
            axes[i][t].spines["right"].set_visible(False)
            axes[i][t].spines["top"].set_visible(False)
            axes[i][2].spines["right"].set_visible(False)
            axes[i][2].spines["top"].set_visible(False)

        # V^(2/3) fitting for SD
        x = data[["vol_mean"]].astype(float)
        y = data["sigma"].astype(float)

        reg = LinearRegression(fit_intercept=False).fit(
            x ** (2 / 3),
            y
        )

        x_plot = np.linspace(0, 0.7, 300)
        y_plot = reg.coef_[0] * x_plot ** (2 / 3) + reg.intercept_

        axes[1][t].plot(
            x_plot,
            y_plot,
            color=color_list[t]
        )

        axes[1][2].plot(
            x_plot,
            y_plot,
            color=color_list[t]
        )

    fig.tight_layout()

    fig.savefig(
        os.path.join(output_dir, "summary_" + tp_name + ".pdf"),
        dpi=300,
        transparent=True
    )

    plt.close(fig)