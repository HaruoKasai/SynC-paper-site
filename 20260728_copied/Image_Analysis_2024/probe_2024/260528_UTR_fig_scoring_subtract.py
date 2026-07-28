import os
import glob
import json
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture


rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

# ============================================================
# 解析対象と出力先
# ============================================================
# INPUT_ROOT配下の各probeフォルダからcontrol spine/back CSVを読み込み、
# 同じ場所に中間CSV、score CSV、summary PDFを保存する。
INPUT_ROOT = r"\\Synology\arima\Probe_paper_2023\dissociate\UTR_screening_data"
OUTPUT_ROOT = INPUT_ROOT

# ============================================================
# score列の定義
# ============================================================
# RAW_SCORE_COL:
#   cellごとにbaseline modeを引いたAS/filler ratio。
#
# CENTERED_SCORE_COL:
#   RAW_SCORE_COLからprobeごとのmedianを引いた値。
#   medianで割ると、medianが低いprobeが人工的に高評価される可能性があるため、
#   ここでは「割る」ではなく「引く」ことでbaseline位置だけをそろえる。
RAW_SCORE_COL = "subtraction_mode_AS_filler_ratio"
CENTERED_SCORE_COL = "subtraction_mode_AS_filler_ratio_probe_median_centered"


def gaussian(x, a, b, c):
    # Base populationを近似するための単峰性Gaussian。
    return a * np.exp(-((x - b) ** 2) / (2 * c ** 2))


def calculate_as_score(spine_csv, back_csv, cell_count, cond, probe_name):
    # ============================================================
    # 1 cell分の基本量を作る
    # ============================================================
    # spine CSVとback CSVから、背景差し引き後のfiller量、mVenus量、
    # AS/filler ratioを計算する。
    # この段階ではglobal scoreはまだ作らない。
    out_dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)

    spine_df = pd.read_csv(spine_csv)
    spine_df = spine_df[spine_df["dendrite"].notna()].reset_index(drop=True)
    back_df = pd.read_csv(back_csv)[:len(spine_df)]

    area_column = "area" if "area" in spine_df.columns else "area_in_pixel"
    area = spine_df[area_column].values

    label = spine_df["label"]
    dendrite = spine_df["dendrite"]

    filler_sum = (
        spine_df["mean_intensity-0"].values
        - back_df["mean_intensity-0"].values
    ) * area

    mvenus_sum = (
        spine_df["mean_intensity-1"].values
        - back_df["mean_intensity-1"].values
    ) * area

    valid_indices = (filler_sum > 0) & (mvenus_sum > 0)

    filler_sum = filler_sum[valid_indices]
    mvenus_sum = mvenus_sum[valid_indices]
    label = label[valid_indices]
    dendrite = dendrite[valid_indices]

    as_filler_ratio = mvenus_sum / filler_sum

    kde = gaussian_kde(as_filler_ratio)
    x_grid = np.linspace(as_filler_ratio.min(), as_filler_ratio.max(), 1000)
    kde_values = kde(x_grid)
    peaks, _ = find_peaks(kde_values)

    if len(peaks) > 0:
        mode_as_filler_ratio = x_grid[peaks].min()
    else:
        mode_as_filler_ratio = x_grid[np.argmax(kde_values)]

    as_filler_ratio_normalized = as_filler_ratio / mode_as_filler_ratio
    filler_normalized = filler_sum / np.mean(filler_sum)

    sd_cell = (
        (mode_as_filler_ratio - as_filler_ratio.min()) / mode_as_filler_ratio
        if mode_as_filler_ratio != 0
        else np.nan
    )

    z_score_cell = as_filler_ratio_normalized / sd_cell

    df = pd.DataFrame({
        "spine_label": label.values.flatten(),
        "AS/filler": as_filler_ratio,
        "AS/filler_normalized": as_filler_ratio_normalized,
        "filler sum": filler_sum,
        "V_normalized": filler_normalized,
        "mVenus sum": mvenus_sum,
        "probe": probe_name,
        "condition": cond,
        "cell": [f"cell{cell_count}"] * len(as_filler_ratio),
        "mode_AS_filler_ratio": mode_as_filler_ratio,
        "SD_cell": sd_cell,
        "z-score_cell": z_score_cell,
        "dendrite": dendrite.values,
    })

    as_csv_path = os.path.join(out_dir, f"AS_filler_ratio_{fname[:-9]}.csv")
    df.to_csv(as_csv_path, index=False)

    return df


def build_all_data(input_root):
    # ============================================================
    # 全probeのcontrolデータを1つのDataFrameにまとめる
    # ============================================================
    # 各probeフォルダを巡回し、controlを含むspine/back CSVだけを読み込む。
    # 出力はspine単位のdf_all。
    probe_dirs = glob.glob(os.path.join(input_root, "[!_]*"))
    df_all = pd.DataFrame()

    for probe_dir in probe_dirs:
        probe_name = os.path.basename(probe_dir)
        data_dir = os.path.join(probe_dir, "violin_plot_box_plot")

        spine_csv_files = glob.glob(os.path.join(data_dir, "*spine.csv"))
        back_csv_files = [
            spine_csv[:-9] + "back.csv"
            for spine_csv in spine_csv_files
        ]

        print(probe_name)
        print(f"Number of files: {len(spine_csv_files)}")

        ctrl_count = 0

        for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
            if "control" not in spine_csv.lower():
                continue

            ctrl_count += 1
            df = calculate_as_score(
                spine_csv=spine_csv,
                back_csv=back_csv,
                cell_count=ctrl_count,
                cond="Ctrl",
                probe_name=probe_name,
            )
            df_all = pd.concat([df_all, df], ignore_index=True)

    return df_all


def add_across_cell_filler_normalization(df_all):
    # ============================================================
    # cell間のfiller強度差を補正する
    # ============================================================
    # cellごとのfiller signal量の違いによるAS/filler ratioのずれを減らす。
    # across_spine_fillerとAS/filler_ratio_filler_normalizedを作る。
    unique_filler_mean = (
        df_all.groupby(["probe", "cell"])["filler sum"]
        .mean()
        .reset_index()
    )

    unique_mvenus_mean = (
        df_all.groupby(["probe", "cell"])["mVenus sum"]
        .mean()
        .reset_index()
    )

    all_filler_mean = unique_filler_mean["filler sum"].mean()
    print(f"all_filler_mean: {all_filler_mean}")

    unique_filler_mean["across_cell_filler_coefficient"] = (
        unique_filler_mean["filler sum"] / all_filler_mean
    )

    df_all = df_all.merge(
        unique_filler_mean[
            ["probe", "cell", "across_cell_filler_coefficient"]
        ],
        on=["probe", "cell"],
        how="left",
    )

    df_all["across_spine_filler"] = (
        df_all["filler sum"] / df_all["across_cell_filler_coefficient"]
    )

    df_all["AS/filler_ratio_filler_normalized"] = (
        df_all["mVenus sum"] / df_all["across_spine_filler"]
    )

    cell_rows = []
    for probe, cell in unique_filler_mean[["probe", "cell"]].drop_duplicates().values:
        filler_intensity = unique_filler_mean[
            (unique_filler_mean["probe"] == probe)
            & (unique_filler_mean["cell"] == cell)
        ]["filler sum"].iloc[0]

        ias_intensity = unique_mvenus_mean[
            (unique_mvenus_mean["probe"] == probe)
            & (unique_mvenus_mean["cell"] == cell)
        ]["mVenus sum"].iloc[0]

        cell_rows.append({
            "probe": probe,
            "cell": cell,
            "filler_intensity": filler_intensity,
            "iAS_intensity": ias_intensity,
        })

    return df_all, pd.DataFrame(cell_rows)


def add_subtraction_mode_score(df_all, df_cell):
    # ============================================================
    # cell内baseline modeを引いたscoreを作る
    # ============================================================
    # 各cell内でAS/filler_ratio_filler_normalizedのmodeをKDEで推定する。
    # そのmodeをbaseline populationとみなし、baseline相当量を引く。
    # baseから上に外れたspineを評価しやすくする。
    mode_rows = []
    subtraction_rows = []

    for (probe, cell), group in df_all.groupby(["probe", "cell"]):
        data_as = group["AS/filler_ratio_filler_normalized"].values

        kde = gaussian_kde(data_as)
        x_grid = np.linspace(data_as.min(), data_as.max(), 100)
        kde_values = kde(x_grid)
        peaks, _ = find_peaks(kde_values)

        if len(peaks) > 0:
            mode_as_filler_ratio = x_grid[peaks].min()
        else:
            mode_as_filler_ratio = x_grid[np.argmax(kde_values)]

        subtraction_mode_as_intensity = (
            group["mVenus sum"]
            - group["across_spine_filler"] * mode_as_filler_ratio
        )

        subtraction_mode_as_filler_ratio = (
            subtraction_mode_as_intensity / group["across_spine_filler"]
        )

        for label, dendrite, intensity, ratio in zip(
            group["spine_label"],
            group["dendrite"],
            subtraction_mode_as_intensity,
            subtraction_mode_as_filler_ratio,
        ):
            subtraction_rows.append({
                "probe": probe,
                "cell": cell,
                "dendrite": dendrite,
                "spine_label": label,
                "subtraction_mode_AS_intensity": intensity,
                RAW_SCORE_COL: ratio,
            })

        mode_rows.append({
            "probe": probe,
            "cell": cell,
            "mode_AS/filler ratio": mode_as_filler_ratio,
        })

    subtraction_mode_df = pd.DataFrame(subtraction_rows)
    mode_cell_df = pd.DataFrame(mode_rows)

    df_all = df_all.merge(
        subtraction_mode_df,
        on=["spine_label", "probe", "cell", "dendrite"],
        how="left",
    )

    df_cell = df_cell.merge(mode_cell_df, on=["probe", "cell"], how="left")

    return df_all, df_cell


def add_probe_median_centering(df_all):
    # ============================================================
    # probeごとのbaseline位置をそろえる
    # ============================================================
    # probeごとの中央値をRAW_SCORE_COLから引く。
    # 絶対的なscaleは残し、probe間のbaseline offsetだけを補正する。
    # medianで割る正規化は、低median probeを人工的に高く見せうるため避ける。
    probe_median_df = (
        df_all.groupby("probe")[RAW_SCORE_COL]
        .median()
        .reset_index()
        .rename(columns={
            RAW_SCORE_COL: "probe_median_subtraction_mode_AS_filler_ratio"
        })
    )

    df_all = df_all.merge(probe_median_df, on="probe", how="left")
    df_all[CENTERED_SCORE_COL] = (
        df_all[RAW_SCORE_COL]
        - df_all["probe_median_subtraction_mode_AS_filler_ratio"]
    )

    return df_all


def find_first_low_kde_peak(data):
    # ============================================================
    # pooled dataから低値側の最初のKDE peakを探す
    # ============================================================
    # hot spot/outlier側の山ではなく、
    # base populationに対応する低値側の山をfit中心候補にする。
    data = pd.Series(data).dropna().values
    kde = stats.gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 1000)
    kde_vals = kde(x_vals)
    peaks, _ = find_peaks(kde_vals)

    if len(peaks) == 0:
        return x_vals[np.argmax(kde_vals)], x_vals, kde_vals

    median_value = np.median(data)
    peak_xs = x_vals[peaks]
    low_peak_xs = peak_xs[peak_xs <= median_value]

    if len(low_peak_xs) > 0:
        base_peak = low_peak_xs.min()
    else:
        base_peak = peak_xs.min()

    return base_peak, x_vals, kde_vals


def fit_global_half_gaussian(data, bin_width=6e-03):
    # ============================================================
    # メイン解析: low-side half-Gaussian fit
    # ============================================================
    # KDEで見つけた低値側base peak以下のデータだけを使う。
    # 左側データをbase peak中心に折り返し、Gaussian fitする。
    # 右側のhot spot/outlier populationに引っ張られにくいglobal base SDを推定する。
    data = pd.Series(data).dropna().values
    base_peak, kde_x, kde_y = find_first_low_kde_peak(data)

    left_data = data[data <= base_peak]
    folded_data = base_peak + np.abs(base_peak - left_data)
    symmetric_data = np.concatenate([left_data, folded_data])

    bin_edges = np.arange(
        symmetric_data.min(),
        symmetric_data.max() + bin_width,
        bin_width,
    )

    hist, bin_edges = np.histogram(symmetric_data, bins=bin_edges, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    popt, _ = curve_fit(
        gaussian,
        bin_centers,
        hist,
        p0=[hist.max(), base_peak, np.std(symmetric_data)],
        maxfev=10000,
    )

    a_fit, b_fit, c_fit = popt

    return {
        "method": "half_gaussian_global",
        "base_peak": float(base_peak),
        "base_mean": float(b_fit),
        "base_sd": float(abs(c_fit)),
        "a_fit": float(a_fit),
        "kde_x": kde_x,
        "kde_y": kde_y,
        "symmetric_data": symmetric_data,
        "popt": popt,
    }


def fit_global_gmm_base(data, max_components=4, min_component_weight=0.05):
    # ============================================================
    # 確認解析: Gaussian Mixture Modelによるbase component推定
    # ============================================================
    # BICで成分数を選び、最も低値側のcomponentをbase populationとして使う。
    # half-Gaussian解析のrobustness checkとして使う。
    data_array = pd.Series(data).dropna().values.reshape(-1, 1)

    best_gmm = None
    best_bic = np.inf

    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=0,
        )
        gmm.fit(data_array)
        bic = gmm.bic(data_array)

        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    means = best_gmm.means_.flatten()
    sds = np.sqrt(
        best_gmm.covariances_.reshape(best_gmm.n_components, -1).flatten()
    )
    weights = best_gmm.weights_.flatten()

    valid_components = np.where(weights >= min_component_weight)[0]
    if len(valid_components) == 0:
        valid_components = np.arange(best_gmm.n_components)

    base_component = valid_components[np.argmin(means[valid_components])]

    return {
        "method": "gmm_base_global",
        "base_mean": float(means[base_component]),
        "base_sd": float(sds[base_component]),
        "base_weight": float(weights[base_component]),
        "n_components": int(best_gmm.n_components),
        "best_bic": float(best_bic),
        "means": means.tolist(),
        "sds": sds.tolist(),
        "weights": weights.tolist(),
        "base_component": int(base_component),
        "gmm": best_gmm,
    }


def summarize_by_dendrite(df, z_col, base_sd):
    # ============================================================
    # spine単位のz-scoreをdendrite単位に集約する
    # ============================================================
    # hotspot_index、z > 2、z > 3の割合、
    # 外れ値spineの平均z-score、dendrite全体の平均/中央値z-scoreを計算する。
    rows = []

    for (probe, cell, dendrite), group in df.groupby(["probe", "cell", "dendrite"]):
        z = group[z_col].dropna()
        if len(z) == 0:
            continue

        diff = z.diff()
        diff.iloc[0] = z.iloc[-1] - z.iloc[0]
        hotspot_index = diff.abs().mean()

        above_2 = z[z > 2]
        above_3 = z[z > 3]

        rows.append({
            "probe": probe,
            "cell": cell,
            "dendrite": dendrite,
            "hotspot_index": hotspot_index,
            "global_base_SD": base_sd,
            "percentage_above_2SD": len(above_2) / len(z) * 100,
            "percentage_above_3SD": len(above_3) / len(z) * 100,
            "above_2SD_mean": above_2.mean() if len(above_2) > 0 else np.nan,
            "above_3SD_mean": above_3.mean() if len(above_3) > 0 else np.nan,
            "mean_z_score": z.mean(),
            "median_z_score": z.median(),
        })

    return pd.DataFrame(rows)


def save_index_csv(summary_df, metric, method, output_root, filename):
    # 既存解析と同じように、各指標をindex_val形式のCSVとして個別保存する。
    summary_df[[metric, "probe", "cell", "dendrite"]].rename(
        columns={metric: "index_val"}
    ).to_csv(
        os.path.join(output_root, filename.format(method=method)),
        index=False,
    )


def holm_adjust(p_values):
    # Holm correction for a list/array of p-values.
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = ~np.isnan(p_values)

    if valid.sum() == 0:
        return adjusted

    valid_indices = np.where(valid)[0]
    sorted_valid_indices = valid_indices[np.argsort(p_values[valid])]
    m = len(sorted_valid_indices)
    running_max = 0

    for rank, original_index in enumerate(sorted_valid_indices):
        corrected = (m - rank) * p_values[original_index]
        running_max = max(running_max, corrected)
        adjusted[original_index] = min(running_max, 1.0)

    return adjusted


def dunn_test_with_holm(df, metric, group_col="probe"):
    # Dunn's test implemented from ranks, followed by Holm correction.
    # Input rows are dendrite-level summary values.
    test_df = df[[group_col, metric]].dropna()
    groups = test_df[group_col].dropna().unique().tolist()

    if len(groups) < 2:
        return pd.DataFrame()

    values = test_df[metric].values
    ranks = stats.rankdata(values)
    test_df = test_df.copy()
    test_df["_rank"] = ranks

    n_total = len(test_df)
    tie_counts = pd.Series(values).value_counts().values
    tie_correction = np.sum(tie_counts ** 3 - tie_counts) / (12 * (n_total - 1)) if n_total > 1 else 0
    variance_base = n_total * (n_total + 1) / 12 - tie_correction

    group_stats = (
        test_df.groupby(group_col)["_rank"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "mean_rank", "count": "n"})
    )

    rows = []
    for i, group1 in enumerate(groups):
        for group2 in groups[i + 1:]:
            n1 = group_stats.loc[group1, "n"]
            n2 = group_stats.loc[group2, "n"]
            mean_rank1 = group_stats.loc[group1, "mean_rank"]
            mean_rank2 = group_stats.loc[group2, "mean_rank"]

            se = np.sqrt(variance_base * (1 / n1 + 1 / n2))
            z_value = (mean_rank1 - mean_rank2) / se if se > 0 else np.nan
            p_value = 2 * stats.norm.sf(abs(z_value)) if not np.isnan(z_value) else np.nan

            rows.append({
                "metric": metric,
                "group1": group1,
                "group2": group2,
                "n1": n1,
                "n2": n2,
                "mean_rank1": mean_rank1,
                "mean_rank2": mean_rank2,
                "z": z_value,
                "p_uncorrected": p_value,
            })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df["p_holm"] = holm_adjust(result_df["p_uncorrected"].values)

    return result_df


def save_nonparametric_tests(summary_df, method, output_root):
    # Run Kruskal-Wallis and Dunn's post hoc test with Holm correction.
    # Tests are performed on dendrite-level values grouped by probe.
    metrics = [
        "hotspot_index",
        "percentage_above_2SD",
        "percentage_above_3SD",
        "above_2SD_mean",
        "above_3SD_mean",
        "mean_z_score",
        "median_z_score",
    ]

    kruskal_rows = []
    dunn_rows = []

    for metric in metrics:
        if metric not in summary_df.columns:
            continue

        test_df = summary_df[["probe", metric]].dropna()
        grouped_values = [
            group[metric].values
            for _, group in test_df.groupby("probe")
            if len(group[metric].dropna()) > 0
        ]

        if len(grouped_values) >= 2:
            h_stat, p_value = stats.kruskal(*grouped_values)
        else:
            h_stat, p_value = np.nan, np.nan

        kruskal_rows.append({
            "method": method,
            "metric": metric,
            "n_groups": len(grouped_values),
            "H": h_stat,
            "p": p_value,
        })

        dunn_df = dunn_test_with_holm(summary_df, metric)
        if not dunn_df.empty:
            dunn_df.insert(0, "method", method)
            dunn_rows.append(dunn_df)

    kruskal_df = pd.DataFrame(kruskal_rows)
    kruskal_df.to_csv(
        os.path.join(output_root, f"_kruskal_wallis_{method}.csv"),
        index=False,
    )

    if len(dunn_rows) > 0:
        dunn_all_df = pd.concat(dunn_rows, ignore_index=True)
    else:
        dunn_all_df = pd.DataFrame()

    dunn_all_df.to_csv(
        os.path.join(output_root, f"_dunn_holm_{method}.csv"),
        index=False,
    )


def save_method_outputs(df_all, fit_result, output_root):
    # ============================================================
    # 1つのfit方法についてCSVを保存する
    # ============================================================
    # half-Gaussian用、GMM用で同じ形式の出力を作る。
    # 各methodのz-score、dendrite summary、個別指標CSV、fit parameter JSONを保存する。
    method = fit_result["method"]
    base_mean = fit_result["base_mean"]
    base_sd = fit_result["base_sd"]
    z_col = f"z_score_{method}"

    df_scored = df_all.copy()
    df_scored[z_col] = (
        df_scored[CENTERED_SCORE_COL] - base_mean
    ) / base_sd

    summary_df = summarize_by_dendrite(df_scored, z_col, base_sd)

    df_scored.to_csv(
        os.path.join(output_root, f"_all_data_{method}.csv"),
        index=False,
    )
    summary_df.to_csv(
        os.path.join(output_root, f"_dendrite_score_summary_{method}.csv"),
        index=False,
    )

    save_index_csv(summary_df, "hotspot_index", method, output_root, "_hotspot_index_{method}.csv")
    save_index_csv(summary_df, "global_base_SD", method, output_root, "_standard_deviation_{method}.csv")
    save_index_csv(summary_df, "percentage_above_2SD", method, output_root, "_percentage_above_2SD_{method}.csv")
    save_index_csv(summary_df, "percentage_above_3SD", method, output_root, "_percentage_above_3SD_{method}.csv")
    save_index_csv(summary_df, "above_2SD_mean", method, output_root, "_above_2SD_mean_{method}.csv")
    save_index_csv(summary_df, "above_3SD_mean", method, output_root, "_above_3SD_mean_{method}.csv")
    save_index_csv(summary_df, "mean_z_score", method, output_root, "_mean_z_score_{method}.csv")

    df_scored[[z_col, "probe", "cell", "dendrite"]].rename(
        columns={z_col: "z_score"}
    ).to_csv(
        os.path.join(output_root, f"_z_score_{method}.csv"),
        index=False,
    )

    serializable_fit = {
        key: value
        for key, value in fit_result.items()
        if key not in ["gmm", "kde_x", "kde_y", "symmetric_data", "popt"]
    }

    with open(os.path.join(output_root, f"_fit_params_{method}.json"), "w") as f:
        json.dump(serializable_fit, f, indent=2)

    save_nonparametric_tests(summary_df, method, output_root)

    return df_scored, summary_df, z_col


def plot_metric_bar(fig, gs, df, metric, position, title, ylabel):
    # PDF下部に、probeごとのsummary指標をbar + strip plotで表示する。
    ax = fig.add_subplot(gs[position])
    plot_df = df.rename(columns={metric: "index_val"})

    sns.barplot(x="probe", y="index_val", data=plot_df, ax=ax)
    sns.stripplot(
        x="probe",
        y="index_val",
        data=plot_df,
        ax=ax,
        color="black",
        alpha=0.6,
        size=2,
        jitter=True,
    )

    ax.set_title(title)
    ax.set_xlabel("Probe")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=90)
    return ax


def plot_summary_pdf(df_scored, summary_df, fit_result, z_col, output_root):
    # ============================================================
    # 1つのfit方法についてsummary PDFを作る
    # ============================================================
    # 各probeを1行として、元signal、median-centered score、
    # global base fit、z-score histogram、z > 2/3、hot spot indexを表示する。
    # half-Gaussian版とGMM版を別々のPDFに保存する。
    method = fit_result["method"]
    probes = df_scored["probe"].dropna().unique().tolist()
    n_rows = max(len(probes), 6)

    fig = plt.figure(figsize=(70, max(35, n_rows * 4.5)))
    gs = gridspec.GridSpec(n_rows + 2, 10)
    plt.rc("font", size=14)

    global_x_min = df_scored[CENTERED_SCORE_COL].min()
    global_x_max = df_scored[CENTERED_SCORE_COL].max()
    x_fit = np.linspace(global_x_min, global_x_max, 1000)
    y_fit = gaussian(x_fit, 1, fit_result["base_mean"], fit_result["base_sd"])
    y_fit = y_fit / y_fit.max()

    # z-score histogramはprobeごとにbin幅が変わると比較しにくいため、
    # このPDF内の全probeで共通のbin edgeを使う。
    z_bin_width = 0.25
    global_z_values = df_scored[z_col].dropna().values
    global_z_min = np.floor(global_z_values.min() / z_bin_width) * z_bin_width
    global_z_max = np.ceil(global_z_values.max() / z_bin_width) * z_bin_width
    z_bin_edges = np.arange(
        global_z_min,
        global_z_max + z_bin_width,
        z_bin_width,
    )

    for p, probe in enumerate(probes):
        df_probe = df_scored[df_scored["probe"] == probe]
        unique_cells = df_probe["cell"].unique()
        colors = cm.get_cmap("tab10", len(unique_cells))

        ax = fig.add_subplot(gs[p, 0])
        for i, cell in enumerate(unique_cells):
            cell_data = df_probe[df_probe["cell"] == cell]
            ax.scatter(
                cell_data["V_normalized"],
                cell_data["mVenus sum"],
                s=20,
                edgecolor="none",
                alpha=0.5,
                color=colors(i),
            )
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x / 1000:.1f}k"))
        ax.set_xlabel("V (a.u.)")
        ax.set_ylabel(f"AS sum {probe}")
        ax.set_title("Control AS sum")
        ax.set_xlim(0, 6)

        ax = fig.add_subplot(gs[p, 1])
        for i, cell in enumerate(unique_cells):
            cell_data = df_probe[df_probe["cell"] == cell]
            ax.scatter(
                cell_data["V_normalized"],
                cell_data["AS/filler_ratio_filler_normalized"],
                s=20,
                edgecolor="none",
                alpha=0.5,
                color=colors(i),
            )
        ax.set_xlabel("V (a.u.)")
        ax.set_ylabel("AS/filler filler-norm")
        ax.set_title("Filler-normalized ratio")
        #ax.set_ylim(-0.2, 1.8)

        ax = fig.add_subplot(gs[p, 2])
        ax.hist(df_probe[RAW_SCORE_COL].dropna(), bins=100, density=True, alpha=0.7)
        ax.set_title("Subtraction-mode ratio")
        ax.set_xlabel(RAW_SCORE_COL)
        ax.set_ylabel("density")
        ax.set_xlim(-0.2, 1.8)

        ax = fig.add_subplot(gs[p, 3])
        ax.hist(df_probe[CENTERED_SCORE_COL].dropna(), bins=100, density=True, alpha=0.7)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("Probe median-centered")
        ax.set_xlabel(CENTERED_SCORE_COL)
        ax.set_ylabel("density")

        ax = fig.add_subplot(gs[p, 4])
        centered_values = df_probe[CENTERED_SCORE_COL].dropna().values
        hist, bin_edges = np.histogram(centered_values, bins=100, density=True)
        hist_max = hist.max()
        hist_normalized = hist / hist_max if hist_max > 0 else hist
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]

        ax.bar(
            bin_centers,
            hist_normalized,
            width=bin_width,
            color="#4C78A8",
            edgecolor="#1F3B5C",
            linewidth=0.25,
            alpha=0.65,
        )
        ax.plot(
            x_fit,
            y_fit,
            color="#D62728",
            linewidth=1.5,
            linestyle="--",
            alpha=0.55,
            label="Gaussian fit",
            zorder=5,
        )
        ax.axvline(
            fit_result["base_mean"],
            color="#D62728",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            label="fit mean",
        )
        if method == "half_gaussian_global":
            ax.axvline(
                fit_result["base_peak"],
                color="#2CA02C",
                linewidth=1.0,
                linestyle=":",
                alpha=0.65,
                label="KDE base peak",
            )
        ax.set_title(f"Global base fit: {method}")
        ax.set_xlabel("median-centered score")
        ax.set_ylabel("normalized density")
        ax.set_ylim(0, 1.1)
        ax.legend(frameon=False, fontsize=9, loc="upper right")

        ax = fig.add_subplot(gs[p, 5])
        z_values = df_probe[z_col].dropna()
        ax.hist(z_values, bins=z_bin_edges, alpha=0.7)
        ax.axvline(2, color="red", linewidth=1)
        ax.axvline(3, color="purple", linewidth=1)
        ax.set_title("z-score")
        ax.set_xlabel("z-score")
        ax.set_ylabel("count")
        ax.set_xlim(-10, 80)

        if len(z_values) > 0 and z_values.max() > 80:
            z_max = z_values.max()
            inset_min = 80
            inset_padding = max(z_bin_width * 4, (z_max - inset_min) * 0.05)
            inset_max = z_max + inset_padding
            inset_bin_edges = np.arange(
                inset_min,
                inset_max + z_bin_width,
                z_bin_width,
            )

            inset_ax = inset_axes(
                ax,
                width="42%",
                height="42%",
                loc="upper right",
                borderpad=1.2,
            )
            inset_ax.hist(
                z_values,
                bins=inset_bin_edges,
                alpha=0.75,
                color="#4C78A8",
                edgecolor="#F28E2B",
                linewidth=0.45,
            )
            inset_ax.set_xlim(inset_min, inset_max)
            inset_ax.set_title("max z zoom", fontsize=8)
            inset_ax.tick_params(axis="both", labelsize=7)

        ax = fig.add_subplot(gs[p, 6])
        ax.hist(z_values, bins=z_bin_edges, alpha=0.7)
        ax.set_yscale("log")
        ax.axvline(2, color="red", linewidth=1)
        ax.axvline(3, color="purple", linewidth=1)
        ax.set_title("z-score log")
        ax.set_xlabel("z-score")
        ax.set_ylabel("count")
        ax.set_xlim(-10, 50)

        probe_summary = summary_df[summary_df["probe"] == probe]

        ax = fig.add_subplot(gs[p, 7])
        sns.stripplot(x="probe", y="percentage_above_2SD", data=probe_summary, color="black", size=3, jitter=True, ax=ax)
        ax.set_title("% > 2SD")
        ax.set_xlabel("")
        ax.set_ylabel("% spine")

        ax = fig.add_subplot(gs[p, 8])
        sns.stripplot(x="probe", y="percentage_above_3SD", data=probe_summary, color="black", size=3, jitter=True, ax=ax)
        ax.set_title("% > 3SD")
        ax.set_xlabel("")
        ax.set_ylabel("% spine")

        ax = fig.add_subplot(gs[p, 9])
        sns.stripplot(x="probe", y="hotspot_index", data=probe_summary, color="black", size=3, jitter=True, ax=ax)
        ax.set_title("Hot spot index")
        ax.set_xlabel("")
        ax.set_ylabel("score")

    row = n_rows
    plot_metric_bar(fig, gs, summary_df, "hotspot_index", (row, slice(0, 2)), "Hot spot index", "score")
    plot_metric_bar(fig, gs, summary_df, "percentage_above_2SD", (row, slice(2, 4)), "Percentage above 2SD", "% spine")
    plot_metric_bar(fig, gs, summary_df, "percentage_above_3SD", (row, slice(4, 6)), "Percentage above 3SD", "% spine")
    plot_metric_bar(fig, gs, summary_df, "above_2SD_mean", (row, slice(6, 8)), "Mean z above 2SD", "z-score")
    plot_metric_bar(fig, gs, summary_df, "mean_z_score", (row, slice(8, 10)), "Mean z-score", "z-score")

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_root, f"_summary_NoAPV_{method}.pdf"),
        dpi=100,
        transparent=True,
    )
    plt.close(fig)


def main():
    # ============================================================
    # 全体の実行順序
    # ============================================================
    # 1. control CSVを読み込む
    # 2. cell間filler補正を行う
    # 3. cell内mode subtraction scoreを作る
    # 4. probeごとのmedianを引いてbaseline位置をそろえる
    # 5. 全probeのmedian-centered dataをpoolする
    # 6. pooled dataにhalf-Gaussian fitを行う
    # 7. pooled dataにGMM base fitを行う
    # 8. それぞれのfit方法でz-score化する
    # 9. dendrite単位の指標CSVを保存する
    # 10. half-Gaussian版とGMM版のsummary PDFを別々に保存する
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df_all = build_all_data(INPUT_ROOT)
    df_all, df_cell = add_across_cell_filler_normalization(df_all)
    df_all, df_cell = add_subtraction_mode_score(df_all, df_cell)
    df_all = add_probe_median_centering(df_all)

    df_all.to_csv(os.path.join(OUTPUT_ROOT, "_all_data_before_global_scoring.csv"), index=False)
    df_cell.to_csv(os.path.join(OUTPUT_ROOT, "_cell_data.csv"), index=False)

    pooled_data = df_all[CENTERED_SCORE_COL].dropna().values

    half_fit = fit_global_half_gaussian(pooled_data)
    gmm_fit = fit_global_gmm_base(pooled_data)

    print("Half-Gaussian global fit")
    print({
        "base_peak": half_fit["base_peak"],
        "base_mean": half_fit["base_mean"],
        "base_sd": half_fit["base_sd"],
    })

    print("GMM global base fit")
    print({
        "base_mean": gmm_fit["base_mean"],
        "base_sd": gmm_fit["base_sd"],
        "base_weight": gmm_fit["base_weight"],
        "n_components": gmm_fit["n_components"],
        "means": gmm_fit["means"],
        "sds": gmm_fit["sds"],
        "weights": gmm_fit["weights"],
    })

    for fit_result in [half_fit, gmm_fit]:
        df_scored, summary_df, z_col = save_method_outputs(
            df_all=df_all,
            fit_result=fit_result,
            output_root=OUTPUT_ROOT,
        )
        plot_summary_pdf(
            df_scored=df_scored,
            summary_df=summary_df,
            fit_result=fit_result,
            z_col=z_col,
            output_root=OUTPUT_ROOT,
        )


if __name__ == "__main__":
    main()
