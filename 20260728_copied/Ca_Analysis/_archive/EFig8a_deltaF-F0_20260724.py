import os
import glob
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from scipy.stats import ks_2samp, wilcoxon
from EEG_Ca_treadmill_analysis import extract_params


# ============================================================
# Plot settings
# ============================================================

plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42


# ============================================================
# ﾎ認/F0 calculation
# ============================================================

def compute_dff(
    F,
    fs=31,
    window_sec=60,
    max_abs_dff=1e6
):
    """
    蜈ｨ譎る俣縺ｮ陋榊�蠑ｷ蠎ｦ縺九ｉﾎ認/F0繧定ｨ育ｮ励☆繧九�

    F0:
        centered rolling 8th percentile

    Parameters
    ----------
    F : np.ndarray
        shape = (cell, frame)
    fs : float
        sampling frequency
    window_sec : float
        rolling window length in seconds

    A cell is excluded from all downstream analyses when any of the
    following is true:

    1. F or F0 contains a non-finite value
    2. F0 is zero or negative at any frame
    3. |ﾎ認/F0| exceeds max_abs_dff at any frame

    Returns
    -------
    dff : np.ndarray
        ﾎ認/F0 for all input cells. Excluded cells are filled with NaN.
    valid_cell_mask : np.ndarray
        Boolean mask indicating cells retained for analysis.
    qc_df : pd.DataFrame
        Cell-level quality-control results.
    """
    window = int(window_sec * fs)

    F0 = (
        pd.DataFrame(F.T)
        .rolling(
            window=window,
            center=True,
            min_periods=1
        )
        .quantile(0.08)
        .to_numpy()
        .T
    )

    nonfinite_F = np.any(~np.isfinite(F), axis=1)
    nonfinite_F0 = np.any(~np.isfinite(F0), axis=1)
    nonpositive_F0 = np.any(F0 <= 0, axis=1)

    # Do not replace invalid F0 with an arbitrarily small positive number.
    # Calculate ﾎ認/F0 only where division is mathematically valid.
    initially_valid = (
        ~nonfinite_F &
        ~nonfinite_F0 &
        ~nonpositive_F0
    )

    dff = np.full(F.shape, np.nan, dtype=float)

    if np.any(initially_valid):
        dff[initially_valid] = (
            F[initially_valid] - F0[initially_valid]
        ) / F0[initially_valid]

    extreme_dff = np.zeros(F.shape[0], dtype=bool)

    if np.any(initially_valid):
        extreme_dff[initially_valid] = np.any(
            np.abs(dff[initially_valid]) > max_abs_dff,
            axis=1
        )

    valid_cell_mask = initially_valid & ~extreme_dff

    # Excluded cells must never enter peak detection accidentally.
    dff[~valid_cell_mask] = np.nan

    min_F = np.full(F.shape[0], np.nan)
    max_F = np.full(F.shape[0], np.nan)
    min_F0 = np.full(F.shape[0], np.nan)
    max_F0 = np.full(F.shape[0], np.nan)
    max_abs_dff_per_cell = np.full(F.shape[0], np.nan)

    for cell in range(F.shape[0]):
        finite_F = F[cell][np.isfinite(F[cell])]
        finite_F0 = F0[cell][np.isfinite(F0[cell])]

        if len(finite_F) > 0:
            min_F[cell] = np.min(finite_F)
            max_F[cell] = np.max(finite_F)

        if len(finite_F0) > 0:
            min_F0[cell] = np.min(finite_F0)
            max_F0[cell] = np.max(finite_F0)

        if initially_valid[cell]:
            max_abs_dff_per_cell[cell] = np.nanmax(
                np.abs(dff[cell])
            )

    exclusion_reason = []

    for cell in range(F.shape[0]):
        reasons = []

        if nonfinite_F[cell]:
            reasons.append("nonfinite_F")

        if nonfinite_F0[cell]:
            reasons.append("nonfinite_F0")

        if nonpositive_F0[cell]:
            reasons.append("F0_le_0")

        if extreme_dff[cell]:
            reasons.append(
                f"abs_dff_gt_{max_abs_dff:g}"
            )

        exclusion_reason.append(
            ";".join(reasons) if reasons else "included"
        )

    qc_df = pd.DataFrame({
        "cell": np.arange(F.shape[0], dtype=int),
        "included": valid_cell_mask,
        "exclusion_reason": exclusion_reason,
        "min_F_corrected": min_F,
        "max_F_corrected": max_F,
        "min_F0": min_F0,
        "max_F0": max_F0,
        "max_abs_dff": max_abs_dff_per_cell
    })

    return dff, valid_cell_mask, qc_df


# ============================================================
# Extract peaks from one trace
# ============================================================

def extract_peaks_from_trace(
    trace,
    dff_threshold=1.0,
    minimum_length=15
):
    """
    1譛ｬ縺ｮﾎ認/F0 trace縺九ｉ縲》hreshold莉･荳翫′荳螳壽悄髢鍋ｶ壹￥event繧呈､懷�縺励�
    蜷�event縺ｮ譛螟ｧﾎ認/F0繧定ｿ斐☆縲�
    """
    trace = np.asarray(trace, dtype=float)

    if len(trace) == 0:
        return []

    valid_mask = np.isfinite(trace)
    threshold_mask = valid_mask & (trace >= dff_threshold)

    transitions = np.diff(threshold_mask.astype(np.int8))

    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if threshold_mask[0]:
        starts = np.r_[0, starts]

    if threshold_mask[-1]:
        ends = np.r_[ends, len(threshold_mask)]

    peak_values = []

    for start, end in zip(starts, ends):

        if (end - start) < minimum_length:
            continue

        segment = trace[start:end]

        if not np.any(np.isfinite(segment)):
            continue

        peak_values.append(np.nanmax(segment))

    return peak_values


# ============================================================
# Process one mouse only once
# ============================================================

def process_mouse_all_immobile_events(
    data_folder,
    dff_threshold=1.0,
    minimum_length=15,
    fs=31
):
    """
    1蛹ｹ縺ｮ繝槭え繧ｹ縺ｫ縺､縺�※莉･荳九ｒ1蝗槭□縺大ｮ溯｡後☆繧九�

    1. F_corrected.npy繧定ｪｭ縺ｿ霎ｼ繧
    2. 蜈ｨ譎る俣縺ｮﾎ認/F0繧定ｨ育ｮ励☆繧�
    3. 蜈ｨimmobile event縺ｫ縺､縺�※Ca event peak繧呈歓蜃ｺ縺吶ｋ

    Returns
    -------
    peak_df : pd.DataFrame
        蜈ｨimmobile event荳ｭ縺ｫ讀懷�縺輔ｌ縺殫eak縺ｮ荳隕ｧ

    immobile_event_df : pd.DataFrame
        蜈ｨimmobile event縺ｮstart/end/duration荳隕ｧ

    valid_cell_ids : np.ndarray
        QC蠕後↓隗｣譫舌∈菴ｿ逕ｨ縺励◆cell ID

    qc_df : pd.DataFrame
        蜈ｨcell縺ｮQC邨先棡
    """
    mouse_name = os.path.basename(data_folder)

    print()
    print("=" * 70)
    print(f"Processing mouse: {mouse_name}")
    print("=" * 70)

    event_path = os.path.join(
        data_folder,
        "_Combined",
        "manual_event.csv"
    )

    frame_time_path = os.path.join(
        data_folder,
        "_Combined",
        "2p_frame_time_combined.csv"
    )

    fluorescence_path = os.path.join(
        data_folder,
        "_GCaMP",
        "suite2p_bleach_corrected",
        "F_corrected.npy"
    )

    iscell_path = os.path.join(
        data_folder,
        "_GCaMP",
        "suite2p",
        "plane0",
        "iscell.npy"
    )

    required_paths = {
        "manual event": event_path,
        "2p frame time": frame_time_path,
        "fluorescence": fluorescence_path,
        "iscell": iscell_path
    }

    for file_description, file_path in required_paths.items():
        if not os.path.exists(file_path):
            print(
                f"Skip {mouse_name}: "
                f"{file_description} file not found"
            )
            print(file_path)

            return (
                pd.DataFrame(),
                pd.DataFrame(),
                np.array([], dtype=int),
                pd.DataFrame()
            )

    # 蜈�さ繝ｼ繝峨→縺ｮ莠呈鋤諤ｧ縺ｮ縺溘ａ螳溯｡�
    *_, contime = extract_params(data_folder)

    event_df = pd.read_csv(event_path)
    frame2p_df = pd.read_csv(frame_time_path)

    F_full = np.load(fluorescence_path)
    iscell = np.load(iscell_path)

    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F_full = F_full[cell_indices]

    cell_num_before_qc = F_full.shape[0]

    if cell_num_before_qc == 0:
        print(f"Skip {mouse_name}: no cells")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            np.array([], dtype=int),
            pd.DataFrame()
        )

    print(f"Number of iscell cells: {cell_num_before_qc}")
    print("Calculating ﾎ認/F0 over the full recording...")

    # 縺薙�驥阪＞險育ｮ励ｒ縲∝推繝槭え繧ｹ縺ｫ縺､縺�※1蝗槭□縺題｡後≧
    dff_all, valid_cell_mask, qc_df = compute_dff(
        F_full,
        fs=fs,
        window_sec=60,
        max_abs_dff=1e6
    )

    qc_df["mouse"] = mouse_name
    qc_df["suite2p_roi_index"] = cell_indices

    valid_cell_ids = np.flatnonzero(valid_cell_mask)
    dff_full = dff_all[valid_cell_mask]

    print(
        f"Included cells: {len(valid_cell_ids)} / "
        f"{cell_num_before_qc}"
    )
    print(
        f"Excluded cells: "
        f"{cell_num_before_qc - len(valid_cell_ids)}"
    )

    if len(valid_cell_ids) == 0:
        print(f"Skip {mouse_name}: no cells passed ﾎ認/F0 QC")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            valid_cell_ids,
            qc_df
        )

    print("ﾎ認/F0 calculation completed.")

    # --------------------------------------------------------
    # Frame information
    # --------------------------------------------------------

    frame2p_df = frame2p_df.copy()

    frame2p_df["time"] = pd.to_numeric(
        frame2p_df["time"],
        errors="coerce"
    )

    frame2p_df["frame"] = pd.to_numeric(
        frame2p_df["frame"],
        errors="coerce"
    )

    frame2p_df = frame2p_df.dropna(
        subset=["time", "frame"]
    )

    frame2p_df["frame"] = frame2p_df["frame"].astype(int)

    frame2p_df = frame2p_df[
        (frame2p_df["frame"] >= 0) &
        (frame2p_df["frame"] < dff_full.shape[1])
    ]

    # --------------------------------------------------------
    # Extract all immobile events
    # --------------------------------------------------------

    event_df = event_df.copy()

    event_df["start_time"] = pd.to_numeric(
        event_df["start_time"],
        errors="coerce"
    )

    event_df["end_time"] = pd.to_numeric(
        event_df["end_time"],
        errors="coerce"
    )

    immobile_event_df = event_df[
        event_df["event_name"].str.contains(
            "immobile",
            case=False,
            na=False
        )
    ].copy()

    immobile_event_df = immobile_event_df.dropna(
        subset=["start_time", "end_time"]
    )

    immobile_event_df = immobile_event_df[
        immobile_event_df["end_time"] >
        immobile_event_df["start_time"]
    ].copy()

    immobile_event_df = immobile_event_df.reset_index(drop=True)

    immobile_event_df["immobile_event_id"] = np.arange(
        len(immobile_event_df)
    )

    immobile_event_df["duration_sec"] = (
        immobile_event_df["end_time"] -
        immobile_event_df["start_time"]
    )

    immobile_event_df["mouse"] = mouse_name

    print(
        f"Number of immobile events: "
        f"{len(immobile_event_df)}"
    )

    # --------------------------------------------------------
    # Extract peaks from every immobile event only once
    # --------------------------------------------------------

    peak_records = []

    for event_number, event_row in immobile_event_df.iterrows():

        event_id = event_row["immobile_event_id"]
        event_start = event_row["start_time"]
        event_end = event_row["end_time"]

        frame_mask = (
            (frame2p_df["time"] >= event_start) &
            (frame2p_df["time"] <= event_end)
        )

        frames = frame2p_df.loc[
            frame_mask,
            "frame"
        ].to_numpy(dtype=int)

        # 蠢ｵ縺ｮ縺溘ａ驥崎､㌶rame繧帝勁縺�
        frames = np.unique(frames)

        if len(frames) == 0:
            continue

        event_dff = dff_full[:, frames]

        for local_cell_index, cell_id in enumerate(valid_cell_ids):

            peak_values = extract_peaks_from_trace(
                event_dff[local_cell_index],
                dff_threshold=dff_threshold,
                minimum_length=minimum_length
            )

            for peak_index, peak_value in enumerate(peak_values):

                peak_records.append({
                    "mouse": mouse_name,
                    "immobile_event_id": event_id,
                    "event_start_time": event_start,
                    "event_end_time": event_end,
                    "event_duration_sec": (
                        event_end - event_start
                    ),
                    # cell is the index among iscell==1 cells before QC.
                    # It is intentionally not renumbered after exclusion.
                    "cell": int(cell_id),
                    "suite2p_roi_index": int(
                        cell_indices[cell_id]
                    ),
                    "peak_index_within_cell_event": peak_index,
                    "peak_dff": peak_value
                })

        if (event_number + 1) % 20 == 0:
            print(
                f"Processed immobile events: "
                f"{event_number + 1}/"
                f"{len(immobile_event_df)}"
            )

    peak_df = pd.DataFrame(peak_records)

    print(
        f"Peak extraction completed: "
        f"{len(peak_df)} peaks"
    )

    return peak_df, immobile_event_df, valid_cell_ids, qc_df


# ============================================================
# Filter events and peaks for one time window
# ============================================================

def select_time_window_data(
    peak_df,
    immobile_event_df,
    tw
):
    """
    謖�ｮ嗾ime window蜀�↓螳悟�縺ｫ蜷ｫ縺ｾ繧後ｋimmobile event縺ｨ縲�
    縺昴ｌ繧峨�event縺ｫ蜷ｫ縺ｾ繧後ｋpeak繧呈歓蜃ｺ縺吶ｋ縲�
    """
    window_start_sec = tw[0] * 60
    window_end_sec = tw[1] * 60

    selected_events = immobile_event_df[
        (immobile_event_df["start_time"] >= window_start_sec) &
        (immobile_event_df["end_time"] <= window_end_sec)
    ].copy()

    if selected_events.empty or peak_df.empty:
        selected_peaks = pd.DataFrame(
            columns=peak_df.columns
        )
    else:
        selected_event_ids = set(
            selected_events["immobile_event_id"].tolist()
        )

        selected_peaks = peak_df[
            peak_df["immobile_event_id"].isin(
                selected_event_ids
            )
        ].copy()

    return selected_peaks, selected_events



# ============================================================
# Mouse-level paired statistics
# ============================================================

def paired_wilcoxon_test(values_1, values_2):
    """
    Paired Wilcoxon signed-rank test across mice.

    NaN pairs are removed. Pairs with exactly zero difference are handled
    using scipy's default zero_method="wilcox" (zero differences excluded).

    Returns
    -------
    statistic : float
    p_value : float
    n_pairs : int
    n_nonzero_differences : int
    note : str
    """
    values_1 = np.asarray(values_1, dtype=float)
    values_2 = np.asarray(values_2, dtype=float)

    valid = np.isfinite(values_1) & np.isfinite(values_2)
    x = values_1[valid]
    y = values_2[valid]
    n_pairs = len(x)

    if n_pairs == 0:
        return np.nan, np.nan, 0, 0, "no_complete_mouse_pairs"

    differences = x - y
    n_nonzero = int(np.sum(differences != 0))

    if n_nonzero == 0:
        return 0.0, 1.0, n_pairs, 0, "all_paired_differences_zero"

    try:
        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="auto"
        )
    except TypeError:
        # Compatibility with older SciPy versions without method=.
        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False
        )

    return (
        float(result.statistic),
        float(result.pvalue),
        n_pairs,
        n_nonzero,
        "ok"
    )


def exact_paired_sign_permutation_test(
    values_1,
    values_2,
    max_exact_nonzero_pairs=20
):
    """
    Exact two-sided paired permutation test across mice.

    The test statistic is the absolute mean paired difference. Under the null,
    the sign of each non-zero mouse-level difference is independently flipped.
    Every one of the 2^n sign assignments is enumerated, so this is exact.

    Parameters
    ----------
    values_1, values_2 : array-like
        Mouse-level values from two time windows, ordered by the same mouse IDs.
    max_exact_nonzero_pairs : int
        Safety limit for exhaustive enumeration. With the expected mouse
        numbers (roughly 5-10), the test remains comfortably exact.

    Returns
    -------
    observed_mean_difference : float
        Mean(values_1 - values_2).
    statistic : float
        Absolute observed mean difference.
    p_value : float
        Exact two-sided permutation p-value.
    n_pairs : int
        Number of complete mouse pairs.
    n_nonzero_differences : int
        Number of non-zero paired differences used for sign enumeration.
    n_permutations : int
        Number of sign assignments enumerated.
    note : str
    """
    values_1 = np.asarray(values_1, dtype=float)
    values_2 = np.asarray(values_2, dtype=float)

    valid = np.isfinite(values_1) & np.isfinite(values_2)
    x = values_1[valid]
    y = values_2[valid]
    n_pairs = len(x)

    if n_pairs == 0:
        return np.nan, np.nan, np.nan, 0, 0, 0, "no_complete_mouse_pairs"

    differences = x - y
    observed_mean_difference = float(np.mean(differences))
    observed_statistic = abs(observed_mean_difference)

    nonzero_differences = differences[differences != 0]
    n_nonzero = len(nonzero_differences)

    if n_nonzero == 0:
        return (
            observed_mean_difference,
            0.0,
            1.0,
            n_pairs,
            0,
            1,
            "all_paired_differences_zero"
        )

    if n_nonzero > max_exact_nonzero_pairs:
        return (
            observed_mean_difference,
            observed_statistic,
            np.nan,
            n_pairs,
            n_nonzero,
            0,
            f"too_many_nonzero_pairs_for_exact_enumeration_gt_{max_exact_nonzero_pairs}"
        )

    n_permutations = 2 ** n_nonzero
    extreme_count = 0
    tolerance = 1e-15

    # Zero differences do not affect the statistic and therefore do not need
    # independent sign assignments.
    for permutation_index in range(n_permutations):
        signs = np.ones(n_nonzero, dtype=float)

        for bit_index in range(n_nonzero):
            if (permutation_index >> bit_index) & 1:
                signs[bit_index] = -1.0

        permuted_mean = np.sum(signs * nonzero_differences) / n_pairs
        permuted_statistic = abs(permuted_mean)

        if permuted_statistic >= observed_statistic - tolerance:
            extreme_count += 1

    p_value = extreme_count / n_permutations

    return (
        observed_mean_difference,
        observed_statistic,
        float(p_value),
        n_pairs,
        n_nonzero,
        n_permutations,
        "ok"
    )


# ============================================================
# Mouse-level paired plot
# ============================================================

def create_paired_mouse_bar_figure(
    values_1,
    values_2,
    mouse_ids,
    window_1,
    window_2,
    mouse_color_map,
    wilcoxon_p_value=np.nan,
    wilcoxon_p_bonferroni=np.nan
):
    """Create one page for the multi-page paired Wilcoxon PDF."""
    values_1 = np.asarray(values_1, dtype=float)
    values_2 = np.asarray(values_2, dtype=float)
    mouse_ids = np.asarray(mouse_ids, dtype=object)

    valid = np.isfinite(values_1) & np.isfinite(values_2)
    x_values = values_1[valid]
    y_values = values_2[valid]
    valid_mouse_ids = mouse_ids[valid]

    if len(x_values) == 0:
        return None

    paired_values = np.column_stack([x_values, y_values])
    means = np.mean(paired_values, axis=0)
    sems = (
        np.std(paired_values, axis=0, ddof=1) / np.sqrt(len(x_values))
        if len(x_values) > 1
        else np.array([0.0, 0.0], dtype=float)
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    positions = np.array([0.0, 1.0])

    ax.bar(
        positions,
        means,
        yerr=sems,
        width=0.62,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        error_kw={"elinewidth": 1.5, "capsize": 4, "capthick": 1.5},
        zorder=1
    )

    for mouse_id, mouse_values in zip(valid_mouse_ids, paired_values):
        ax.plot(
            positions,
            mouse_values,
            color=mouse_color_map[str(mouse_id)],
            linewidth=1.5,
            alpha=0.9,
            marker=None,
            zorder=2
        )

    label_1 = window_1.replace("to", " to ").replace("min", " min")
    label_2 = window_2.replace("to", " to ").replace("min", " min")
    ax.set_xticks(positions)
    ax.set_xticklabels([label_1, label_2], rotation=25, ha="right")
    ax.set_ylabel(r"Mean peak $\Delta F/F_0$ per mouse")

    raw_p_text = f"{wilcoxon_p_value:.3g}" if np.isfinite(wilcoxon_p_value) else "NA"
    corrected_p_text = (
        f"{wilcoxon_p_bonferroni:.3g}"
        if np.isfinite(wilcoxon_p_bonferroni)
        else "NA"
    )
    ax.set_title(
        f"Paired Wilcoxon, n = {len(x_values)}\n"
        f"p = {raw_p_text}, Bonferroni p = {corrected_p_text}"
    )

    legend_handles = [
        Line2D([0], [0], color=mouse_color_map[str(mouse_id)], linewidth=2.0, label=str(mouse_id))
        for mouse_id in valid_mouse_ids
    ]
    ax.legend(
        handles=legend_handles,
        title="Mouse",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    return fig


# ============================================================
# Group analysis
# ============================================================

def process_group(path):

    analysis_time_window = [
        [-45, 0],
        [-60, 0],
        [0, 45],
        [45, 120],
        [0, 120],
        [0, 90],
        [20, 60]
    ]

    mouse_list = sorted(
        glob.glob(os.path.join(path, "202*"))
    )

    outdir = os.path.join(
        path,
        "_group_analysis"
    )

    os.makedirs(outdir, exist_ok=True)

    # --------------------------------------------------------
    # Time-window縺斐→縺ｮ菫晏ｭ倬�伜沺
    # --------------------------------------------------------

    peaks_by_window = {}
    rates_by_window = {}
    total_cell_minutes_by_window = {}

    # One mean peak amplitude per mouse and time window. These values are the
    # independent experimental units used for Wilcoxon and exact permutation.
    mouse_mean_peak_by_window = {}
    mouse_mean_records = []

    cell_qc_records = []

    for tw in analysis_time_window:

        window_label = f"{tw[0]}to{tw[1]}min"

        peaks_by_window[window_label] = []
        rates_by_window[window_label] = []
        total_cell_minutes_by_window[window_label] = 0.0
        mouse_mean_peak_by_window[window_label] = {}

    # ========================================================
    # Mouse loop
    #
    # 蜷��繧ｦ繧ｹ縺ｫ縺､縺�※縲�㍾縺��逅��縺薙％縺ｧ1蝗槭□縺題｡後≧
    # ========================================================

    for mouse_path in mouse_list:

        mouse_name = os.path.basename(mouse_path)

        (
            all_peak_df,
            immobile_event_df,
            valid_cell_ids,
            cell_qc_df
        ) = (
            process_mouse_all_immobile_events(
                mouse_path,
                dff_threshold=1.0,
                minimum_length=15,
                fs=31
            )
        )

        if not cell_qc_df.empty:
            cell_qc_records.append(cell_qc_df)

        valid_cell_num = len(valid_cell_ids)

        if valid_cell_num == 0:
            continue

        # ----------------------------------------------------
        # 荳蠎ｦ謚ｽ蜃ｺ縺励◆event/peak繧稚ime window縺斐→縺ｫ謖ｯ繧雁�縺代ｋ
        # ----------------------------------------------------

        for tw in analysis_time_window:

            window_label = f"{tw[0]}to{tw[1]}min"

            selected_peaks, selected_events = (
                select_time_window_data(
                    all_peak_df,
                    immobile_event_df,
                    tw
                )
            )

            immobile_time_sec = (
                selected_events["duration_sec"].sum()
                if not selected_events.empty
                else 0.0
            )

            immobile_time_min = immobile_time_sec / 60.0

            # -----------------------------------------------
            # Mouse-level mean peak amplitude
            # -----------------------------------------------

            if not selected_peaks.empty:
                mouse_peak_values = pd.to_numeric(
                    selected_peaks["peak_dff"],
                    errors="coerce"
                )
                mouse_peak_values = (
                    mouse_peak_values
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                mouse_peak_values = mouse_peak_values[
                    mouse_peak_values > 0
                ]
            else:
                mouse_peak_values = pd.Series(dtype=float)

            if len(mouse_peak_values) > 0:
                mouse_mean_peak_dff = float(mouse_peak_values.mean())
                mouse_median_peak_dff = float(mouse_peak_values.median())
                mouse_n_peaks = int(len(mouse_peak_values))
            else:
                mouse_mean_peak_dff = np.nan
                mouse_median_peak_dff = np.nan
                mouse_n_peaks = 0

            mouse_mean_peak_by_window[window_label][mouse_name] = (
                mouse_mean_peak_dff
            )

            mouse_mean_records.append({
                "record_type": "mouse_summary",
                "mouse": mouse_name,
                "time_window_1": window_label,
                "time_window_2": np.nan,
                "window_1_start_min": tw[0],
                "window_1_end_min": tw[1],
                "window_2_start_min": np.nan,
                "window_2_end_min": np.nan,
                "mouse_mean_peak_dff": mouse_mean_peak_dff,
                "mouse_median_peak_dff": mouse_median_peak_dff,
                "mouse_n_peaks": mouse_n_peaks,
                "mouse_immobile_time_min": immobile_time_min
            })

            # -----------------------------------------------
            # Peak list
            # -----------------------------------------------

            if not selected_peaks.empty:

                selected_peaks = selected_peaks.copy()

                selected_peaks["time_window"] = window_label
                selected_peaks["window_start_min"] = tw[0]
                selected_peaks["window_end_min"] = tw[1]

                peaks_by_window[window_label].append(
                    selected_peaks
                )

            # -----------------------------------------------
            # Event rate for every cell
            # -----------------------------------------------

            if immobile_time_min > 0:

                total_cell_minutes_by_window[window_label] += (
                    valid_cell_num * immobile_time_min
                )

                if not selected_peaks.empty:
                    counts = selected_peaks.groupby(
                        "cell"
                    ).size()
                else:
                    counts = pd.Series(dtype=float)

                all_cells = pd.Index(
                    valid_cell_ids,
                    name="cell"
                )

                counts = counts.reindex(
                    all_cells,
                    fill_value=0
                )

                rate_df = (
                    counts
                    .rename("event_count")
                    .to_frame()
                    .reset_index()
                    .rename(columns={"index": "cell"})
                )

                rate_df["events_per_min"] = (
                    rate_df["event_count"] /
                    immobile_time_min
                )

                rate_df["mouse"] = mouse_name
                rate_df["time_window"] = window_label
                rate_df["window_start_min"] = tw[0]
                rate_df["window_end_min"] = tw[1]
                rate_df["immobile_time_min"] = (
                    immobile_time_min
                )

                rates_by_window[window_label].append(
                    rate_df
                )

    # ========================================================
    # Save cell-level quality control
    # ========================================================

    if len(cell_qc_records) > 0:
        all_cell_qc_df = pd.concat(
            cell_qc_records,
            axis=0,
            ignore_index=True
        )
    else:
        all_cell_qc_df = pd.DataFrame(columns=[
            "cell",
            "included",
            "exclusion_reason",
            "min_F_corrected",
            "max_F_corrected",
            "min_F0",
            "max_F0",
            "max_abs_dff",
            "mouse",
            "suite2p_roi_index"
        ])

    all_cell_qc_df.to_csv(
        os.path.join(
            outdir,
            "dff_cell_quality_control_all_cells.csv"
        ),
        index=False
    )

    excluded_cell_qc_df = all_cell_qc_df[
        ~all_cell_qc_df["included"].astype(bool)
    ].copy()

    excluded_cell_qc_df.to_csv(
        os.path.join(
            outdir,
            "dff_excluded_cells.csv"
        ),
        index=False
    )

    print()
    print(
        f"Total excluded cells: "
        f"{len(excluded_cell_qc_df)}"
    )

    # ========================================================
    # Save per-window results and make histograms
    # ========================================================

    peak_values_by_window = {}
    summary_records = []

    for tw in analysis_time_window:

        window_label = f"{tw[0]}to{tw[1]}min"

        # ----------------------------------------------------
        # Combine peak data
        # ----------------------------------------------------

        if len(peaks_by_window[window_label]) > 0:
            all_peaks_df = pd.concat(
                peaks_by_window[window_label],
                axis=0,
                ignore_index=True
            )
        else:
            all_peaks_df = pd.DataFrame(columns=[
                "mouse",
                "immobile_event_id",
                "event_start_time",
                "event_end_time",
                "event_duration_sec",
                "cell",
                "suite2p_roi_index",
                "peak_index_within_cell_event",
                "peak_dff",
                "time_window",
                "window_start_min",
                "window_end_min"
            ])

        # ----------------------------------------------------
        # Combine event-rate data
        # ----------------------------------------------------

        if len(rates_by_window[window_label]) > 0:
            all_event_rate_df = pd.concat(
                rates_by_window[window_label],
                axis=0,
                ignore_index=True
            )
        else:
            all_event_rate_df = pd.DataFrame(columns=[
                "cell",
                "event_count",
                "events_per_min",
                "mouse",
                "time_window",
                "window_start_min",
                "window_end_min",
                "immobile_time_min"
            ])

        # ----------------------------------------------------
        # Save peak list
        # ----------------------------------------------------

        peak_list_path = os.path.join(
            outdir,
            f"_Ca_event_list_{tw[0]}-{tw[1]}min.csv"
        )

        all_peaks_df.to_csv(
            peak_list_path,
            index=False
        )

        # ----------------------------------------------------
        # Save event rate
        # ----------------------------------------------------

        event_rate_path = os.path.join(
            outdir,
            f"_Ca_event_rate_{tw[0]}-{tw[1]}min.csv"
        )

        all_event_rate_df.to_csv(
            event_rate_path,
            index=False
        )

        # ----------------------------------------------------
        # Prepare peak distribution
        # ----------------------------------------------------

        if (
            not all_peaks_df.empty and
            "peak_dff" in all_peaks_df.columns
        ):
            values = pd.to_numeric(
                all_peaks_df["peak_dff"],
                errors="coerce"
            )

            values = (
                values
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            values = values[
                values > 0
            ].to_numpy(dtype=float)

        else:
            values = np.array([], dtype=float)

        peak_values_by_window[window_label] = values

        # ----------------------------------------------------
        # Mean and SD
        # ----------------------------------------------------

        n_peaks = len(values)

        mean_peak_dff = (
            np.mean(values)
            if n_peaks > 0
            else np.nan
        )

        # 讓呎悽讓呎ｺ門￥蟾ｮ
        sd_peak_dff = (
            np.std(values, ddof=1)
            if n_peaks > 1
            else np.nan
        )

        summary_records.append({
            "record_type": "summary",
            "time_window_1": window_label,
            "time_window_2": np.nan,
            "window_1_start_min": tw[0],
            "window_1_end_min": tw[1],
            "window_2_start_min": np.nan,
            "window_2_end_min": np.nan,
            "n_window_1": n_peaks,
            "n_window_2": np.nan,
            "mean_peak_dff": mean_peak_dff,
            "sd_peak_dff": sd_peak_dff,
            "ks_statistic": np.nan,
            "ks_p_value": np.nan,
            "ks_p_bonferroni": np.nan,
            "significant_raw_p_lt_0.05": np.nan,
            "significant_bonferroni_p_lt_0.05": np.nan
        })

        # ----------------------------------------------------
        # Histogram
        # ----------------------------------------------------

        fig, ax = plt.subplots(figsize=(4, 4))

        bins = np.arange(0, 31, 1)

        total_cell_minutes = (
            total_cell_minutes_by_window[window_label]
        )

        if n_peaks > 0 and total_cell_minutes > 0:

            weights = (
                np.ones(n_peaks, dtype=float) /
                total_cell_minutes
            )

            sns.histplot(
                x=values,
                bins=bins,
                kde=False,
                stat="count",
                color="#466eb4",
                edgecolor="black",
                weights=weights,
                ax=ax
            )

        ax.set_xlabel(
            r"ﾎ認 / F$_0$",
            fontsize=12
        )

        ax.set_ylabel(
            "Events / cell / min",
            fontsize=12
        )

        if np.isfinite(mean_peak_dff):
            mean_text = f"{mean_peak_dff:.3f}"
        else:
            mean_text = "NA"

        if np.isfinite(sd_peak_dff):
            sd_text = f"{sd_peak_dff:.3f}"
        else:
            sd_text = "NA"

        ax.set_title(
            f"{tw[0]} to {tw[1]} min\n"
            f"mean = {mean_text}, "
            f"SD = {sd_text}, "
            f"n = {n_peaks}"
        )

        fig.tight_layout()

        histogram_path = os.path.join(
            outdir,
            f"peak_dff_distribution_{window_label}.pdf"
        )

        fig.savefig(
            histogram_path,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
            dpi=300
        )

        plt.close(fig)

    # ========================================================
    # Save mouse-level summary values
    # ========================================================

    mouse_mean_df = pd.DataFrame(mouse_mean_records)

    mouse_mean_path = os.path.join(
        outdir,
        "peak_dff_mouse_level_summary.csv"
    )

    mouse_mean_df.to_csv(
        mouse_mean_path,
        index=False
    )

    # ========================================================
    # Pairwise two-sample KS tests
    # ========================================================

    window_labels = [
        f"{tw[0]}to{tw[1]}min"
        for tw in analysis_time_window
    ]

    window_information = {
        f"{tw[0]}to{tw[1]}min": tw
        for tw in analysis_time_window
    }

    window_pairs = list(
        combinations(window_labels, 2)
    )

    number_of_comparisons = len(window_pairs)

    ks_records = []

    for window_1, window_2 in window_pairs:

        values_1 = peak_values_by_window[window_1]
        values_2 = peak_values_by_window[window_2]

        tw_1 = window_information[window_1]
        tw_2 = window_information[window_2]

        if len(values_1) > 0 and len(values_2) > 0:

            ks_result = ks_2samp(
                values_1,
                values_2,
                alternative="two-sided",
                method="auto"
            )

            ks_statistic = float(
                ks_result.statistic
            )

            ks_p_value = float(
                ks_result.pvalue
            )

            ks_p_bonferroni = min(
                ks_p_value * number_of_comparisons,
                1.0
            )

        else:
            ks_statistic = np.nan
            ks_p_value = np.nan
            ks_p_bonferroni = np.nan

        ks_records.append({
            "record_type": "KS_test",
            "time_window_1": window_1,
            "time_window_2": window_2,
            "window_1_start_min": tw_1[0],
            "window_1_end_min": tw_1[1],
            "window_2_start_min": tw_2[0],
            "window_2_end_min": tw_2[1],
            "n_window_1": len(values_1),
            "n_window_2": len(values_2),
            "mean_peak_dff": np.nan,
            "sd_peak_dff": np.nan,
            "ks_statistic": ks_statistic,
            "ks_p_value": ks_p_value,
            "ks_p_bonferroni": ks_p_bonferroni,
            "significant_raw_p_lt_0.05": (
                ks_p_value < 0.05
                if np.isfinite(ks_p_value)
                else np.nan
            ),
            "significant_bonferroni_p_lt_0.05": (
                ks_p_bonferroni < 0.05
                if np.isfinite(ks_p_bonferroni)
                else np.nan
            )
        })

    # ========================================================
    # Pairwise mouse-level Wilcoxon and exact permutation tests
    # ========================================================

    wilcoxon_records = []
    permutation_records = []
    paired_plot_records = []

    all_mouse_ids = sorted({
        mouse
        for window_dict in mouse_mean_peak_by_window.values()
        for mouse in window_dict.keys()
    })
    color_values = plt.cm.tab20(np.linspace(0, 1, max(len(all_mouse_ids), 1)))
    mouse_color_map = {
        mouse: color_values[index % len(color_values)]
        for index, mouse in enumerate(all_mouse_ids)
    }

    paired_plot_path = os.path.join(
        outdir,
        "mouse_mean_peak_dff_all_28_paired_wilcoxon_comparisons.pdf"
    )

    for window_1, window_2 in window_pairs:

        tw_1 = window_information[window_1]
        tw_2 = window_information[window_2]

        mouse_dict_1 = mouse_mean_peak_by_window[window_1]
        mouse_dict_2 = mouse_mean_peak_by_window[window_2]

        common_mice = sorted(
            set(mouse_dict_1.keys()) & set(mouse_dict_2.keys())
        )

        paired_mice = []
        values_1 = []
        values_2 = []

        for mouse in common_mice:
            value_1 = mouse_dict_1[mouse]
            value_2 = mouse_dict_2[mouse]

            if np.isfinite(value_1) and np.isfinite(value_2):
                paired_mice.append(mouse)
                values_1.append(value_1)
                values_2.append(value_2)

        values_1 = np.asarray(values_1, dtype=float)
        values_2 = np.asarray(values_2, dtype=float)

        if len(values_1) > 0:
            mean_mouse_value_1 = float(np.mean(values_1))
            mean_mouse_value_2 = float(np.mean(values_2))
            median_paired_difference = float(
                np.median(values_1 - values_2)
            )
        else:
            mean_mouse_value_1 = np.nan
            mean_mouse_value_2 = np.nan
            median_paired_difference = np.nan

        (
            wilcoxon_statistic,
            wilcoxon_p_value,
            wilcoxon_n_pairs,
            wilcoxon_n_nonzero,
            wilcoxon_note
        ) = paired_wilcoxon_test(values_1, values_2)

        wilcoxon_p_bonferroni = (
            min(wilcoxon_p_value * number_of_comparisons, 1.0)
            if np.isfinite(wilcoxon_p_value)
            else np.nan
        )

        paired_plot_records.append({
            "values_1": values_1.copy(),
            "values_2": values_2.copy(),
            "mouse_ids": paired_mice.copy(),
            "window_1": window_1,
            "window_2": window_2,
            "wilcoxon_p_value": wilcoxon_p_value,
            "wilcoxon_p_bonferroni": wilcoxon_p_bonferroni
        })

        wilcoxon_records.append({
            "record_type": "mouse_paired_wilcoxon",
            "time_window_1": window_1,
            "time_window_2": window_2,
            "window_1_start_min": tw_1[0],
            "window_1_end_min": tw_1[1],
            "window_2_start_min": tw_2[0],
            "window_2_end_min": tw_2[1],
            "n_mouse_pairs": wilcoxon_n_pairs,
            "n_nonzero_paired_differences": wilcoxon_n_nonzero,
            "paired_mouse_ids": ";".join(paired_mice),
            "paired_plot_pdf": paired_plot_path,
            "mean_mouse_peak_dff_window_1": mean_mouse_value_1,
            "mean_mouse_peak_dff_window_2": mean_mouse_value_2,
            "mean_paired_difference_window_1_minus_2": (
                float(np.mean(values_1 - values_2))
                if len(values_1) > 0
                else np.nan
            ),
            "median_paired_difference_window_1_minus_2": (
                median_paired_difference
            ),
            "wilcoxon_statistic": wilcoxon_statistic,
            "wilcoxon_p_value": wilcoxon_p_value,
            "wilcoxon_p_bonferroni": wilcoxon_p_bonferroni,
            "significant_raw_p_lt_0.05": (
                wilcoxon_p_value < 0.05
                if np.isfinite(wilcoxon_p_value)
                else np.nan
            ),
            "significant_bonferroni_p_lt_0.05": (
                wilcoxon_p_bonferroni < 0.05
                if np.isfinite(wilcoxon_p_bonferroni)
                else np.nan
            ),
            "test_note": wilcoxon_note
        })

        (
            observed_mean_difference,
            permutation_statistic,
            permutation_p_value,
            permutation_n_pairs,
            permutation_n_nonzero,
            n_exact_permutations,
            permutation_note
        ) = exact_paired_sign_permutation_test(
            values_1,
            values_2,
            max_exact_nonzero_pairs=20
        )

        permutation_p_bonferroni = (
            min(permutation_p_value * number_of_comparisons, 1.0)
            if np.isfinite(permutation_p_value)
            else np.nan
        )

        permutation_records.append({
            "record_type": "mouse_exact_paired_permutation",
            "time_window_1": window_1,
            "time_window_2": window_2,
            "window_1_start_min": tw_1[0],
            "window_1_end_min": tw_1[1],
            "window_2_start_min": tw_2[0],
            "window_2_end_min": tw_2[1],
            "n_mouse_pairs": permutation_n_pairs,
            "n_nonzero_paired_differences": permutation_n_nonzero,
            "paired_mouse_ids": ";".join(paired_mice),
            "mean_mouse_peak_dff_window_1": mean_mouse_value_1,
            "mean_mouse_peak_dff_window_2": mean_mouse_value_2,
            "mean_paired_difference_window_1_minus_2": (
                observed_mean_difference
            ),
            "median_paired_difference_window_1_minus_2": (
                median_paired_difference
            ),
            "permutation_statistic_abs_mean_difference": (
                permutation_statistic
            ),
            "permutation_p_value": permutation_p_value,
            "permutation_p_bonferroni": permutation_p_bonferroni,
            "n_exact_permutations": n_exact_permutations,
            "significant_raw_p_lt_0.05": (
                permutation_p_value < 0.05
                if np.isfinite(permutation_p_value)
                else np.nan
            ),
            "significant_bonferroni_p_lt_0.05": (
                permutation_p_bonferroni < 0.05
                if np.isfinite(permutation_p_bonferroni)
                else np.nan
            ),
            "test_note": permutation_note
        })

    # ========================================================
    # Save all 28 paired Wilcoxon plots into one multi-page PDF
    # ========================================================

    with PdfPages(paired_plot_path) as pdf:
        for plot_record in paired_plot_records:
            fig = create_paired_mouse_bar_figure(
                values_1=plot_record["values_1"],
                values_2=plot_record["values_2"],
                mouse_ids=plot_record["mouse_ids"],
                window_1=plot_record["window_1"],
                window_2=plot_record["window_2"],
                mouse_color_map=mouse_color_map,
                wilcoxon_p_value=plot_record["wilcoxon_p_value"],
                wilcoxon_p_bonferroni=plot_record["wilcoxon_p_bonferroni"]
            )
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight", facecolor="white", transparent=False, dpi=300)
                plt.close(fig)

    # ========================================================
    # Save summary and KS results in one CSV
    # ========================================================

    result_df = pd.DataFrame(
        summary_records
        + mouse_mean_records
        + ks_records
        + wilcoxon_records
        + permutation_records
    )

    result_path = os.path.join(
        outdir,
        "peak_dff_summary_and_all_statistical_tests.csv"
    )

    result_df.to_csv(
        result_path,
        index=False
    )

    print()
    print("=" * 70)
    print("Analysis completed")
    print("=" * 70)
    print(f"Output folder: {outdir}")
    print(f"Summary and all statistical results: {result_path}")
    print(f"All paired Wilcoxon plots: {paired_plot_path}")
    print(f"Mouse-level summary: {mouse_mean_path}")


# ============================================================
# Main
# ============================================================

def main():

    path = r"X:\Behavior\Ca_imaging\SynC"

    process_group(path)


if __name__ == "__main__":
    main()