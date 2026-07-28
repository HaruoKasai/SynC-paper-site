import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# ============================================================
# User settings
# ============================================================

INPUT_CSV = (
    r"X:\Behavior\Ca_imaging\_Dendrites\_summary"
    r"\summary_spine_only_events.csv"
)

OUTPUT_DIR = r"X:\Behavior\Ca_imaging\_Dendrites\_summary"

# The supplied CSV contains 5-min before and 5-min after analysis periods.
# Change these values if the analyzed durations are changed upstream.
PHASE_DURATION_MIN = {
    "before": 5.0,
    "after": 5.0,
}

PHASE_ORDER = ["before", "after"]

BIN_WIDTH_DFF = 0.1

N_MONTE_CARLO = 100000
RANDOM_SEED = 0
MAX_EXACT_NONZERO_PAIRS = 20


# ============================================================
# Plot settings
# ============================================================

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["pdf.fonttype"] = 42

PHASE_COLORS = {
    "before": "#808080",
    "after": "#d55e00",
}


# ============================================================
# Paired sign-permutation test
# ============================================================

def paired_sign_permutation_test(
    before_values,
    after_values,
    max_exact_nonzero_pairs=20,
    n_monte_carlo=100000,
    random_seed=0,
):
    """
    Two-sided paired sign-permutation test.

    The analysis unit is one spine. The test statistic is the absolute mean
    paired difference:

        abs(mean(after - before))

    Exact enumeration is used when the number of non-zero differences is at
    most max_exact_nonzero_pairs. Otherwise, Monte Carlo sign permutation is
    used with a fixed random seed.
    """
    before_values = np.asarray(before_values, dtype=float)
    after_values = np.asarray(after_values, dtype=float)

    valid = (
        np.isfinite(before_values) &
        np.isfinite(after_values)
    )

    before_values = before_values[valid]
    after_values = after_values[valid]

    n_pairs = len(before_values)

    if n_pairs == 0:
        return {
            "observed_mean_difference_after_minus_before": np.nan,
            "statistic_abs_mean_difference": np.nan,
            "permutation_p_value_two_sided": np.nan,
            "n_spine_pairs": 0,
            "n_nonzero_differences": 0,
            "permutation_method": "none",
            "n_permutations": 0,
            "random_seed": random_seed,
            "test_note": "no_complete_spine_pairs",
        }

    differences = after_values - before_values

    observed_mean_difference = float(np.mean(differences))
    observed_statistic = abs(observed_mean_difference)

    nonzero_differences = differences[differences != 0]
    n_nonzero = len(nonzero_differences)

    if n_nonzero == 0:
        return {
            "observed_mean_difference_after_minus_before": (
                observed_mean_difference
            ),
            "statistic_abs_mean_difference": 0.0,
            "permutation_p_value_two_sided": 1.0,
            "n_spine_pairs": n_pairs,
            "n_nonzero_differences": 0,
            "permutation_method": "exact",
            "n_permutations": 1,
            "random_seed": random_seed,
            "test_note": "all_paired_differences_zero",
        }

    tolerance = 1e-15

    if n_nonzero <= max_exact_nonzero_pairs:
        n_permutations = 2 ** n_nonzero
        extreme_count = 0

        for permutation_index in range(n_permutations):
            signs = np.ones(n_nonzero, dtype=float)

            for bit_index in range(n_nonzero):
                if (permutation_index >> bit_index) & 1:
                    signs[bit_index] = -1.0

            permuted_mean = (
                np.sum(signs * nonzero_differences) /
                n_pairs
            )

            if (
                abs(permuted_mean) >=
                observed_statistic - tolerance
            ):
                extreme_count += 1

        p_value = extreme_count / n_permutations
        permutation_method = "exact"

    else:
        rng = np.random.default_rng(random_seed)
        extreme_count = 0
        completed = 0
        chunk_size = 5000

        while completed < n_monte_carlo:
            current_chunk_size = min(
                chunk_size,
                n_monte_carlo - completed,
            )

            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(current_chunk_size, n_nonzero),
            )

            permuted_means = (
                signs @ nonzero_differences
            ) / n_pairs

            extreme_count += int(np.sum(
                np.abs(permuted_means) >=
                observed_statistic - tolerance
            ))

            completed += current_chunk_size

        # Plus-one correction prevents a zero Monte Carlo p-value.
        p_value = (
            (extreme_count + 1) /
            (n_monte_carlo + 1)
        )

        n_permutations = n_monte_carlo
        permutation_method = "monte_carlo"

    return {
        "observed_mean_difference_after_minus_before": (
            observed_mean_difference
        ),
        "statistic_abs_mean_difference": observed_statistic,
        "permutation_p_value_two_sided": float(p_value),
        "n_spine_pairs": n_pairs,
        "n_nonzero_differences": n_nonzero,
        "permutation_method": permutation_method,
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "test_note": "ok",
    }


# ============================================================
# Input validation and preparation
# ============================================================

def load_and_prepare_events(input_csv):
    required_columns = {
        "pair",
        "phase",
        "peak_dff_spine",
        "start_time_s",
        "end_time_s",
        "mouse",
    }

    events = pd.read_csv(input_csv)

    missing_columns = required_columns - set(events.columns)

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            + ", ".join(sorted(missing_columns))
        )

    events = events.copy()

    events["phase"] = (
        events["phase"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    events = events[
        events["phase"].isin(PHASE_ORDER)
    ].copy()

    events["peak_dff_spine"] = pd.to_numeric(
        events["peak_dff_spine"],
        errors="coerce",
    )

    events = events.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            "mouse",
            "pair",
            "phase",
            "peak_dff_spine",
        ]
    )

    events = events[
        events["peak_dff_spine"] > 0
    ].copy()

    # "pair" labels are repeated across mice, so mouse must be included in
    # the unique spine identifier.
    events["spine_id"] = (
        events["mouse"].astype(str)
        + "::"
        + events["pair"].astype(str)
    )

    events = events.reset_index(drop=True)

    if events.empty:
        raise ValueError(
            "No valid before/after spine events remained after filtering."
        )

    return events


# ============================================================
# Plot peak-dF/F distributions as events / spine / min
# ============================================================

def save_distribution_plot(
    events,
    all_spine_ids,
    output_path,
):
    n_spines = len(all_spine_ids)

    maximum_peak = float(
        events["peak_dff_spine"].max()
    )

    upper_edge = (
        np.ceil(maximum_peak / BIN_WIDTH_DFF) *
        BIN_WIDTH_DFF
    )

    bins = np.arange(
        0.0,
        upper_edge + BIN_WIDTH_DFF,
        BIN_WIDTH_DFF,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8, 4),
        sharex=True,
        sharey=True,
    )

    for axis, phase in zip(axes, PHASE_ORDER):
        phase_values = (
            events.loc[
                events["phase"] == phase,
                "peak_dff_spine",
            ]
            .to_numpy(dtype=float)
        )

        total_spine_minutes = (
            n_spines *
            PHASE_DURATION_MIN[phase]
        )

        weights = (
            np.ones(len(phase_values), dtype=float) /
            total_spine_minutes
        )

        axis.hist(
            phase_values,
            bins=bins,
            weights=weights,
            color=PHASE_COLORS[phase],
            edgecolor="black",
            linewidth=0.6,
        )

        mean_value = np.mean(phase_values)
        sd_value = (
            np.std(phase_values, ddof=1)
            if len(phase_values) > 1
            else np.nan
        )

        axis.set_title(
            f"{phase.capitalize()}\n"
            f"mean={mean_value:.3f}, "
            f"SD={sd_value:.3f}, "
            f"n={len(phase_values)}"
        )

        axis.set_xlabel(r"Peak $\Delta F/F_0$")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Events / spine / min")

    fig.tight_layout()

    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# Main analysis
# ============================================================

def analyze_spine_peak_dff(
    input_csv=INPUT_CSV,
    output_dir=OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)

    events = load_and_prepare_events(input_csv)

    all_spine_ids = sorted(
        events["spine_id"].unique()
    )

    n_total_spines = len(all_spine_ids)

    # --------------------------------------------------------
    # Event-level summary
    # --------------------------------------------------------

    summary_records = []

    for phase in PHASE_ORDER:
        phase_values = (
            events.loc[
                events["phase"] == phase,
                "peak_dff_spine",
            ]
            .to_numpy(dtype=float)
        )

        summary_records.append({
            "record_type": "event_level_summary",
            "phase": phase,
            "n_events": len(phase_values),
            "n_total_spines": n_total_spines,
            "phase_duration_min": PHASE_DURATION_MIN[phase],
            "total_spine_minutes": (
                n_total_spines *
                PHASE_DURATION_MIN[phase]
            ),
            "event_rate_per_spine_per_min": (
                len(phase_values) /
                (
                    n_total_spines *
                    PHASE_DURATION_MIN[phase]
                )
            ),
            "mean_peak_dff": float(np.mean(phase_values)),
            "sd_peak_dff": (
                float(np.std(phase_values, ddof=1))
                if len(phase_values) > 1
                else np.nan
            ),
        })

    # --------------------------------------------------------
    # One mean peak-dF/F value per spine and phase
    # --------------------------------------------------------

    spine_phase_summary = (
        events
        .groupby(
            ["spine_id", "mouse", "pair", "phase"],
            as_index=False,
        )
        .agg(
            mean_peak_dff=("peak_dff_spine", "mean"),
            median_peak_dff=("peak_dff_spine", "median"),
            sd_peak_dff=("peak_dff_spine", "std"),
            n_events=("peak_dff_spine", "size"),
        )
    )

    for phase in PHASE_ORDER:
        phase_spine_means = (
            spine_phase_summary.loc[
                spine_phase_summary["phase"] == phase,
                "mean_peak_dff",
            ]
            .to_numpy(dtype=float)
        )

        summary_records.append({
            "record_type": "spine_mean_level_summary",
            "phase": phase,
            "n_spines_with_events": len(phase_spine_means),
            "mean_of_spine_mean_peak_dff": (
                float(np.mean(phase_spine_means))
            ),
            "sd_of_spine_mean_peak_dff": (
                float(np.std(phase_spine_means, ddof=1))
                if len(phase_spine_means) > 1
                else np.nan
            ),
        })

    # --------------------------------------------------------
    # Pair the same spines across before and after
    # --------------------------------------------------------

    paired_spines = (
        spine_phase_summary
        .pivot(
            index=["spine_id", "mouse", "pair"],
            columns="phase",
            values="mean_peak_dff",
        )
        .reset_index()
    )

    for phase in PHASE_ORDER:
        if phase not in paired_spines.columns:
            paired_spines[phase] = np.nan

    paired_spines["paired_complete"] = (
        paired_spines["before"].notna() &
        paired_spines["after"].notna()
    )

    paired_complete = paired_spines[
        paired_spines["paired_complete"]
    ].copy()

    permutation_result = paired_sign_permutation_test(
        paired_complete["before"].to_numpy(dtype=float),
        paired_complete["after"].to_numpy(dtype=float),
        max_exact_nonzero_pairs=MAX_EXACT_NONZERO_PAIRS,
        n_monte_carlo=N_MONTE_CARLO,
        random_seed=RANDOM_SEED,
    )

    paired_complete["difference_after_minus_before"] = (
        paired_complete["after"] -
        paired_complete["before"]
    )

    permutation_record = {
        "record_type": "spine_paired_permutation",
        "phase_1": "before",
        "phase_2": "after",
        "analysis_unit": "mouse_x_spine_pair",
        "mean_paired_before": (
            float(paired_complete["before"].mean())
            if len(paired_complete) > 0
            else np.nan
        ),
        "mean_paired_after": (
            float(paired_complete["after"].mean())
            if len(paired_complete) > 0
            else np.nan
        ),
        "sd_paired_before": (
            float(paired_complete["before"].std(ddof=1))
            if len(paired_complete) > 1
            else np.nan
        ),
        "sd_paired_after": (
            float(paired_complete["after"].std(ddof=1))
            if len(paired_complete) > 1
            else np.nan
        ),
        **permutation_result,
    }

    summary_records.append(permutation_record)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    plot_path = os.path.join(
        output_dir,
        "peak_dff_distribution_before_after_"
        "events_per_spine_min.pdf",
    )

    summary_path = os.path.join(
        output_dir,
        "peak_dff_before_after_summary_and_"
        "spine_permutation.csv",
    )

    spine_summary_path = os.path.join(
        output_dir,
        "peak_dff_spine_phase_summary.csv",
    )

    paired_spine_path = os.path.join(
        output_dir,
        "peak_dff_paired_spines_before_after.csv",
    )

    save_distribution_plot(
        events,
        all_spine_ids,
        plot_path,
    )

    pd.DataFrame(summary_records).to_csv(
        summary_path,
        index=False,
    )

    spine_phase_summary.to_csv(
        spine_summary_path,
        index=False,
    )

    paired_spines.to_csv(
        paired_spine_path,
        index=False,
    )

    print("Analysis completed.")
    print(f"Input: {input_csv}")
    print(f"Plot: {plot_path}")
    print(f"Summary and permutation: {summary_path}")
    print(f"Spine phase summary: {spine_summary_path}")
    print(f"Paired spines: {paired_spine_path}")

    return {
        "plot_path": plot_path,
        "summary_path": summary_path,
        "spine_summary_path": spine_summary_path,
        "paired_spine_path": paired_spine_path,
    }


def main():
    analyze_spine_peak_dff()


if __name__ == "__main__":
    main()
