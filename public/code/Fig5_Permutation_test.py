#!/usr/bin/env python3
"""
Fig. 5 permutation test

Reproduces the N=10 analysis used for Fig. 5:
- Core 8 mice plus 250722_z253-2* and 260205_z286-1*
- Metrics: firing rate, Spearman pairwise correlation, PR norm, PC50 norm
- Conditions: immobile/mobile; Before (-45 to 0 min), After (0 to 45 min),
  and 1 h after (45 to 120 min)
- Test: two-sided paired exact sign-flip permutation test
- Statistic: mean within-mouse difference
- Comparisons: After vs Before; 1 h after vs Before

Usage:
    python "Fig.5 Permutation test.py"

Optional:
    python "Fig.5 Permutation test.py" --input INPUT.csv --output-dir OUTPUT_DIR

Dependencies:
    numpy, pandas, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_INPUT = (
    "StatisticalInputValues_Be-45_0_Af0_45_Re45_120_"
    "win150_nw20_mouse_spearmanPearson_seed1_.csv"
)

ADDED_MICE = ["250722_z253-2*", "260205_z286-1*"]

METRICS = [
    ("firing_rate", "Firing rate"),
    ("mouse_all_cell_spearman_pairwise_correlation", "Spearman pairwise"),
    ("participation_ratio_norm_all_cells", "PR norm"),
    ("pca_pc50_norm_all_cells", "PC50 norm"),
]

CONDITION_ORDER = [
    "I_Before", "I_After", "I_1h",
    "M_Before", "M_After", "M_1h",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the N=10 Fig. 5 exact permutation analysis."
    )
    parser.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("fig5_permutation_output")
    )
    return parser.parse_args()


def condition_name(row: pd.Series) -> str | None:
    key = (
        str(row["event_name"]),
        int(row["tw_start_min"]),
        int(row["tw_end_min"]),
    )
    mapping = {
        ("immobile", -45, 0): "I_Before",
        ("immobile", 0, 45): "I_After",
        ("immobile", 45, 120): "I_1h",
        ("mobile", -45, 0): "M_Before",
        ("mobile", 0, 45): "M_After",
        ("mobile", 45, 120): "M_1h",
    }
    return mapping.get(key)


def validate_input(df: pd.DataFrame) -> None:
    required = {
        "metric", "mouse_id", "event_name",
        "tw_start_min", "tw_end_min", "value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df["value"].isna().any():
        raise ValueError("Missing values found in 'value'.")


def exact_sign_flip_test(differences: Iterable[float]) -> dict[str, float | int]:
    diff = np.asarray(list(differences), dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        raise ValueError("No finite paired differences supplied.")

    observed = float(diff.mean())
    observed_abs = abs(observed)
    n = diff.size
    total = 2 ** n
    extreme = 0

    for signs in itertools.product((-1.0, 1.0), repeat=n):
        statistic = abs(float(np.mean(np.asarray(signs) * diff)))
        if statistic >= observed_abs - 1e-15:
            extreme += 1

    return {
        "n": n,
        "mean_paired_difference": observed,
        "sem_paired_difference": (
            float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        ),
        "total_sign_assignments": total,
        "extreme_assignments": extreme,
        "exact_p": extreme / total,
    }


def significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def select_n10_mice(df: pd.DataFrame) -> list[str]:
    all_mice = sorted(df["mouse_id"].astype(str).unique())
    core8 = sorted(mouse for mouse in all_mice if "*" not in mouse)
    if len(core8) != 8:
        raise ValueError(f"Expected 8 non-starred mice; found {len(core8)}")
    missing = [m for m in ADDED_MICE if m not in all_mice]
    if missing:
        raise ValueError(f"Added mice absent from input: {missing}")
    return core8 + ADDED_MICE


def build_pivot(df: pd.DataFrame, metric_key: str, mice: list[str]) -> pd.DataFrame:
    subset = df[(df["metric"] == metric_key) & (df["mouse_id"].isin(mice))]
    pivot = subset.pivot(index="mouse_id", columns="condition", values="value")
    pivot = pivot.reindex(index=mice, columns=CONDITION_ORDER)
    if pivot.isna().any().any():
        raise ValueError(f"Missing values for metric: {metric_key}")
    return pivot


def create_test_tables(
    df: pd.DataFrame, mice: list[str], output_dir: Path
) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    individual_rows: list[dict[str, object]] = []

    specs = [
        ("Immobile", "Before vs After", "I_Before", "I_After", "After - Before"),
        ("Immobile", "Before vs 1 h after", "I_Before", "I_1h", "1 h after - Before"),
        ("Mobile", "Before vs After", "M_Before", "M_After", "After - Before"),
        ("Mobile", "Before vs 1 h after", "M_Before", "M_1h", "1 h after - Before"),
    ]

    for metric_key, metric_label in METRICS:
        pivot = build_pivot(df, metric_key, mice)
        state_values = {
            "Immobile": (pivot["I_Before"], pivot["I_After"], pivot["I_1h"]),
            "Mobile": (pivot["M_Before"], pivot["M_After"], pivot["M_1h"]),
        }

        for state, comparison, time1, time2, difference_definition in specs:
            before, after, one_hour = state_values[state]
            differences = pivot[time2] - pivot[time1]
            test = exact_sign_flip_test(differences)

            summary_rows.append({
                "metric": metric_label,
                "state": state,
                "comparison": comparison,
                "n": len(mice),
                "before_mean": before.mean(),
                "before_sem": before.sem(),
                "after_mean": after.mean(),
                "after_sem": after.sem(),
                "1h_after_mean": one_hour.mean(),
                "1h_after_sem": one_hour.sem(),
                "mean_paired_difference": test["mean_paired_difference"],
                "sem_paired_difference": test["sem_paired_difference"],
                "difference_definition": difference_definition,
                "total_sign_assignments": test["total_sign_assignments"],
                "extreme_assignments": test["extreme_assignments"],
                "exact_p": test["exact_p"],
                "significance": significance_label(float(test["exact_p"])),
            })

            for mouse in mice:
                individual_rows.append({
                    "metric": metric_label,
                    "state": state,
                    "comparison": comparison,
                    "mouse_id": mouse,
                    "before": before.loc[mouse],
                    "after": after.loc[mouse],
                    "1h_after": one_hour.loc[mouse],
                    "paired_difference_used": differences.loc[mouse],
                })

    summary = pd.DataFrame(summary_rows)
    individual = pd.DataFrame(individual_rows)

    summary.to_csv(output_dir / "Fig5_permutation_test_summary.csv", index=False)
    with pd.ExcelWriter(
        output_dir / "Fig5_permutation_test_complete_table.xlsx",
        engine="openpyxl",
    ) as writer:
        summary.to_excel(writer, sheet_name="Test_summary", index=False)
        individual.to_excel(writer, sheet_name="Individual_values", index=False)
        pd.DataFrame({"mouse_id": mice}).to_excel(
            writer, sheet_name="Mouse_key", index=False
        )
        pd.DataFrame({
            "item": [
                "Test", "Statistic", "Enumeration", "Comparisons",
                "Multiplicity", "Time windows"
            ],
            "definition": [
                "Two-sided paired exact sign-flip permutation test",
                "Mean within-mouse difference",
                f"All 2^{len(mice)} = {2**len(mice):,} sign assignments",
                "After vs Before; 1 h after vs Before",
                (
                    "Nominal exact P values; no MaxT or Bonferroni adjustment. "
                    "Under a prespecified fixed sequence, the second comparison "
                    "is confirmatory only if the first is significant."
                ),
                "Before -45 to 0 min; After 0 to 45 min; 1 h after 45 to 120 min",
            ],
        }).to_excel(writer, sheet_name="Notes", index=False)

    return summary


def add_significance_bracket(ax, x1, x2, y, text):
    h = max(abs(y) * 0.015, 1e-6)
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black")
    ax.text((x1+x2)/2, y+1.5*h, text, ha="center", va="bottom")


def plot_metric(df, mice, metric_key, metric_label, output_dir):
    pivot = build_pivot(df, metric_key, mice)
    x_i = np.array([0.0, 1.0, 2.0])
    x_m = np.array([4.0, 5.0, 6.0])

    fig, ax = plt.subplots(figsize=(8, 6))
    for mouse in mice:
        ax.plot(x_i, pivot.loc[mouse, ["I_Before", "I_After", "I_1h"]],
                marker="o", color="dimgray", alpha=0.40, linewidth=0.9)
        ax.plot(x_m, pivot.loc[mouse, ["M_Before", "M_After", "M_1h"]],
                marker="o", color="sandybrown", alpha=0.45, linewidth=0.9)

    ax.errorbar(x_i, pivot[["I_Before", "I_After", "I_1h"]].mean(),
                yerr=pivot[["I_Before", "I_After", "I_1h"]].sem(),
                marker="o", linewidth=2.5, capsize=4, color="black", label="Immobile")
    ax.errorbar(x_m, pivot[["M_Before", "M_After", "M_1h"]].mean(),
                yerr=pivot[["M_Before", "M_After", "M_1h"]].sem(),
                marker="o", linewidth=2.5, capsize=4, color="darkorange", label="Mobile")

    tests = {
        "I_BA": exact_sign_flip_test(pivot["I_After"] - pivot["I_Before"]),
        "I_B1": exact_sign_flip_test(pivot["I_1h"] - pivot["I_Before"]),
        "M_BA": exact_sign_flip_test(pivot["M_After"] - pivot["M_Before"]),
        "M_B1": exact_sign_flip_test(pivot["M_1h"] - pivot["M_Before"]),
    }

    values = pivot.to_numpy(dtype=float)
    ymin, ymax = float(np.nanmin(values)), float(np.nanmax(values))
    yrange = ymax-ymin if ymax>ymin else 1.0
    ax.set_ylim(min(0.0, ymin-0.08*yrange), ymax+0.34*yrange)
    base = ymax+0.07*yrange
    add_significance_bracket(ax, x_i[0], x_i[1], base,
                             significance_label(float(tests["I_BA"]["exact_p"])))
    add_significance_bracket(ax, x_i[0], x_i[2], base+0.10*yrange,
                             significance_label(float(tests["I_B1"]["exact_p"])))
    add_significance_bracket(ax, x_m[0], x_m[1], base,
                             significance_label(float(tests["M_BA"]["exact_p"])))
    add_significance_bracket(ax, x_m[0], x_m[2], base+0.10*yrange,
                             significance_label(float(tests["M_B1"]["exact_p"])))

    ax.set_xticks(np.concatenate([x_i, x_m]))
    ax.set_xticklabels(["I\nBefore", "I\nAfter", "I\n1 h",
                        "M\nBefore", "M\nAfter", "M\n1 h"])
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} (n = 10)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_label.lower().replace(' ', '_')}_N10.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    validate_input(df)
    df["condition"] = df.apply(condition_name, axis=1)
    df = df.dropna(subset=["condition"]).copy()

    mice = select_n10_mice(df)
    create_test_tables(df, mice, args.output_dir)
    for metric_key, metric_label in METRICS:
        plot_metric(df, mice, metric_key, metric_label, args.output_dir)

    print(f"Completed. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
