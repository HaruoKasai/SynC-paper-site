from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

CSV_PATH = Path(r"\\Synology\arima\ELITE_SE880\cLTP\_summary\20min_interval_c1_delta(%).csv")
PDF_PATH = CSV_PATH.with_name("pre_stim_1h_bar_strip.pdf")

GROUPS = {
    "pre": [-50, -30, -10],
    "stim ~1h": [25, 45, 65],
}


def main():
    df = pd.read_csv(CSV_PATH)

    date_cols = [
        col for col in df.columns
        if col not in {"min", "average", "sem"} and not str(col).startswith("Unnamed")
    ]
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df[date_cols] = df[date_cols].apply(pd.to_numeric, errors="coerce")

    plot_rows = []
    for group_name, minutes in GROUPS.items():
        subset = df[df["min"].isin(minutes)]
        if len(subset) != len(minutes):
            found = sorted(subset["min"].dropna().astype(int).tolist())
            raise ValueError(f"{group_name}: expected {minutes}, found {found}")

        per_date = subset[date_cols].mean(axis=0)
        for date, value in per_date.items():
            plot_rows.append({"group": group_name, "date": date, "value": value})

    plot_df = pd.DataFrame(plot_rows)
    group_order = list(GROUPS.keys())
    bar_means = plot_df.groupby("group")["value"].mean().reindex(group_order)
    bar_sem = plot_df.groupby("group")["value"].sem().reindex(group_order)

    rng = np.random.default_rng(4)
    x = np.arange(len(group_order))

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)
    ax.bar(
        x,
        bar_means,
        yerr=bar_sem,
        width=0.58,
        color=["#b9d7d0", "yellow"],
        edgecolor="#222222",
        linewidth=1.2,
        capsize=5,
        zorder=1,
    )

    marker_map = {
        date: marker
        for date, marker in zip(date_cols, ["o", "s", "^", "D", "v", "P", "X", "*"])
    }

    for i, group_name in enumerate(group_order):
        group_points = plot_df[plot_df["group"] == group_name]
        jitter = rng.normal(0, 0.045, size=len(group_points))
        for j, (_, row) in enumerate(group_points.iterrows()):
            ax.scatter(
                i + jitter[j],
                row["value"],
                s=48,
                marker=marker_map[row["date"]],
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.0,
                zorder=3,
                label=row["date"] if i == 0 else None,
            )

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, group_order)
    ax.set_ylabel("c0 delta (%)")
    ax.set_title("Pre vs stim ~1h")
    ax.legend(title="Date", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(PDF_PATH, bbox_inches="tight")
    print(f"Saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
