#!/usr/bin/env python3
"""Recompute the adopted Fig. 6g/h FOV-level comparisons.

The input contains one equally weighted row per FOV. The SynC comparisons use
a heteroscedastic Normal parametric bootstrap with a mouse random intercept.
For the permissive endpoint, each input row is the within-FOV mean of fixed
spine posterior probabilities calculated using the fitted condition-specific
mixture fraction. The bootstrap generates FOV-level values; it does not
regenerate individual spine responses or refit the mixture model.

The prespecified before-vs-acute contrast and the recovery SynC contrast are
reported without multiplicity adjustment. dGAP contrasts are descriptive
two-sided comparisons. WT remains in the shared source table used for Extended
Data Fig. 10 but is not included in the Fig. 6g/h hypothesis-test output.

The default seeds and 10,000 replicates reproduce the manuscript-facing CSV.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SYNC_GROUPS = ("SynC before", "SynC 0-60", "SynC 60-180")
SYNC_CONTRASTS = np.asarray(
    [
        [1.0, -1.0, 0.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
    ],
    dtype=float,
)
SYNC_IDS = (
    "sync_before_vs_0_60",
    "sync_0_60_vs_60_180",
    "sync_before_vs_60_180",
)
DGAP_COMPARISONS = (
    ("dgap_before_vs_0_60", "dGAP before", "dGAP 0-60"),
    ("dgap_0_60_vs_60_180", "dGAP 0-60", "dGAP 60-180"),
)


def _variance_components(
    frame: pd.DataFrame,
    means: np.ndarray,
    groups: tuple[str, ...],
    metric: str,
) -> tuple[float, dict[str, float]]:
    work = frame.copy()
    mean_map = dict(zip(groups, means, strict=True))
    work["residual"] = work[metric] - work["group"].map(mean_map).astype(float)
    mouse_effect = work.groupby("mouse_id")["residual"].mean()
    within = work["residual"] - work["mouse_id"].map(mouse_effect)

    residual_sd: dict[str, float] = {}
    for group in groups:
        values = within.loc[work["group"].eq(group)].to_numpy(float)
        residual_sd[group] = max(float(np.std(values, ddof=1)), 1.0e-3)

    mouse_counts = work.groupby("mouse_id").size().astype(float)
    mean_sampling_variance = float(
        np.mean(
            [
                np.mean(
                    [
                        residual_sd[group] ** 2
                        for group in work.loc[
                            work["mouse_id"].eq(mouse), "group"
                        ].astype(str)
                    ]
                )
                / count
                for mouse, count in mouse_counts.items()
            ]
        )
    )
    tau2 = max(float(mouse_effect.var(ddof=1)) - mean_sampling_variance, 0.0)
    return math.sqrt(tau2), residual_sd


def _group_mean_covariance(
    frame: pd.DataFrame,
    groups: tuple[str, ...],
    mouse_sd: float,
    residual_sd: dict[str, float],
) -> np.ndarray:
    group_counts = np.asarray(
        [int(frame["group"].eq(group).sum()) for group in groups], dtype=float
    )
    covariance = np.diag(
        [
            residual_sd[group] ** 2 / max(group_counts[index], 1.0)
            for index, group in enumerate(groups)
        ]
    )
    if mouse_sd <= 0.0:
        return covariance

    for mouse in frame["mouse_id"].astype(str).unique():
        mouse_rows = frame.loc[frame["mouse_id"].astype(str).eq(mouse)]
        weights = np.asarray(
            [
                int(mouse_rows["group"].eq(group).sum())
                / max(group_counts[index], 1.0)
                for index, group in enumerate(groups)
            ],
            dtype=float,
        )
        covariance += mouse_sd**2 * np.outer(weights, weights)
    return covariance


def _project_means_gls(
    means: np.ndarray,
    covariance: np.ndarray,
    constraints: np.ndarray,
) -> np.ndarray:
    correction = (
        covariance
        @ constraints.T
        @ np.linalg.pinv(constraints @ covariance @ constraints.T)
    )
    return means - correction @ constraints @ means


def _simulate_group_means(
    frame: pd.DataFrame,
    means: np.ndarray,
    groups: tuple[str, ...],
    mouse_sd: float,
    residual_sd: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    mouse_effects = {
        mouse: rng.normal(0.0, mouse_sd)
        for mouse in frame["mouse_id"].astype(str).unique()
    }
    generated = np.asarray(
        [
            means[groups.index(group)]
            + mouse_effects[str(mouse)]
            + rng.normal(0.0, residual_sd[group])
            for group, mouse in frame[["group", "mouse_id"]].itertuples(
                index=False
            )
        ],
        dtype=float,
    )
    labels = frame["group"].to_numpy(str)
    return np.asarray(
        [generated[labels == group].mean() for group in groups], dtype=float
    )


def _step_down(
    *,
    active_indices: list[int],
    observed: np.ndarray,
    means: np.ndarray,
    covariance: np.ndarray,
    contrasts: np.ndarray,
    statistics,
    selected: pd.DataFrame,
    groups: tuple[str, ...],
    mouse_sd: float,
    residual_sd: dict[str, float],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(active_indices, key=lambda index: observed[index], reverse=True)
    stage_p = np.full(len(observed), np.nan)
    adjusted = np.full(len(observed), np.nan)
    previous = 0.0
    for stage, current in enumerate(order):
        active = order[stage:]
        constrained = _project_means_gls(
            means, covariance, contrasts[active]
        )
        exceedances = 0
        for _ in range(repetitions):
            simulated_means = _simulate_group_means(
                selected,
                constrained,
                groups,
                mouse_sd,
                residual_sd,
                rng,
            )
            simulated = statistics(contrasts @ simulated_means)
            exceedances += int(np.max(simulated[active]) >= observed[current])
        current_p = (exceedances + 1.0) / (repetitions + 1.0)
        stage_p[current] = current_p
        adjusted[current] = max(previous, current_p)
        previous = adjusted[current]
    return stage_p, adjusted


def _sync_tests(
    fov: pd.DataFrame,
    *,
    metric: str,
    effect_scale: float,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    selected = fov.loc[fov["group"].isin(SYNC_GROUPS)].dropna(subset=[metric]).copy()
    means = np.asarray(
        [
            selected.loc[selected["group"].eq(group), metric].mean()
            for group in SYNC_GROUPS
        ],
        dtype=float,
    )
    effects = SYNC_CONTRASTS @ means
    mouse_sd, residual_sd = _variance_components(
        selected, means, SYNC_GROUPS, metric
    )
    covariance = _group_mean_covariance(
        selected, SYNC_GROUPS, mouse_sd, residual_sd
    )
    rng = np.random.default_rng(seed)

    pilot = np.vstack(
        [
            SYNC_CONTRASTS
            @ _simulate_group_means(
                selected,
                means,
                SYNC_GROUPS,
                mouse_sd,
                residual_sd,
                rng,
            )
            for _ in range(1_000)
        ]
    )
    scale = np.maximum(pilot.std(axis=0, ddof=1), 1.0e-6)

    def statistics(values: np.ndarray) -> np.ndarray:
        return np.abs(values / scale)

    observed = statistics(effects.copy())
    # Preserve the documented simulation stream used for the frozen result.
    # These maxT draws are not reported or used for the adopted P values.
    _step_down(
        active_indices=[0, 1, 2],
        observed=observed,
        means=means,
        covariance=covariance,
        contrasts=SYNC_CONTRASTS,
        statistics=statistics,
        selected=selected,
        groups=SYNC_GROUPS,
        mouse_sd=mouse_sd,
        residual_sd=residual_sd,
        repetitions=repetitions,
        rng=rng,
    )

    single_p = np.full(3, np.nan)
    for current in (0, 1):
        constrained = _project_means_gls(
            means, covariance, SYNC_CONTRASTS[[current]]
        )
        exceedances = 0
        for _ in range(repetitions):
            simulated_means = _simulate_group_means(
                selected,
                constrained,
                SYNC_GROUPS,
                mouse_sd,
                residual_sd,
                rng,
            )
            simulated = statistics(SYNC_CONTRASTS @ simulated_means)
            exceedances += int(simulated[current] >= observed[current])
        single_p[current] = (exceedances + 1.0) / (repetitions + 1.0)

    return pd.DataFrame(
        [
            {
                "metric": metric,
                "contrast_id": SYNC_IDS[index],
                "alternative": "two-sided",
                "effect": effect_scale * effects[index],
                "display_p": single_p[index],
                "seed": seed,
                "repetitions": repetitions,
                "inference_role": "primary" if index == 0 else "secondary",
            }
            for index in (0, 1)
        ]
    )


def _pairwise_tests(
    fov: pd.DataFrame,
    *,
    metric: str,
    effect_scale: float,
    comparisons: Iterable[tuple[str, str, str]],
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for contrast_id, first, second in comparisons:
        groups = (first, second)
        selected = fov.loc[fov["group"].isin(groups)].dropna(subset=[metric]).copy()
        means = np.asarray(
            [
                selected.loc[selected["group"].eq(group), metric].mean()
                for group in groups
            ],
            dtype=float,
        )
        contrast = np.asarray([[1.0, -1.0]])
        mouse_sd, residual_sd = _variance_components(
            selected, means, groups, metric
        )
        covariance = _group_mean_covariance(
            selected, groups, mouse_sd, residual_sd
        )
        constrained = _project_means_gls(means, covariance, contrast)
        observed = float(means[0] - means[1])
        exceedances = 0
        for _ in range(repetitions):
            simulated = _simulate_group_means(
                selected,
                constrained,
                groups,
                mouse_sd,
                residual_sd,
                rng,
            )
            exceedances += int(abs(simulated[0] - simulated[1]) >= abs(observed))
        p_value = (exceedances + 1.0) / (repetitions + 1.0)
        rows.append(
            {
                "metric": metric,
                "contrast_id": contrast_id,
                "alternative": "two-sided",
                "effect": effect_scale * observed,
                "display_p": p_value,
                "seed": seed,
                "repetitions": repetitions,
                "inference_role": "descriptive",
            }
        )
    return pd.DataFrame(rows)


def _run_metric(
    fov: pd.DataFrame,
    *,
    metric: str,
    effect_scale: float,
    sync_seed: int,
    dgap_seed: int,
    repetitions: int,
) -> pd.DataFrame:
    return pd.concat(
        [
            _sync_tests(
                fov,
                metric=metric,
                effect_scale=effect_scale,
                repetitions=repetitions,
                seed=sync_seed,
            ),
            _pairwise_tests(
                fov,
                metric=metric,
                effect_scale=effect_scale,
                comparisons=DGAP_COMPARISONS,
                repetitions=repetitions,
                seed=dgap_seed,
            ),
        ],
        ignore_index=True,
        sort=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("Fig6_ExFig10_FOV_input.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Fig6_ExFig10_recomputed_tests.csv"),
    )
    parser.add_argument("--repetitions", type=int, default=10_000)
    args = parser.parse_args()

    fov = pd.read_csv(args.input_csv)
    delta = _run_metric(
        fov,
        metric="mean_delta_v_40_80_percent",
        effect_scale=1.0,
        sync_seed=2026072801,
        dgap_seed=2026073001,
        repetitions=args.repetitions,
    )
    permissive = _run_metric(
        fov,
        metric="fov_mean_posterior_score",
        effect_scale=100.0,
        sync_seed=2026072901,
        dgap_seed=2026073101,
        repetitions=args.repetitions,
    )
    results = pd.concat([delta, permissive], ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results[["metric", "contrast_id", "effect", "display_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
