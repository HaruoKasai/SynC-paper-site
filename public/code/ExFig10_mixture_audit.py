#!/usr/bin/env python3
"""Audit the frozen Extended Data Fig. 10 Normal-Exponential mixture.

The latent positive component is Exponential because the fitted Gamma shape is
exactly one. The displayed positive density is its convolution with the
zero-centred Normal measurement distribution. This script recalculates both
densities, spine-level posterior probabilities, and condition-specific mixture
fractions from the public frozen endpoint table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import exponnorm, norm


def _parameter_map(parameters: pd.Series) -> dict[str, float]:
    return {
        "WT": float(parameters["pi_WT"]),
        "SynC before": float(parameters["pi_SynC before"]),
        "SynC 0-60": float(parameters["pi_SynC 0-60"]),
        "SynC 60-180": float(parameters["pi_SynC 60-180"]),
        "dGAP before": float(parameters["pi_dGAP before"]),
        "dGAP 0-60": float(parameters["pi_dGAP 0-60"]),
        "dGAP 60-180": float(parameters["pi_dGAP 60-180"]),
    }


def _densities(
    values: np.ndarray,
    sigma: np.ndarray,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    null_density = norm.pdf(values, loc=0.0, scale=sigma)
    # scipy's exponnorm is Normal(0, sigma) + Exponential(theta).
    positive_density = exponnorm.pdf(
        values,
        theta / sigma,
        loc=0.0,
        scale=sigma,
    )
    return null_density, positive_density


def _posterior(
    pi_value: float,
    null_density: np.ndarray,
    positive_density: np.ndarray,
) -> np.ndarray:
    numerator = pi_value * positive_density
    denominator = (1.0 - pi_value) * null_density + numerator
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )


def _refit_pi(
    null_density: np.ndarray,
    positive_density: np.ndarray,
) -> float:
    def negative_log_likelihood(pi_value: float) -> float:
        density = (1.0 - pi_value) * null_density + pi_value * positive_density
        return -float(np.sum(np.log(np.maximum(density, 1.0e-300))))

    result = minimize_scalar(
        negative_log_likelihood,
        bounds=(1.0e-9, 1.0 - 1.0e-9),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not result.success:
        raise RuntimeError(f"Mixture-fraction fit failed: {result.message}")
    return float(result.x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spine_csv",
        nargs="?",
        type=Path,
        default=Path("Fig6_ExFig10_spine_input.csv"),
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        default=Path("Fig6_ExFig10_mixture_parameters.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ExFig10_mixture_audit.csv"),
    )
    args = parser.parse_args()

    data = pd.read_csv(args.spine_csv)
    parameters = pd.read_csv(args.parameters).iloc[0]
    shape = float(parameters["gamma_shape"])
    if not np.isclose(shape, 1.0, atol=1.0e-12):
        raise ValueError("This audit expects Gamma shape = 1 (Exponential).")
    theta = float(parameters["gamma_scale_theta_percent"])
    pi_by_group = _parameter_map(parameters)

    stim = data.loc[data["role"].eq("stim")].copy()
    values = stim["corrected_delta_v_40_80_percent"].to_numpy(float)
    sigma = np.full(len(stim), float(parameters["null_sigma_percent"]))
    null_density, positive_density = _densities(values, sigma, theta)
    stim["normal_density"] = null_density
    stim["normal_convolved_exponential_density"] = positive_density

    recalculated = np.empty(len(stim), dtype=float)
    summaries: list[dict[str, float | int | str]] = []
    for group, indices in stim.groupby("group").groups.items():
        positions = stim.index.get_indexer(indices)
        group_null = null_density[positions]
        group_positive = positive_density[positions]
        frozen_pi = pi_by_group[str(group)]
        group_posterior = _posterior(frozen_pi, group_null, group_positive)
        recalculated[positions] = group_posterior
        summaries.append(
            {
                "group": group,
                "n_spines": len(positions),
                "frozen_pi": frozen_pi,
                "refitted_pi_with_frozen_components": _refit_pi(
                    group_null, group_positive
                ),
                "mean_recalculated_posterior": float(group_posterior.mean()),
            }
        )

    stim["posterior_recalculated"] = recalculated
    stim["posterior_absolute_difference"] = np.abs(
        stim["posterior_recalculated"] - stim["posterior_permissive"]
    )
    max_difference = float(stim["posterior_absolute_difference"].max())
    if max_difference > 1.0e-3:
        raise RuntimeError(
            "Frozen posterior audit exceeded tolerance: "
            f"maximum absolute difference = {max_difference:.6g}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).sort_values("group").to_csv(args.output, index=False)
    spine_output = args.output.with_name(f"{args.output.stem}_spines.csv")
    stim.to_csv(spine_output, index=False)
    fov_output = args.output.with_name(f"{args.output.stem}_fov.csv")
    (
        stim.groupby(["group", "mouse_id", "fov_id"], as_index=False)
        .agg(
            n_spines=("spine_id", "size"),
            fov_mean_posterior_score=("posterior_permissive", "mean"),
        )
        .to_csv(fov_output, index=False)
    )
    print(pd.DataFrame(summaries).sort_values("group").to_string(index=False))
    print(f"Maximum posterior absolute difference: {max_difference:.6g}")


if __name__ == "__main__":
    main()
