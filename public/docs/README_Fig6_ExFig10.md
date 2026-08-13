# Figure 6 and Extended Data Figure 10 reproducibility record

This package publishes the frozen 40-80-s spine-enlargement endpoint, FOV
summaries, fitted mixture parameters, and the 10,000-replicate two-sided test
results used for the manuscript-facing Fig. 6 and Extended Data Fig. 10.

## Frozen source

- Freeze version: `tyx_fig6_f2_g3_frozen_integrated_20260810_v4`
- Source commit: `9c6a8a01a53769e9bec234886af31654bd60ee6e`
- Frozen ROI rows: 1,088
- Stimulated: 565 spines, 158 FOVs, 11 mice
- Neighbouring: 523 spines, 135 FOVs, 9 mice

SHA-256 hashes of the internal frozen inputs are:

| Internal frozen file | SHA-256 |
| --- | --- |
| `fov_summary.csv` | `FB81B66BB0C4ACB4ABFF23F51B0504456821C2F7B2B98C8F26133117803332DB` |
| `spine_endpoint_40_80_and_posterior.csv` | `BC47A7E3231DA81491172E25C12652B970845424FBDB1402718B0CC9D832FC47` |
| `mixture_parameters.csv` | `E431E8CB0D6B6ED84EF4AF0607AC3F2860BBB0D975A333ADECE2581A5827336E` |
| 10,000-replicate two-sided result CSV | `D503AC1D219FC3AB9E0714B9AEDF73250C02895A80DC9D248186F233E1E5557D` |

The public CSVs exclude local filesystem paths and redundant internal columns.
Original mouse, FOV, and spine labels are replaced by stable opaque aliases.
These aliases preserve the mouse/FOV/spine hierarchy across files, but the
private correspondence table is not distributed. Endpoint values,
condition-specific posterior probabilities, and group mixture fractions are
unchanged. The Fig. 6h FOV table additionally contains the documented
common-prior posterior-score sensitivity endpoint.

## Cohort

| Condition | Stimulated spines | FOVs | Mice |
| --- | ---: | ---: | ---: |
| WT | 82 | 24 | 2 |
| SynC -A/C | 110 | 30 | 5 |
| SynC +A/C 0-60 min | 55 | 16 | 5 |
| SynC +A/C 60-180 min | 115 | 33 | 5 |
| dGAP -A/C | 79 | 21 | 4 |
| dGAP +A/C 0-60 min | 54 | 14 | 4 |
| dGAP +A/C 60-180 min | 70 | 20 | 3 |

## Image quantification and endpoint

Spine-head fluorescence was locally background-subtracted and corrected for
field-wide and local intensity fluctuations using an equal-weight combination
of ROI-external global and local reference signals from the same FOV. Once
specified, the same pipeline was applied to stimulated and neighbouring ROIs
in every genotype and drug condition. Genotype, drug condition, ROI role, and
response magnitude were not inputs during application of the correction.

The primary endpoint is mean Delta V from 40 to 80 s after stimulation. No
ROI-response-derived early rescue was used for this endpoint. Measurements
during optical stimulation (0-4 s) were omitted because of stimulation
artefacts. Binning in Fig. 6c-f is for display only; endpoint estimation and
testing use retained unbinned measurements.

## Figure 6g: continuous endpoint

Each FOV contributes one equally weighted mean. SynC and dGAP contrasts use a
two-sided heteroscedastic Normal parametric bootstrap of FOV means with a mouse
random intercept. Group-specific residual SDs are estimated, so
`heteroscedastic` does not merely refer to comparing multiple groups.

The adopted run uses 10,000 replicates and the following seeds:

- SynC Delta V: `2026072801`
- dGAP Delta V: `2026073001`
- SynC permissive fraction: `2026072901`
- dGAP permissive fraction: `2026073101`

The multiplicity policy is:

1. SynC before versus 0-60 min is the single prespecified primary contrast and
   is reported without multiplicity adjustment.
2. The SynC 60-180-min versus 0-60-min recovery contrast is secondary and is
   reported without multiplicity adjustment.
3. dGAP contrasts are two-sided descriptive comparisons without adjustment.

The displayed P values are:

| Metric | Contrast | P |
| --- | --- | ---: |
| Delta V | SynC before vs 0-60 min | 0.0488 |
| Delta V | SynC 60-180 vs 0-60 min | 0.0059 |
| Delta V | dGAP before vs 0-60 min | 0.7620 |
| Delta V | dGAP 0-60 vs 60-180 min | 0.4814 |

Run the public implementation from a directory containing the Python file and
FOV CSV:

```bash
python Fig6_FOV_parametric_bootstrap.py Fig6_ExFig10_FOV_input.csv
```

Dependencies are Python 3.12+, NumPy, and pandas.

## Figure 6h and Extended Data Figure 10: mixture model

The pooled neighbouring-spine endpoints define a zero-centred Normal null with
sigma = 9.3740%. The latent positive component is Gamma with fitted shape 1.0;
it is therefore Exponential with scale theta = 34.4291%. The observed positive
density is the convolution of that Exponential component with Normal
measurement noise.

Condition-specific fitted positive-component fractions are:

| Condition | pi |
| --- | ---: |
| WT | 37.64% |
| SynC before | 26.57% |
| SynC 0-60 min | 4.14% |
| SynC 60-180 min | 18.90% |
| dGAP before | 23.69% |
| dGAP 0-60 min | 26.58% |
| dGAP 60-180 min | 23.01% |

For the condition-independent sensitivity analysis in Fig. 6h, spine-level
posterior scores are recalculated with a common prior mixture fraction
(`pi = 0.241672`) and averaged within FOV. These FOV means are compared by the
same heteroscedastic Normal parametric-bootstrap structure as the continuous
endpoint, including the mouse random intercept. Bootstrap samples are generated
at the FOV level. Individual spine responses are not regenerated from the
mixture distribution, and the mixture parameters and posterior scores are held
fixed rather than refitted inside every replicate.

The Fig. 6h bars show the condition-specific model-estimated fractions listed
above; the significance annotations report the separate common-prior
posterior-score sensitivity analysis.

| Contrast | P |
| --- | ---: |
| SynC before vs 0-60 min | 0.0049 |
| SynC 60-180 vs 0-60 min | 0.0144 |
| dGAP before vs 0-60 min | 0.7952 |
| dGAP 0-60 vs 60-180 min | 0.8823 |

The selected Extended Data Fig. 10 uses one-percentile percentograms. Bin
widths vary so that each bin contains approximately equal numbers of points;
the height is `count / (sample size x bin width)`, preserving density area.
Percentograms are visualisation only. Model fitting, posterior calculation,
and testing use unbinned endpoints.

Run the mixture audit with:

```bash
python ExFig10_mixture_audit.py Fig6_ExFig10_spine_input.csv \
  --parameters Fig6_ExFig10_mixture_parameters.csv
```

This script requires NumPy, pandas, and SciPy. It validates the frozen posterior
probabilities against the Normal-convolved Exponential density and reports the
condition-specific mixture fractions.

## Public files

- `Fig6_ExFig10_FOV_input.csv`: one stimulated-spine summary row per FOV.
- `Fig6_ExFig10_spine_input.csv`: path-sanitised frozen ROI endpoint table.
- `Fig6_ExFig10_cohort_counts.csv`: role/group cohort counts.
- `Fig6_ExFig10_mixture_parameters.csv`: frozen joint model parameters.
- `Fig6_ExFig10_reported_tests.csv`: adopted 10,000-replicate Fig. 6g/h
  SynC and dGAP results.

WT rows and the frozen `pi_WT` parameter remain in the spine-level source and
mixture-model files because WT is displayed in Extended Data Fig. 10. No WT
contrast is reported in the Fig. 6g/h hypothesis-test output.
