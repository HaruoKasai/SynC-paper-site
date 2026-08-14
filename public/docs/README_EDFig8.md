# Extended Data Figure 8 reproducibility record

This package publishes the laser-chasing and food-approaching behavioural
analyses used for the manuscript-facing Extended Data Fig. 8, together with a
demonstration dataset sufficient to run both pipelines end to end.

Two independent goal-directed approach tasks are analysed: approach towards a
moving laser spot, and approach towards a food pellet. Both are evaluated in
three time windows surrounding A/C administration and compared between Control
and SynC animals.

## Package layout

```
code/
  EDFig8_laser.py
  EDFig8_food.py
data/
  edf8_demo/
    _Group_analysis/
      _group_analysis.json     <- group definition; all outputs are written here
      .gitignore               <- excludes generated PDFs and CSVs
    Ctrl_WT/
      WT_demo/
        01 ... 08/             <- recording segments
        _analysis_param.json
    SynC/
      SynC_demo/
        01 ... 07/
        _analysis_param.json
```

Both scripts resolve their input relative to their own location: they look for
`_group_analysis.json` in `code/../data/edf8_demo/_Group_analysis/`, falling
back to the first `*.json` in that directory if it is absent. No paths need to
be edited before running, and a single group-definition file serves both
analyses.

Generated PDFs and CSVs are written into `_Group_analysis/` and are excluded
from version control by the `.gitignore` in that directory; only the
group-definition JSON is tracked.

## Input data

`_group_analysis.json` lists the mice belonging to each experimental group.
Mouse paths inside the file are resolved relative to `data/edf8_demo/`.

For each mouse, the following per-segment files are read from
`<mouse>/<segment>/_DLC_analysis/`:

| File | Used by |
| --- | --- |
| `dlc_data.csv` | both - DeepLabCut body-part coordinates and likelihoods |
| `laser_timepoints.json` | laser - laser onset times and laser-spot coordinates |
| `light_rising_frames_top.csv`, `light_rising_frames_side.csv` | laser - top/side camera synchronisation |
| `food_location_frames_ep{N}.csv` | food - per-frame pellet position for epoch *N* |

Segment start times and continuity information are read from
`<mouse>/_analysis_param.json`. Epochs lacking the corresponding food-location
file are skipped, as are segments whose `laser_timepoints.json` contains no
laser events.

The demonstration dataset contains one animal per experimental group and is a
subset of the frozen source; body-part coordinates, event times and segment
timing are unchanged from the values used in the manuscript.

## Time windows

Every stimulus event is assigned to a time window according to its time
relative to A/C administration. Windows are half-open intervals
`[start, end)`; where they overlap, a single event may contribute to more than
one window.

| Analysis | Window | Interval |
| --- | --- | --- |
| Laser | Before A/C | -30 to 0 min |
| Laser | After A/C 0-50 | 0 to 50 min |
| Laser | After A/C 60 | 60 min onwards |
| Food | Before | before 0 min |
| Food | After 0-60 | 0 to 60 min |
| Food | After 60+ | 60 min onwards |

Event time is computed as the segment start time plus the within-segment frame
offset, so a window may be narrower than a segment or span a segment boundary.

## Laser chasing

For each laser onset, the snout-to-laser distance is computed frame by frame
over a fixed window of -5 to +10 s relative to onset, and converted from pixels
to millimetres. Two measures are reported: **distance to laser** and
**velocity**.

For each measure the area under the curve is computed over a configurable
window relative to laser onset. A *smaller* distance AUC indicates that the
animal spent more time close to the laser, that is, a stronger approach; this
is the opposite direction from velocity AUC.

| Parameter | Value |
| --- | ---: |
| `laser_zone_radius_px` | 75 |
| `orientation_thresh_deg` | 30 |
| `speed_smooth_window_s` | 0.20 |
| `trajectory_bodypart` | `snout` |
| `stats_basis` | `mouse` |
| `plot_n_points` | 15 |
| `auc_t_start_s` | 5.0 |
| `auc_t_end_s` | 10.0 |
| `line_plot_ymax` | 200 |

`speed_smooth_window_s` is retained for reference only; velocity uses a fixed
4-frame rolling window matching the upstream DLC analysis. `plot_n_points` = 15
corresponds to 1.0 s per point across the 15-s window.

## Food approaching

For each food-placement event, the distance between the selected body part and
the per-frame pellet position is computed as

$$
d_t = \sqrt{(x_t - x_{\mathrm{food},t})^2 + (y_t - y_{\mathrm{food},t})^2},
$$

over a pre-placement and post-placement window relative to placement
(*t* = 0). Distance and velocity are reported, each with an AUC over a
configurable window. Detected eating onset is marked below the x-axis of the
distance plots. Frames whose tracking likelihood falls below the retained
threshold are interpolated out before the distance is computed.

| Parameter | Value |
| --- | ---: |
| `px_to_mm` | 0.76 |
| `speed_smooth_s` | 0.20 |
| `pre_window_s` | 30 |
| `post_window_s` | 90 |
| `distance_bodypart` | `head_center` |
| `velocity_bodypart` | `body_center` |
| `plot_points` | 24 |
| `auc_start_s` | 30.0 |
| `auc_end_s` | 90.0 |
| `line_plot_ymax` | 200 |

Setting `px_to_mm` to `None` keeps pixel units throughout; the output filenames
record the scale that was used.

## Statistics basis

All group-level curves and bars are computed on a per-mouse-average basis: each
animal contributes one averaged trace or value per time window, and the group
mean +/- s.e.m. is computed across animals, so *n* is the number of mice.
Individual animals are additionally shown as thin lines and overlaid dots.

The laser script also supports a per-epoch basis, in which every stimulus event
contributes independently and *n* is the number of epochs. Output filenames are
suffixed with the basis so that the two do not overwrite one another. The
manuscript reports the per-mouse basis.

No hypothesis tests are performed by these scripts. The published CSVs contain
the values used to draw each panel, so the figures can be regenerated or
replotted without re-running the analysis.

## Parameters and switches

All analysis parameters are collected in a `PARAMS` dictionary at the top of
each script; the distributed values are those used in the manuscript and are
also the defaults shown in the parameter window. Two switches are available in
both scripts:

- `use_gui` - when `False` (default) the script runs immediately using the
  parameters above. When `True`, a window opens in which the parameters can be
  adjusted before running.
- `make_trajectory_pdf` - when `True`, an additional PDF is written showing the
  movement trajectory of each individual epoch, including the pellet trajectory
  for the food analysis. These figures are not used in the manuscript and are
  disabled by default.

## Outputs

All outputs are written to `data/edf8_demo/_Group_analysis/`.

`EDFig8_laser.py` generates:

- `laser_response_overview_mouse.pdf`: distance and velocity time courses for
  each time window, with the corresponding AUC comparisons across time windows
  and experimental groups.
- `laser_response_curves_mouse.csv`: the plotted curves in long format, one row
  per time point for each animal and for the group mean with s.e.m.
- `laser_response_auc_mouse.csv`: individual AUC values with the corresponding
  group mean and s.e.m.

`EDFig8_food.py` generates:

- `food_approaching_analysis_<part>_<scale>.pdf`: distance and velocity time
  courses for each time window, with AUC summaries.
- `food_curves_<part>_<scale>.csv`: the plotted curves in long format.
- `food_auc_<part>_<scale>.csv`: individual AUC values together with detected
  eating-onset times.

If `make_trajectory_pdf` is enabled, an additional `*_trajectories*.pdf` is
written alongside these files.

## Running the analysis

From the package root:

```bash
pip install "numpy>=1.24,<3" "pandas>=2.0" "matplotlib>=3.7"
python code/EDFig8_laser.py
python code/EDFig8_food.py
```

Either script can also be run by opening it directly in an editor such as
PyCharm or VS Code. To open the parameter window instead, set
`PARAMS["use_gui"] = True` or pass `--gui`. A specific group-definition file
may be given as an argument:

```bash
python code/EDFig8_laser.py data/edf8_demo/_Group_analysis/_group_analysis.json
```

Both pipelines have been run end to end on the demonstration dataset
distributed here; the files listed under **Outputs** are produced without
further configuration.

The analysis requires Python 3.10+, NumPy, pandas and Matplotlib; no other
third-party packages are used. The pipelines are compatible with both NumPy 1.x
and 2.x. On some Linux distributions the standard-library `tkinter` module must
be installed separately.