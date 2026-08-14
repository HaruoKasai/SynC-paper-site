"""
Group Analysis - Laser Response in Openfield (v2)
==================================================
GUI-based analysis tool for analyzing mouse responses to laser stimulation.

New in v2:
  - Time groups are no longer hard-coded. They are editable in the GUI as a
    list of (name, start_min, end_min) rows. `end_min` may be left blank to
    mean "open-ended" (everything from start_min onward). Groups of different
    widths (e.g. 10 min and 30 min) can be freely mixed.
  - A "Statistics basis" toggle (Per-Mouse Average / Per-Epoch) controls how
    ALL plots and the stats CSV are computed for this run:
      * Per-Mouse Average: each mouse contributes one averaged value/trace,
        then group mean +/- SEM is computed across mice (n = n_mice).
      * Per-Epoch: every laser event (epoch) contributes its own value/trace,
        group mean +/- SEM is computed across all epochs pooled together
        (n = n_epochs).
    Outputs that depend on the chosen basis are suffixed with "_mouse" or
    "_epoch" so that re-running with the other basis does not overwrite the
    first run. Per-mouse-average CSV, per-epoch detail CSV, and the
    trajectory PDF are unaffected by this toggle and are always generated.

Workflow:
1. Select main folder (containing group JSON)
2. Configure parameters, time groups, and statistics basis in GUI
3. Run analysis -> PDF + CSV outputs
"""

import os, json, glob, math, warnings
from pathlib import Path
from collections import defaultdict
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════
FPS = 20
PULSE_INTERVAL_SEC = 10.0

PX_TO_MM = 250 / 345  # 345px = 25cm = 250mm

# Default time groups shown when the GUI starts (fully editable by the user).
# Format: name -> (start_min, end_min). end_min = None means "open-ended"
# (i.e. every segment whose start time is >= start_min is included).
DEFAULT_TIME_GROUPS = None   # replaced by TIME_GROUPS below (defined with PARAMS)

LASER_POINT_KEYS = ["up-left", "up-right", "down-left", "down-right"]

# Body part used for trajectory plots (snout or head_center)
VALID_BODY_PARTS = ["snout", "head_center"]


# ===============================================================
#  ANALYSIS PARAMETERS  --  values used in the manuscript
#  These are also the defaults shown in the GUI.  Edit here (not in
#  the GUI code) if the analysis settings ever need to change.
# ===============================================================
PARAMS = {
    # -- 2. Analysis Parameters --------------------------------------
    "laser_zone_radius_px":   75,     # laser zone radius (px)
    "orientation_thresh_deg": 30,     # head-orientation agreement threshold
    "speed_smooth_window_s":  0.20,   # (unused: velocity uses fixed roll=4 frames)
    "trajectory_bodypart":    "snout",
    "stats_basis":            "mouse",  # "mouse" = Per-Mouse Average, "epoch" = Per-Epoch
    "plot_n_points":          15,     # 15 pts over -5~+10 s  ->  1.0 s/point
    "auc_t_start_s":          5.0,    # AUC window start (s, relative to laser ON)
    "auc_t_end_s":            10.0,   # AUC window end   (s)
    # -- plotting ----------------------------------------------------
    "line_plot_ymax":         200.0,  # y-axis upper limit for ALL line plots
    # -- how to launch -----------------------------------------------
    # False = run straight away with the parameters above (no window).
    # True  = open the GUI so the parameters can be adjusted by hand.
    "use_gui":                False,
    # -- optional output ---------------------------------------------
    # Trajectory PDF (one panel per laser epoch, per mouse).  Not used in
    # the manuscript, so it is OFF by default.  Set to True to generate it.
    "make_trajectory_pdf":    False,
}

# -- 4. Time Groups (minutes; None = open-ended) ---------------------
TIME_GROUPS = {
    "Before A/C":     (-30, 0),
    "After A/C 0-50": (0,  50),
    "After A/C 60":   (60, None),
}

# -- Input / output locations ---------------------------------------
# Layout assumed by this script:
#     <repo>/code/03_laser_chasing_analysis_v2.py   <- this file
#     <repo>/data/edf8_demo/_Group_analysis/*.json  <- config + outputs
CODE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = CODE_DIR.parent / "data" / "edf8_demo"
CONFIG_DIR  = DATA_DIR / "_Group_analysis"
OUTPUT_DIR  = CONFIG_DIR              # all outputs are written next to the JSON


DEFAULT_TIME_GROUPS = TIME_GROUPS   # backwards-compatible alias


def find_default_config() -> Optional[Path]:
    """Locate the group-definition JSON inside CONFIG_DIR."""
    p = CONFIG_DIR / "_group_analysis.json"
    if p.exists():
        return p
    cands = sorted(CONFIG_DIR.glob("*.json"))
    return cands[0] if cands else None

# ═══════════════════════════════════════════════════════════════
#  TIME SYNC UTILITIES
# ═══════════════════════════════════════════════════════════════

def sec_to_top_frame(sec: float, side_frames: np.ndarray, top_frames: np.ndarray) -> int:
    side_frame_float = sec * FPS
    offsets = side_frames - top_frames
    idx = np.searchsorted(side_frames, side_frame_float)
    lower = max(0, idx - 1)
    upper = min(len(side_frames) - 1, idx)
    if lower == upper:
        offset = offsets[lower]
    else:
        frac = ((side_frame_float - side_frames[lower]) /
                (side_frames[upper] - side_frames[lower]))
        offset = offsets[lower] + frac * (offsets[upper] - offsets[lower])
    return int(round(side_frame_float - offset))


def side_frame_to_top_frame(side_frame: float,
                             side_frames: np.ndarray,
                             top_frames: np.ndarray,
                             log_fn=None) -> int:
    """
    Verbose version: prints an explicit message if the sync conversion fails.
    """
    try:
        offsets = side_frames - top_frames
        idx = np.searchsorted(side_frames, float(side_frame))
        lower = max(0, idx - 1)
        upper = min(len(side_frames) - 1, idx)

        if lower == upper:
            offset = offsets[lower]
        else:
            if side_frames[upper] - side_frames[lower] == 0:
                raise ValueError("side_frames contains duplicate frames; cannot interpolate")
            frac = ((float(side_frame) - side_frames[lower]) /
                    (side_frames[upper] - side_frames[lower]))
            offset = offsets[lower] + frac * (offsets[upper] - offsets[lower])

        result = int(round(float(side_frame) - offset))
        return result

    except Exception as e:
        if log_fn is not None:
            log_fn(f"    [SYNC ERROR] side_frame={side_frame:.0f}  |  "
                   f"side_frames len={len(side_frames)}  |  "
                   f"top_frames len={len(top_frames)}  |  error={e}")
        raise


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_dlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    return df


def load_laser(path: Path, log_fn=None) -> Optional[dict]:
    """Verbose version: reports the status of every -time field."""
    if not path.exists():
        if log_fn:
            log_fn(f"    [SKIP] File not found: {path.name}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        laser = data.get("Laser", {})

        if log_fn:
            log_fn(f"    [DIAG] Checking: {path.parent.name} / {path.name}")
            event_summary = []
            for key in LASER_POINT_KEYS:
                time_key = f"{key}-time"
                if time_key in laser:
                    events = laser[time_key]
                    count = len(events) if isinstance(events, list) else 0
                    event_summary.append(f"{key}: {count} events")
                else:
                    event_summary.append(f"{key}: NO time key")
            log_fn(f"    → {', '.join(event_summary)}")

        has_events = any(
            k.endswith("-time") and isinstance(v, list) and len(v) > 0
            for k, v in laser.items()
        )

        if has_events:
            if log_fn:
                log_fn(f"    [OK] Valid laser events found -> proceeding with analysis")
            return laser
        else:
            if log_fn:
                log_fn(f"    [SKIP seg] File exists but contains no laser events (all -time fields empty)")
            return None

    except Exception as e:
        if log_fn:
            log_fn(f"    [ERROR] Failed to read laser_timepoints.json: {e}")
        return None


def load_sync_frames(top_path: Path, side_path: Path):
    top = pd.read_csv(top_path)["rising_frame"].values.astype(float)
    side = pd.read_csv(side_path)["rising_frame"].values.astype(float)
    return top, side


def load_analysis_param(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  PATH RESOLUTION
# ═══════════════════════════════════════════════════════════════

def resolve_mouse_paths(config: dict, base_dir: Path) -> dict:
    groups = config.get("Group", {})
    result = {}
    for group_name, path_list in groups.items():
        mice = []
        seen = set()
        for rel_path in path_list:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rel_clean = rel_path.replace("\\", os.sep).replace("/", os.sep)
            abs_path = base_dir / rel_clean
            mice.append({
                "name": abs_path.name,
                "rel_path": rel_path,
                "path": abs_path,
                "exists": abs_path.exists(),
            })
        result[group_name] = mice
    return result


def get_segment_folders(mouse_path: Path) -> List[Path]:
    folders = [p for p in mouse_path.iterdir() if p.is_dir() and not p.name.startswith("_")]
    def sort_key(p):
        try:
            return int(p.name.split("_")[0])
        except ValueError:
            return 9999
    return sorted(folders, key=sort_key)


def match_segments_to_continuous(seg_folders: List[Path], continuous: list) -> dict:
    result = {}
    for i, interval in enumerate(continuous):
        if i < len(seg_folders):
            result[i] = {"folder": seg_folders[i], "time_range": (interval[0], interval[1])}
    return result


# ═══════════════════════════════════════════════════════════════
#  ANALYSIS CORE
# ═══════════════════════════════════════════════════════════════

def compute_distance(sn_x: np.ndarray, sn_y: np.ndarray,
                     laser_x: float, laser_y: float,
                     radius: float) -> np.ndarray:
    """
    Distance from snout to laser point, offset by the zone radius so that
    the laser-zone boundary = 0 mm (instead of the laser centre = 0 mm).
    Clips at 0 so values inside the zone stay at 0 rather than going negative.

    Formula:  dist = max(euclidean(snout, laser) - radius, 0) × PX_TO_MM
    """
    dist = np.sqrt((sn_x - laser_x) ** 2 + (sn_y - laser_y) ** 2)
    dist = np.clip(dist - radius, 0.0, None)
    return dist * PX_TO_MM  # convert px → mm


def compute_speed(sn_x: np.ndarray, sn_y: np.ndarray,
                  smooth_window_sec: float = 0.2) -> np.ndarray:
    """
    Velocity calculation matching DLCAnalysis.time_series_velocity:

      roll = 4 (frames)
      x_avg_prev = mean of frames [t+1 .. t+4]   (next-4 rolling mean)
      x_avg_next = mean of frames [t-4 .. t-1]   (prev-4 rolling mean)
      velocity   = sqrt((x_avg_prev - x_avg_next)² + (y_avg_prev - y_avg_next)²)
                   × PX_TO_MM / frame_time / (roll + 1)

    This is a centred difference between the local average of the 4 future
    frames and the local average of the 4 past frames, divided by the
    effective time span (roll+1 frames).  It is more stable than single-frame
    finite differences and avoids the over-smoothing artefacts of a plain
    box-average.  NaNs at the edges are filled by back-fill then forward-fill,
    matching the original .fillna(method='bfill').fillna(method='ffill').

    Note: smooth_window_sec is accepted for API compatibility but is NOT used
    by this implementation (roll=4 is fixed, matching DLCAnalysis).
    """
    roll = 4
    frame_time = 1.0 / FPS   # seconds per frame

    x = pd.Series(sn_x.astype(float))
    y = pd.Series(sn_y.astype(float))

    x_avg_prev = x.shift(-roll).rolling(roll, min_periods=1).mean()
    x_avg_next = x.shift(1).rolling(roll, min_periods=1).mean()
    y_avg_prev = y.shift(-roll).rolling(roll, min_periods=1).mean()
    y_avg_next = y.shift(1).rolling(roll, min_periods=1).mean()

    velocity = (np.sqrt((x_avg_prev - x_avg_next) ** 2 +
                        (y_avg_prev - y_avg_next) ** 2)
                * PX_TO_MM / frame_time / (roll + 1))

    # Edge NaN fill: bfill then ffill (same as original)
    velocity = velocity.bfill().ffill()

    return velocity.to_numpy()


def compute_orientation_angle(hc_x: np.ndarray, hc_y: np.ndarray,
                               snout_x: np.ndarray, snout_y: np.ndarray,
                               laser_x: float, laser_y: float) -> np.ndarray:
    orient_dx = snout_x - hc_x
    orient_dy = snout_y - hc_y
    laser_dx = laser_x - snout_x
    laser_dy = laser_y - snout_y
    dot = orient_dx * laser_dx + orient_dy * laser_dy
    mag1 = np.sqrt(orient_dx**2 + orient_dy**2)
    mag2 = np.sqrt(laser_dx**2 + laser_dy**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.clip(dot / (mag1 * mag2 + 1e-9), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def analyze_laser_event(dlc: pd.DataFrame,
                         start_frame: int, end_frame: int,
                         laser_x: float, laser_y: float,
                         radius: float,
                         orient_thresh_deg: float,
                         pre_frames: int = 0,
                         smooth_window_sec: float = 0.2,
                         auc_t_start: float = 0.0,
                         auc_t_end: float = 10.0) -> dict:
    start_frame = max(0, start_frame)
    end_frame = min(len(dlc) - 1, end_frame)
    if start_frame >= end_frame:
        return None

    hc_x = dlc["head_center"]["x"].values.astype(float)
    hc_y = dlc["head_center"]["y"].values.astype(float)
    sn_x = dlc["snout"]["x"].values.astype(float)
    sn_y = dlc["snout"]["y"].values.astype(float)

    # Likelihood filter: frames where snout likelihood < 0.8 are replaced by
    # linear interpolation between the nearest valid frames, so that position
    # is smoothly filled rather than left as NaN (which would cause huge jumps
    # in speed/distance due to np.diff across missing frames).
    LIKELIHOOD_THRESH = 0.8
    try:
        sn_like = dlc["snout"]["likelihood"].values.astype(float)
    except KeyError:
        raise KeyError(
            f"[likelihood missing] 'snout' does not have a 'likelihood' column.\n"
            f"  Available columns under 'snout': {list(dlc['snout'].columns)}\n"
            f"  Please check the DLC CSV structure."
        )

    def _interp_bad_frames(arr: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
        """Replace flagged frames with linear interpolation of valid neighbours."""
        arr = arr.copy()
        idx = np.arange(len(arr))
        good = ~bad_mask
        if good.sum() < 2:
            return arr  # not enough valid frames to interpolate
        arr[bad_mask] = np.interp(idx[bad_mask], idx[good], arr[good])
        return arr

    sn_bad = sn_like < LIKELIHOOD_THRESH
    sn_x = _interp_bad_frames(sn_x, sn_bad)
    sn_y = _interp_bad_frames(sn_y, sn_bad)
    # head_center has no likelihood column → use as-is
    hc_x = hc_x.copy()
    hc_y = hc_y.copy()

    win_start = max(0, start_frame - pre_frames)
    win_end = end_frame

    hc_x_win = hc_x[win_start:win_end + 1]
    hc_y_win = hc_y[win_start:win_end + 1]
    sn_x_win = sn_x[win_start:win_end + 1]
    sn_y_win = sn_y[win_start:win_end + 1]

    n_frames = len(sn_x_win)
    time_axis = (np.arange(n_frames) - pre_frames) / FPS

    distance = compute_distance(sn_x_win, sn_y_win, laser_x, laser_y, radius)
    speed = compute_speed(sn_x_win, sn_y_win, smooth_window_sec)

    sn_x_on = sn_x[start_frame:win_end + 1]
    sn_y_on = sn_y[start_frame:win_end + 1]
    hc_x_on = hc_x[start_frame:win_end + 1]
    hc_y_on = hc_y[start_frame:win_end + 1]

    dist_on = compute_distance(sn_x_on, sn_y_on, laser_x, laser_y, radius)
    # With the new distance definition (dist = euclidean - radius, clipped at 0),
    # "inside zone" still means dist == 0 — snout is within `radius` px of laser.
    inside_on = dist_on == 0.0

    first_inside = np.where(inside_on)[0]
    latency_sec = first_inside[0] / FPS if len(first_inside) > 0 else np.nan
    dwell_sec = np.sum(inside_on) / FPS

    entry_frames = []
    prev_inside = False
    for i, ins in enumerate(inside_on):
        if ins and not prev_inside:
            entry_frames.append(i / FPS)
        prev_inside = ins
    entry_timepoints = entry_frames

    angle_on = compute_orientation_angle(hc_x_on, hc_y_on, sn_x_on, sn_y_on, laser_x, laser_y)
    orient_agree = angle_on <= orient_thresh_deg
    orient_agree_sec = np.sum(orient_agree) / FPS
    orient_timepoints = np.where(orient_agree)[0] / FPS

    distance_auc_0_10s = compute_auc_0_10s(time_axis, distance, auc_t_start, auc_t_end)
    # Speed AUC is computed over the FULL time window (-5s pre-laser to end),
    # not just the user-configured AUC range.  This captures the complete
    # speed profile including the baseline period before laser onset.
    speed_auc_0_10s = compute_auc_0_10s(time_axis, speed,
                                         float(time_axis[0]), float(time_axis[-1]))

    return {
        "time_axis":        time_axis,
        "distance":         distance,
        "speed":            speed,
        "latency_sec":      latency_sec,
        "dwell_sec":        dwell_sec,
        "entry_timepoints": entry_timepoints,
        "orient_agree_sec": orient_agree_sec,
        "orient_timepoints": orient_timepoints.tolist(),
        "distance_auc_0_10s": distance_auc_0_10s,
        "speed_auc_0_10s":    speed_auc_0_10s,
        "laser_x":          laser_x,
        "laser_y":          laser_y,
        "start_frame":      start_frame,
        "end_frame":        end_frame,
        "hc_x":             hc_x_win,
        "hc_y":             hc_y_win,
        "snout_x":          sn_x_win,
        "snout_y":          sn_y_win,
        "orient_angle":     compute_orientation_angle(hc_x_win, hc_y_win, sn_x_win, sn_y_win, laser_x, laser_y),
        "in_zone":          (distance == 0.0).astype(int),
        "orient_agree_arr": (compute_orientation_angle(hc_x_win, hc_y_win, sn_x_win, sn_y_win, laser_x, laser_y) <= orient_thresh_deg).astype(int),
    }


def compute_auc_0_10s(time_axis: np.ndarray, values: np.ndarray,
                      t_start: float = 0.0, t_end: float = 10.0) -> float:
    """
    Trapezoidal AUC of `values` over the window [t_start, t_end] seconds
    relative to laser ON (time_axis = 0). Frames outside this window
    are ignored; if fewer than 2 points fall in the window, returns NaN.
    t_start and t_end are configurable from the GUI (default 0 to 10 s).
    """
    mask = (time_axis >= t_start) & (time_axis <= t_end)
    t_win = time_axis[mask]
    v_win = values[mask]
    if len(t_win) < 2:
        return np.nan
    order = np.argsort(t_win)
    _trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz_fn(v_win[order], t_win[order]))


# ═══════════════════════════════════════════════════════════════
#  MOUSE-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_mouse(mouse_path: Path, radius: float, orient_thresh: float,
                  smooth_window_sec: float = 0.2,
                  time_groups: Optional[dict] = None,
                  auc_t_start: float = 0.0,
                  auc_t_end: float = 10.0,
                  log_fn=print) -> Optional[dict]:
    if time_groups is None:
        time_groups = DEFAULT_TIME_GROUPS

    param_path = mouse_path / "_analysis_param.json"
    if not param_path.exists():
        log_fn(f"  [SKIP] No _analysis_param.json: {mouse_path.name}")
        return None

    try:
        param = load_analysis_param(param_path)
        continuous = param["Time"]["Continuous"]
    except Exception as e:
        log_fn(f"  [SKIP] Error reading _analysis_param.json: {e}")
        return None

    seg_folders = get_segment_folders(mouse_path)
    if not seg_folders:
        log_fn(f"  [SKIP] No segment folders: {mouse_path.name}")
        return None

    seg_map = match_segments_to_continuous(seg_folders, continuous)

    def assign_time_groups(event_time_min):
        """
        Return every time-group name matching an event's absolute time (min)
        relative to A/C. Intervals are [start, end); end=None means open-ended.
        Overlapping intervals are allowed: one event may match several groups.
        """
        names = []
        for group_name, (tg_start, tg_end) in time_groups.items():
            if tg_end is None:
                if event_time_min >= tg_start:
                    names.append(group_name)
            else:
                if tg_start <= event_time_min < tg_end:
                    names.append(group_name)
        return names

    results_by_group = {name: [] for name in time_groups}

    for seg_idx, info in seg_map.items():
        seg_folder = info["folder"]
        seg_start_min, _seg_end_min = info["time_range"]
        dlc_dir = seg_folder / "_DLC_analysis"

        dlc_path   = dlc_dir / "dlc_data.csv"
        laser_path = dlc_dir / "laser_timepoints.json"
        top_path   = dlc_dir / "light_rising_frames_top.csv"
        side_path  = dlc_dir / "light_rising_frames_side.csv"

        laser = load_laser(laser_path, log_fn)
        if laser is None:
            log_fn(f"  [SKIP seg] No laser data: {seg_folder.name}")
            continue

        if not dlc_path.exists() or not top_path.exists() or not side_path.exists():
            log_fn(f"  [SKIP seg] Missing DLC/sync files: {seg_folder.name}")
            continue

        try:
            dlc = load_dlc(dlc_path)
            top_frames, side_frames = load_sync_frames(top_path, side_path)
        except Exception as e:
            log_fn(f"  [SKIP seg] Load error in {seg_folder.name}: {e}")
            continue

        # === Arena corner coordinates are read per segment ===
        arena_coords = None
        try:
            with open(laser_path, "r", encoding="utf-8") as f:
                laser_tmp = json.load(f).get("Laser", {})
            corners = {c: laser_tmp[c] for c in LASER_POINT_KEYS if c in laser_tmp}
            if len(corners) == 4:
                arena_coords = corners
        except Exception:
            pass

        for key in LASER_POINT_KEYS:
            time_key = f"{key}-time"
            if key not in laser or time_key not in laser:
                continue
            coords = laser[key]
            events = laser[time_key]
            if not events:
                continue

            lx, ly = float(coords[0]), float(coords[1])

            for (t_start_frame, t_end_frame) in events:
                try:
                    sf = side_frame_to_top_frame(t_start_frame, side_frames, top_frames, log_fn=log_fn)
                    ef = side_frame_to_top_frame(t_end_frame, side_frames, top_frames, log_fn=log_fn)
                except Exception:
                    continue

                # Absolute event time (min) relative to A/C = segment start time
                # + the event's frame offset within the segment (converted to
                # minutes), so time groups may be narrower than, or span, segments.
                event_time_min = seg_start_min + (sf / FPS) / 60.0
                matched_groups = assign_time_groups(event_time_min)
                if not matched_groups:
                    continue

                pre_frames = int(5 * FPS)
                res = analyze_laser_event(
                    dlc, sf, ef, lx, ly,
                    radius, orient_thresh,
                    pre_frames=pre_frames,
                    smooth_window_sec=smooth_window_sec,
                    auc_t_start=auc_t_start,
                    auc_t_end=auc_t_end
                )
                if res is not None:
                    res["segment"]        = seg_folder.name
                    res["laser_point"]    = key
                    res["t_start_frame"]  = t_start_frame
                    res["laser_x"]        = lx
                    res["laser_y"]        = ly
                    res["arena"]          = arena_coords   # each event stores its own arena
                    res["event_time_min"] = event_time_min
                    for group_name in matched_groups:
                        results_by_group[group_name].append(res)

    if not any(results_by_group.values()):
        log_fn(f"  [SKIP] No valid laser events: {mouse_path.name}")
        return None

    return {"events": results_by_group}


# ═══════════════════════════════════════════════════════════════
#  AGGREGATION & PLOTTING
# ═══════════════════════════════════════════════════════════════

def aggregate_group(mice_results: List[dict], time_groups: Optional[dict] = None,
                    plot_n_points: int = 300) -> dict:
    if time_groups is None:
        time_groups = DEFAULT_TIME_GROUPS

    agg = {}
    for group_name in time_groups:
        per_mouse_latency = []
        per_mouse_dwell = []
        per_mouse_orient = []
        per_mouse_distance_auc = []
        per_mouse_speed_auc = []
        all_distances = []
        all_speeds = []

        auc_mouse_names = []
        for mouse_res in mice_results:
            events = mouse_res["events"].get(group_name, [])
            if not events:
                continue
            auc_mouse_names.append(mouse_res.get("mouse_name", f"mouse_{len(auc_mouse_names)+1:02d}"))
            latencies = [e["latency_sec"] for e in events if not np.isnan(e["latency_sec"])]
            dwells = [e["dwell_sec"] for e in events]
            orients = [e["orient_agree_sec"] for e in events]
            dist_aucs = [e["distance_auc_0_10s"] for e in events if not np.isnan(e.get("distance_auc_0_10s", np.nan))]
            spd_aucs  = [e["speed_auc_0_10s"]    for e in events if not np.isnan(e.get("speed_auc_0_10s", np.nan))]

            if latencies:
                per_mouse_latency.append(np.mean(latencies))
            if dwells:
                per_mouse_dwell.append(np.mean(dwells))
            if orients:
                per_mouse_orient.append(np.mean(orients))
            if dist_aucs:
                per_mouse_distance_auc.append(np.mean(dist_aucs))
            if spd_aucs:
                per_mouse_speed_auc.append(np.mean(spd_aucs))

            for e in events:
                all_distances.append((e["time_axis"], e["distance"]))
                all_speeds.append((e["time_axis"], e["speed"]))

        def mean_sem(arr):
            if not arr:
                return np.nan, np.nan
            return np.nanmean(arr), (np.nanstd(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0)

        # Per-mouse averaged traces (one mean trace per mouse)
        per_mouse_dist_traces = []
        per_mouse_spd_traces  = []
        trace_mouse_names     = []
        for mouse_res in mice_results:
            evs = mouse_res["events"].get(group_name, [])
            if not evs:
                continue
            trace_mouse_names.append(mouse_res.get("mouse_name", f"mouse_{len(trace_mouse_names)+1:02d}"))
            m_dist = [(e["time_axis"], e["distance"]) for e in evs]
            m_spd  = [(e["time_axis"], e["speed"])    for e in evs]
            t_d, mn_d, _ = interpolate_traces(m_dist, n_points=plot_n_points)
            t_s, mn_s, _ = interpolate_traces(m_spd,  n_points=plot_n_points)
            if t_d is not None:
                per_mouse_dist_traces.append((t_d, mn_d))
            if t_s is not None:
                per_mouse_spd_traces.append((t_s, mn_s))

        # ── Per-Epoch (pooled across all epochs/mice) values ──────────────
        # Used when the "Statistics basis" GUI toggle is set to "Per-Epoch".
        latency_per_epoch = [e["latency_sec"] for m in mice_results
                              for e in m["events"].get(group_name, [])
                              if not np.isnan(e["latency_sec"])]
        dwell_per_epoch   = [e["dwell_sec"] for m in mice_results
                              for e in m["events"].get(group_name, [])]
        orient_per_epoch  = [e["orient_agree_sec"] for m in mice_results
                              for e in m["events"].get(group_name, [])]
        distance_auc_per_epoch = [e["distance_auc_0_10s"] for m in mice_results
                                   for e in m["events"].get(group_name, [])
                                   if not np.isnan(e.get("distance_auc_0_10s", np.nan))]
        speed_auc_per_epoch    = [e["speed_auc_0_10s"] for m in mice_results
                                   for e in m["events"].get(group_name, [])
                                   if not np.isnan(e.get("speed_auc_0_10s", np.nan))]

        lat_mean_epoch,    lat_sem_epoch    = mean_sem(latency_per_epoch)
        dwell_mean_epoch,  dwell_sem_epoch  = mean_sem(dwell_per_epoch)
        orient_mean_epoch, orient_sem_epoch = mean_sem(orient_per_epoch)
        dist_auc_mean_epoch, dist_auc_sem_epoch = mean_sem(distance_auc_per_epoch)
        spd_auc_mean_epoch,  spd_auc_sem_epoch  = mean_sem(speed_auc_per_epoch)

        agg[group_name] = {
            # ── Per-Mouse Average basis ──
            "latency_mean":          mean_sem(per_mouse_latency)[0],
            "latency_sem":           mean_sem(per_mouse_latency)[1],
            "dwell_mean":            mean_sem(per_mouse_dwell)[0],
            "dwell_sem":             mean_sem(per_mouse_dwell)[1],
            "orient_mean":           mean_sem(per_mouse_orient)[0],
            "orient_sem":            mean_sem(per_mouse_orient)[1],
            "distance_auc_mean":     mean_sem(per_mouse_distance_auc)[0],
            "distance_auc_sem":      mean_sem(per_mouse_distance_auc)[1],
            "speed_auc_mean":        mean_sem(per_mouse_speed_auc)[0],
            "speed_auc_sem":         mean_sem(per_mouse_speed_auc)[1],
            "n_mice":                len(per_mouse_latency),
            "distance_mouse_traces": per_mouse_dist_traces,
            "trace_mouse_names":     trace_mouse_names,
            "auc_mouse_names":       auc_mouse_names,
            "speed_mouse_traces":    per_mouse_spd_traces,
            "latency_per_mouse":     list(per_mouse_latency),
            "dwell_per_mouse":       list(per_mouse_dwell),
            "orient_per_mouse":      list(per_mouse_orient),
            "distance_auc_per_mouse": list(per_mouse_distance_auc),
            "speed_auc_per_mouse":    list(per_mouse_speed_auc),

            # ── Per-Epoch basis ──
            "latency_mean_epoch":    lat_mean_epoch,
            "latency_sem_epoch":     lat_sem_epoch,
            "dwell_mean_epoch":      dwell_mean_epoch,
            "dwell_sem_epoch":       dwell_sem_epoch,
            "orient_mean_epoch":     orient_mean_epoch,
            "orient_sem_epoch":      orient_sem_epoch,
            "distance_auc_mean_epoch": dist_auc_mean_epoch,
            "distance_auc_sem_epoch":  dist_auc_sem_epoch,
            "speed_auc_mean_epoch":    spd_auc_mean_epoch,
            "speed_auc_sem_epoch":     spd_auc_sem_epoch,
            "n_epochs":              len(dwell_per_epoch),
            "distance_traces":       all_distances,
            "speed_traces":          all_speeds,
            "latency_per_epoch":     latency_per_epoch,
            "dwell_per_epoch":       dwell_per_epoch,
            "orient_per_epoch":      orient_per_epoch,
            "distance_auc_per_epoch": distance_auc_per_epoch,
            "speed_auc_per_epoch":    speed_auc_per_epoch,
        }
    return agg


COLORS = {
    "Control":          "#212121",
    "SynC":             "#E91E8C",
    "dGAP":             "#4CAF50",
}
DEFAULT_COLOR = "#9E9E9E"

# Single neutral colormap used for ALL trajectory plots, regardless of
# experiment group. It represents elapsed time within the laser-ON window
# (light yellow = just after laser onset, deep red = end of window) and is
# deliberately a different color family from the Control/SynC/dGAP palette
# above, so a trajectory's color is never mistaken for a group identity.
TRAJECTORY_CMAP_NAME = "YlOrRd"
TRAJECTORY_LINE_COLOR_DARK = "#B71C1C"   # end-of-trajectory marker (deep red)
TRAJECTORY_LINE_COLOR_LIGHT = "#FDD835"  # start-of-trajectory marker (light amber)


def interpolate_traces(traces: list, n_points: int = 300) -> tuple:
    """
    Fixed time window: -5 s to +10 s (15 s total).
    """
    if not traces:
        return None, None, None

    valid = [(t_arr, v_arr) for t_arr, v_arr in traces if len(t_arr) >= 2]
    if not valid:
        return None, None, None

    t_min = -5.0
    t_max = 10.0
    common_t = np.linspace(t_min, t_max, n_points)

    interp_vals = []
    for t_arr, v_arr in valid:
        interp = np.interp(common_t, t_arr, v_arr, left=np.nan, right=np.nan)
        interp_vals.append(interp)

    arr = np.array(interp_vals)
    mean = np.nanmean(arr, axis=0)
    sem = (np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros(n_points))
    return common_t, mean, sem


def plot_time_series_epoch(ax, agg_by_exp_group: dict, metric: str,
                           ylabel: str, title: str, group_name: str,
                           plot_n_points: int = 300):
    has_data = False
    for exp_group, agg in agg_by_exp_group.items():
        data   = agg.get(group_name, {})
        traces = data.get(f"{metric}_traces", [])
        color  = COLORS.get(exp_group, DEFAULT_COLOR)

        if not traces:
            continue
        has_data = True

        for t_arr, v_arr in traces:
            ax.plot(t_arr, v_arr, color=color, lw=0.6, alpha=0.25)

        t_common, mean, sem = interpolate_traces(traces, n_points=plot_n_points)
        if t_common is not None:
            ax.plot(t_common, mean, color=color, lw=2.0,
                    label=f"{exp_group} (n={len(traces)})")
            ax.fill_between(t_common, mean - sem, mean + sem,
                            color=color, alpha=0.25)

    ax.axvline(0, color="forestgreen", lw=1.0, ls="--", alpha=0.8, label="Laser ON")
    ax.set_xlabel("Time from laser ON (s)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, PARAMS["line_plot_ymax"])
    if has_data:
        ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1.5)


def plot_time_series_mouse(ax, agg_by_exp_group: dict, metric: str,
                           ylabel: str, title: str, group_name: str,
                           plot_n_points: int = 300):
    """Time series: one thin line per mouse (individual mean), thick group mean±SEM."""
    has_data = False
    mouse_key = f"{metric}_mouse_traces"
    for exp_group, agg in agg_by_exp_group.items():
        data   = agg.get(group_name, {})
        traces = data.get(mouse_key, [])
        color  = COLORS.get(exp_group, DEFAULT_COLOR)
        if not traces:
            continue
        has_data = True
        for t_arr, v_arr in traces:
            ax.plot(t_arr, v_arr, color=color, lw=0.9, alpha=0.4)
        t_common, mean, sem = interpolate_traces(traces, n_points=plot_n_points)
        if t_common is not None:
            ax.plot(t_common, mean, color=color, lw=2.2,
                    label=f"{exp_group} (n={len(traces)})")
            ax.fill_between(t_common, mean - sem, mean + sem,
                            color=color, alpha=0.25)
    ax.axvline(0, color="forestgreen", lw=1.0, ls="--", alpha=0.8, label="Laser ON")
    ax.set_xlabel("Time from laser ON (s)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_ylim(0, PARAMS["line_plot_ymax"])
    if has_data:
        ax.legend(fontsize=10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_box_aspect(1.5)


def plot_auc_summary_grouped(ax, agg_by_exp_group: dict,
                              metric_mean: str, metric_sem: str, metric_per_unit: str,
                              ylabel: str, title: str,
                              time_groups: Optional[dict] = None):
    """
    All time groups on one x-axis, one cluster of bars per time group.
    Within each cluster: one bar per experiment group (colors from COLORS).
    Individual dots (per mouse or per epoch) are overlaid on each bar.
    This allows direct visual comparison of AUC across time groups and
    across experiment groups in a single panel.
    """
    if time_groups is None:
        time_groups = DEFAULT_TIME_GROUPS

    group_names  = list(time_groups.keys())
    exp_groups   = list(agg_by_exp_group.keys())
    n_tg         = len(group_names)
    n_exp        = len(exp_groups)
    bar_width    = 0.75 / max(n_exp, 1)
    x            = np.arange(n_tg)
    has_data     = False

    for i, exp_group in enumerate(exp_groups):
        agg    = agg_by_exp_group[exp_group]
        color  = COLORS.get(exp_group, DEFAULT_COLOR)
        offset = (i - n_exp / 2 + 0.5) * bar_width

        means = [agg.get(g, {}).get(metric_mean, np.nan) for g in group_names]
        sems  = [agg.get(g, {}).get(metric_sem,  np.nan) for g in group_names]

        valid = [(xi, m, s, g) for xi, m, s, g in zip(x, means, sems, group_names)
                 if not np.isnan(m)]
        if not valid:
            continue
        has_data = True

        vx   = np.array([v[0] for v in valid]) + offset
        vm   = np.array([v[1] for v in valid])
        vsem = np.array([0.0 if np.isnan(v[2]) else v[2] for v in valid])

        ax.bar(vx, vm, bar_width * 0.88, color=color, label=exp_group, alpha=0.80)
        ax.errorbar(vx, vm, yerr=vsem,
                    fmt="none", color="black", capsize=3, lw=1.2)

        for xi, _, _, gn in valid:
            unit_vals = agg.get(gn, {}).get(metric_per_unit, [])
            if unit_vals:
                jitter = np.random.uniform(-bar_width * 0.28, bar_width * 0.28,
                                           size=len(unit_vals))
                ax.scatter([xi + offset] * len(unit_vals) + jitter, unit_vals,
                           color=color, s=18, alpha=0.75, zorder=5,
                           edgecolors="black", linewidths=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=11, rotation=15, ha="right")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    if has_data:
        ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)



def generate_pdf(agg_by_exp_group: dict, output_path: Path,
                 time_groups: dict, stats_basis: str = "mouse", log_fn=print,
                 plot_n_points: int = 300,
                 auc_label: str = "0-10s"):
    """
    Generates the single "Overview: All Time Groups" page used in the
    manuscript.  Layout: 2 rows x (N time-group columns + 1 AUC column)

        row 0:  snout->laser distance time series per time group | distance AUC
        row 1:  velocity time series per time group              | velocity AUC

    Note: distance-AUC is the area under the snout->laser distance curve over
    the AUC window after laser ON.  A SMALLER distance-AUC means the mouse
    spent more time close to the laser (stronger approach) - the opposite
    direction from velocity-AUC, where bigger reads as "more".
    """
    log_fn("Generating PDF (Overview page)...")
    basis_label = "Per-Mouse Average" if stats_basis == "mouse" else "Per-Epoch"

    with PdfPages(str(output_path)) as pdf:
        n_groups   = len(time_groups)
        n_cols_ov  = n_groups + 1
        col_widths = [1.0] * n_groups + [1.3]

        fig_ov = plt.figure(figsize=(5.5 * n_groups + 6.0, 16.0))
        fig_ov.suptitle(f"Overview: All Time Groups  [{basis_label}]",
                        fontsize=18, fontweight="bold")
        gs_ov = gridspec.GridSpec(2, n_cols_ov, figure=fig_ov,
                                  hspace=0.50, wspace=0.38,
                                  width_ratios=col_widths)

        for col, group_name in enumerate(time_groups):
            ax_dist = fig_ov.add_subplot(gs_ov[0, col])
            ax_vel  = fig_ov.add_subplot(gs_ov[1, col])

            ts = plot_time_series_mouse if stats_basis == "mouse" else plot_time_series_epoch
            ts(ax_dist, agg_by_exp_group, "distance",
               "Distance (mm)", f"Distance [{group_name}]", group_name,
               plot_n_points=plot_n_points)
            ts(ax_vel, agg_by_exp_group, "speed",
               "Velocity (mm/s)", f"Velocity [{group_name}]", group_name,
               plot_n_points=plot_n_points)

        ax_dist_auc_ov = fig_ov.add_subplot(gs_ov[0, n_groups])
        ax_vel_auc_ov  = fig_ov.add_subplot(gs_ov[1, n_groups])

        sfx = "" if stats_basis == "mouse" else "_epoch"
        unit = "per_mouse" if stats_basis == "mouse" else "per_epoch"
        plot_auc_summary_grouped(ax_dist_auc_ov, agg_by_exp_group,
                                 f"distance_auc_mean{sfx}", f"distance_auc_sem{sfx}",
                                 f"distance_auc_{unit}",
                                 "Distance AUC (mm*s)",
                                 f"Distance AUC {auc_label} - All Groups", time_groups)
        plot_auc_summary_grouped(ax_vel_auc_ov, agg_by_exp_group,
                                 f"speed_auc_mean{sfx}", f"speed_auc_sem{sfx}",
                                 f"speed_auc_{unit}",
                                 "Velocity AUC (mm)",
                                 f"Velocity AUC {auc_label} - All Groups", time_groups)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig_ov, bbox_inches="tight")
        plt.close(fig_ov)

    log_fn(f"PDF saved: {output_path}")


# ===============================================================
#  CSV EXPORT - only the data needed to reproduce the Overview page
# ===============================================================

METRIC_LABEL = {"distance": "distance", "speed": "velocity"}


def export_curves_csv(agg_by_exp_group: dict, output_path: Path,
                      time_groups: dict, stats_basis: str = "mouse",
                      plot_n_points: int = 300, log_fn=print):
    """
    Source data for the line plots (rows 0-1 of the Overview page).

    One row per (experiment_group, time_group, metric, trace_id, time point).
    trace_id = mouse name (thin lines) or "GROUP_MEAN" (thick line; the
    `sem` column is filled only for GROUP_MEAN rows).
    """
    rows = []
    for exp_group, agg in agg_by_exp_group.items():
        for group_name in time_groups:
            data = agg.get(group_name, {})
            names = data.get("trace_mouse_names", [])
            for metric in ("distance", "speed"):
                key = f"{metric}_mouse_traces" if stats_basis == "mouse" else f"{metric}_traces"
                traces = data.get(key, [])
                if not traces:
                    continue
                for i, (t_arr, v_arr) in enumerate(traces):
                    tid = names[i] if (stats_basis == "mouse" and i < len(names)) else f"unit_{i+1:02d}"
                    for t, v in zip(np.asarray(t_arr), np.asarray(v_arr)):
                        rows.append({
                            "experiment_group": exp_group,
                            "time_group": group_name,
                            "metric": METRIC_LABEL[metric],
                            "trace_id": tid,
                            "time_s": round(float(t), 4),
                            "value": None if np.isnan(v) else round(float(v), 4),
                            "sem": None,
                            "n": 1,
                        })
                t_c, mean, sem = interpolate_traces(traces, n_points=plot_n_points)
                if t_c is None:
                    continue
                for t, m, s in zip(t_c, mean, sem):
                    rows.append({
                        "experiment_group": exp_group,
                        "time_group": group_name,
                        "metric": METRIC_LABEL[metric],
                        "trace_id": "GROUP_MEAN",
                        "time_s": round(float(t), 4),
                        "value": None if np.isnan(m) else round(float(m), 4),
                        "sem": None if np.isnan(s) else round(float(s), 4),
                        "n": len(traces),
                    })
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        log_fn(f"Curve data CSV saved: {output_path}")
    else:
        log_fn("No curve data to save.")


def export_auc_csv(agg_by_exp_group: dict, output_path: Path,
                   time_groups: dict, stats_basis: str = "mouse",
                   auc_label: str = "0-10s", log_fn=print):
    """
    Source data for the AUC bar panels (rightmost column of the Overview page).
    One row per (experiment_group, time_group, metric, unit); `group_mean` /
    `group_sem` are the bar height and error bar, repeated on every row.
    """
    sfx  = "" if stats_basis == "mouse" else "_epoch"
    unit = "per_mouse" if stats_basis == "mouse" else "per_epoch"
    rows = []
    for exp_group, agg in agg_by_exp_group.items():
        for group_name in time_groups:
            data = agg.get(group_name, {})
            names = data.get("auc_mouse_names", [])
            for metric in ("distance", "speed"):
                vals = data.get(f"{metric}_auc_{unit}", [])
                mean = data.get(f"{metric}_auc_mean{sfx}", np.nan)
                sem  = data.get(f"{metric}_auc_sem{sfx}",  np.nan)
                for i, v in enumerate(vals):
                    uid = names[i] if (stats_basis == "mouse" and i < len(names)) else f"unit_{i+1:02d}"
                    rows.append({
                        "experiment_group": exp_group,
                        "time_group": group_name,
                        "metric": METRIC_LABEL[metric],
                        "auc_window": auc_label,
                        "unit_id": uid,
                        "auc_value": round(float(v), 4),
                        "group_mean": None if np.isnan(mean) else round(float(mean), 4),
                        "group_sem": None if np.isnan(sem) else round(float(sem), 4),
                        "n": len(vals),
                    })
    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        log_fn(f"AUC data CSV saved: {output_path}")
    else:
        log_fn("No AUC data to save.")


# ===============================================================
#  OPTIONAL: TRAJECTORY PDF   (PARAMS["make_trajectory_pdf"])
#  One panel per laser epoch.  Panels are deliberately minimal:
#  only "Epoch N" as the title, no axes, ticks or numeric annotations.
# ===============================================================

def _draw_arena(ax, arena: dict):
    if arena is None:
        return
    xs = [arena["up-left"][0], arena["up-right"][0], arena["down-right"][0],
          arena["down-left"][0], arena["up-left"][0]]
    ys = [arena["up-left"][1], arena["up-right"][1], arena["down-right"][1],
          arena["down-left"][1], arena["up-left"][1]]
    ax.plot(xs, ys, color="black", lw=1.5, zorder=2)


def _epoch_axis_limits(arena: dict, margin: int = 30):
    if arena is None:
        return (0, 400), (0, 420)
    all_x = [v[0] for v in arena.values()]
    all_y = [v[1] for v in arena.values()]
    return (min(all_x) - margin, max(all_x) + margin), (min(all_y) - margin, max(all_y) + margin)


def _draw_epoch_panel(ax, ev: dict, radius: float, group_name: str, ep_label: str,
                      bodypart: str = "snout", exp_group: Optional[str] = None):
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection

    arena = ev.get("arena")
    xlim, ylim = _epoch_axis_limits(arena)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # No axes, no ticks, no numbers - the arena outline is the only frame.
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Title: epoch number only.
    ax.set_title(ep_label, fontsize=8, pad=3)

    _draw_arena(ax, arena)

    if bodypart == "head_center":
        track_x = ev.get("hc_x", np.array([]))
        track_y = ev.get("hc_y", np.array([]))
    else:
        track_x = ev.get("snout_x", np.array([]))
        track_y = ev.get("snout_y", np.array([]))

    hc_x = ev.get("hc_x", np.array([]))
    hc_y = ev.get("hc_y", np.array([]))
    pre_n = int(5 * FPS)

    if len(track_x) < 2:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", fontsize=7, color="grey")
        return

    pre_end = min(pre_n, len(track_x))

    if pre_end > 1:
        ax.plot(track_x[:pre_end], track_y[:pre_end], color="lightgrey",
                lw=0.9, ls="--", alpha=0.8, zorder=3)

    laser_x_arr = track_x[pre_end:]
    laser_y_arr = track_y[pre_end:]
    n_seg = len(laser_x_arr)
    if n_seg >= 2:
        points = np.array([laser_x_arr, laser_y_arr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(0, max(n_seg - 1, 1))
        lc = LineCollection(segments, cmap=TRAJECTORY_CMAP_NAME, norm=norm,
                            linewidth=1.4, alpha=0.95, zorder=4)
        lc.set_array(np.arange(n_seg - 1))
        ax.add_collection(lc)
        ax.scatter(laser_x_arr[0], laser_y_arr[0], color=TRAJECTORY_LINE_COLOR_LIGHT,
                   s=25, zorder=6, marker="o", edgecolors="black", linewidths=0.6)
        ax.scatter(laser_x_arr[-1], laser_y_arr[-1], color=TRAJECTORY_LINE_COLOR_DARK,
                   s=30, zorder=6, marker="s", edgecolors="black", linewidths=0.6)

    if len(hc_x) > pre_end and len(track_x) > pre_end:
        ax.annotate("", xy=(track_x[pre_end], track_y[pre_end]),
                    xytext=(hc_x[pre_end], hc_y[pre_end]),
                    arrowprops=dict(arrowstyle="->", color="dodgerblue",
                                    lw=1.2, alpha=0.85), zorder=7)

    lx = ev.get("laser_x", np.nan)
    ly = ev.get("laser_y", np.nan)
    if not (np.isnan(lx) or np.isnan(ly)):
        circle = plt.Circle((lx, ly), radius, color="limegreen", alpha=0.25, zorder=7)
        ax.add_patch(circle)
        ax.scatter(lx, ly, color="green", s=55, marker="*", zorder=8)

    handles = [
        Line2D([0], [0], color="lightgrey", lw=1, ls="--", label="pre-laser"),
        Line2D([0], [0], color=TRAJECTORY_LINE_COLOR_DARK, lw=1.4, label="laser ON (time -->)"),
        plt.matplotlib.patches.Patch(facecolor="limegreen", alpha=0.3, label="laser zone"),
    ]
    ax.legend(handles=handles, fontsize=4.5, loc="lower right",
              framealpha=0.6, borderpad=0.4)


def generate_trajectory_pdf(all_mouse_data: list, output_path: Path,
                            radius: float, bodypart: str = "snout",
                            time_groups: Optional[dict] = None, log_fn=print):
    if time_groups is None:
        time_groups = DEFAULT_TIME_GROUPS

    COLS = 4
    PANEL_SIZE = 3.5
    log_fn("Generating trajectory PDF...")

    valid_mice = [m for m in all_mouse_data
                  if any(m["events_by_group"].get(g) for g in time_groups)]
    if not valid_mice:
        log_fn("  [SKIP] No trajectory data to plot.")
        return

    group_names = list(time_groups.keys())

    with PdfPages(str(output_path)) as pdf:
        for mouse in valid_mice:
            m_name = mouse["mouse_name"]
            m_grp = mouse["experiment_group"]

            row_plan = []
            for g in group_names:
                evs = mouse["events_by_group"].get(g, [])
                n_panel_rows = max(1, math.ceil(len(evs) / COLS))
                row_plan.append((g, evs, n_panel_rows))

            gs_rows = []
            for _, _, npr in row_plan:
                gs_rows.append(0.25)
                for _ in range(npr):
                    gs_rows.append(1.0)

            fig_h = 1.2 + sum(npr for _, _, npr in row_plan) * PANEL_SIZE * 1.1
            fig_w = COLS * PANEL_SIZE

            fig = plt.figure(figsize=(fig_w, fig_h))
            fig.suptitle(f"{m_name}  [{m_grp}]", fontsize=11, fontweight="bold")

            gs = gridspec.GridSpec(len(gs_rows), COLS, figure=fig,
                                   height_ratios=gs_rows, hspace=0.55, wspace=0.35)

            gs_row = 0
            for g, evs, n_panel_rows in row_plan:
                n_ep = len(evs)
                ax_hdr = fig.add_subplot(gs[gs_row, :])
                ax_hdr.axis("off")
                ax_hdr.text(0.0, 0.5, f"  {g}  -  {n_ep} epoch{'s' if n_ep != 1 else ''}",
                            transform=ax_hdr.transAxes, fontsize=9, fontweight="bold",
                            va="center", color="white",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="#444",
                                      edgecolor="none"))
                gs_row += 1

                for ep_idx, ev in enumerate(evs):
                    ax = fig.add_subplot(gs[gs_row + ep_idx // COLS, ep_idx % COLS])
                    _draw_epoch_panel(ax, ev, radius, g, f"Epoch {ep_idx + 1}",
                                      bodypart=bodypart, exp_group=m_grp)

                for blank_col in range(n_ep % COLS if n_ep % COLS != 0 else COLS, COLS):
                    fig.add_subplot(gs[gs_row + n_panel_rows - 1, blank_col]).axis("off")

                if n_ep == 0:
                    ax_nd = fig.add_subplot(gs[gs_row, 0])
                    ax_nd.axis("off")
                    ax_nd.text(0.5, 0.5, "no data", transform=ax_nd.transAxes,
                               ha="center", va="center", fontsize=9, color="grey")

                gs_row += n_panel_rows

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log_fn(f"Trajectory PDF saved: {output_path}")


def run_analysis(groups: dict, config: dict,
                 radius: float, orient_thresh: float,
                 output_dir: Path,
                 bodypart: str = "snout",
                 smooth_window_sec: float = 0.2,
                 time_groups: Optional[dict] = None,
                 stats_basis: str = "mouse",
                 plot_n_points: int = 300,
                 auc_t_start: float = 0.0,
                 auc_t_end: float = 10.0,
                 make_trajectory_pdf: Optional[bool] = None,
                 log_fn=print):
    if time_groups is None:
        time_groups = DEFAULT_TIME_GROUPS

    agg_by_exp_group = {}
    all_mouse_events = []
    all_mouse_trajectory_data = []
    mice_names_by_group = {}

    log_fn(f"Time groups: " +
           ", ".join(f"{name} [{s}, {'inf' if e is None else e}) min" for name, (s, e) in time_groups.items()))
    log_fn(f"Statistics basis: {'Per-Mouse Average' if stats_basis == 'mouse' else 'Per-Epoch'}")
    log_fn(f"Plot resolution: {plot_n_points} points ({15/plot_n_points*1000:.0f} ms/point over -5~+10s window)")
    log_fn(f"AUC window: {auc_t_start} s to {auc_t_end} s")

    for exp_group, mice in groups.items():
        log_fn(f"\n{'='*50}")
        log_fn(f"Processing group: {exp_group} ({len(mice)} mice)")
        mice_results = []

        for mouse_info in mice:
            if not mouse_info["exists"]:
                log_fn(f"  [SKIP] Path not found: {mouse_info['name']}")
                continue

            log_fn(f"  Analyzing: {mouse_info['name']}")
            result = analyze_mouse(mouse_info["path"], radius, orient_thresh,
                                       smooth_window_sec, time_groups,
                                       auc_t_start=auc_t_start,
                                       auc_t_end=auc_t_end,
                                       log_fn=log_fn)
            if result is None:
                continue

            result["mouse_name"] = mouse_info["name"]
            mice_results.append(result)

            for group_name, events in result["events"].items():
                for ev in events:
                    epoch_id = f"{mouse_info['name']}__{group_name}__{ev.get('laser_point','')}__fr{int(ev.get('t_start_frame', 0))}"
                    all_mouse_events.append({
                        **ev,
                        "experiment_group": exp_group,
                        "mouse_name": mouse_info["name"],
                        "time_group": group_name,
                        "epoch_id": epoch_id,
                    })

            all_mouse_trajectory_data.append({
                "mouse_name": mouse_info["name"],
                "experiment_group": exp_group,
                "events_by_group": result["events"],
            })

        if mice_results:
            agg_by_exp_group[exp_group] = aggregate_group(mice_results, time_groups,
                                                          plot_n_points=plot_n_points)
            mice_names_by_group[exp_group] = [
                m["name"] for m in mice if m["exists"]
                and any(ev for r in mice_results
                        for g_evs in r["events"].values()
                        for ev in g_evs)
            ][:len(mice_results)]
            log_fn(f"  → {len(mice_results)} mice with valid data")
        else:
            log_fn(f"  → No valid data for {exp_group}")

    if not agg_by_exp_group:
        log_fn("\n[ERROR] No data to analyze across all groups.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    basis_suffix = "mouse" if stats_basis == "mouse" else "epoch"

    auc_label = f"{auc_t_start:.1f}-{auc_t_end:.1f}s"
    generate_pdf(agg_by_exp_group,
                 output_dir / f"laser_response_overview_{basis_suffix}.pdf",
                 time_groups, stats_basis, log_fn, plot_n_points=plot_n_points,
                 auc_label=auc_label)
    export_curves_csv(agg_by_exp_group,
                      output_dir / f"laser_response_curves_{basis_suffix}.csv",
                      time_groups, stats_basis, plot_n_points, log_fn)
    export_auc_csv(agg_by_exp_group,
                   output_dir / f"laser_response_auc_{basis_suffix}.csv",
                   time_groups, stats_basis, auc_label, log_fn)

    if PARAMS["make_trajectory_pdf"] if make_trajectory_pdf is None else make_trajectory_pdf:
        generate_trajectory_pdf(
            all_mouse_trajectory_data,
            output_dir / "laser_response_trajectories.pdf",
            radius, bodypart, time_groups, log_fn
        )

    log_fn(f"\n✓ Analysis complete. Outputs saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════

class TimeGroupEditor(tk.LabelFrame):
    """
    Editable list of (name, start_min, end_min) time-group rows.
    `end_min` may be left blank, meaning "open-ended" (everything from
    start_min onward is included). Rows can be added/removed freely, and
    different rows may span different widths (e.g. 10 min and 30 min).
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent,
                          text="4. Time Groups (minutes; leave 'End' blank for open-ended)",
                          padx=8, pady=6, **kwargs)

        header = tk.Frame(self)
        header.pack(fill="x")
        tk.Label(header, text="Name", width=24, anchor="w").grid(row=0, column=0, padx=2)
        tk.Label(header, text="Start (min)", width=10, anchor="w").grid(row=0, column=1, padx=2)
        tk.Label(header, text="End (min)", width=10, anchor="w").grid(row=0, column=2, padx=2)

        self.rows_frame = tk.Frame(self)
        self.rows_frame.pack(fill="x")

        self.rows = []

        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", pady=(4, 0))
        tk.Button(btn_frame, text="+ Add Row", command=self.add_row,
                  bg="#4A90D9", fg="white").pack(side="left")

        for name, (s, e) in DEFAULT_TIME_GROUPS.items():
            self.add_row(name, s, e)

    def add_row(self, name="", start="", end=""):
        row_frame = tk.Frame(self.rows_frame)
        row_frame.pack(fill="x", pady=1)

        name_var = tk.StringVar(value=name)
        start_var = tk.StringVar(value="" if start in (None, "") else str(start))
        end_var = tk.StringVar(value="" if end is None else str(end))

        tk.Entry(row_frame, textvariable=name_var, width=24).grid(row=0, column=0, padx=2)
        tk.Entry(row_frame, textvariable=start_var, width=10).grid(row=0, column=1, padx=2)
        tk.Entry(row_frame, textvariable=end_var, width=10).grid(row=0, column=2, padx=2)
        remove_btn = tk.Button(row_frame, text="✕", width=2, fg="#C0392B",
                                command=lambda: self.remove_row(row_frame))
        remove_btn.grid(row=0, column=3, padx=4)

        self.rows.append({"frame": row_frame, "name": name_var,
                           "start": start_var, "end": end_var})

    def remove_row(self, frame):
        self.rows = [r for r in self.rows if r["frame"] is not frame]
        frame.destroy()

    def get_time_groups(self) -> dict:
        """Return an ordered dict {name: (start_min, end_min_or_None)}.
        Raises ValueError with a human-readable message on invalid input."""
        result = {}
        for r in self.rows:
            name = r["name"].get().strip()
            start_str = r["start"].get().strip()
            end_str = r["end"].get().strip()
            if not name:
                continue
            if name in result:
                raise ValueError(f"Duplicate time group name: '{name}'")
            try:
                start = float(start_str)
            except ValueError:
                raise ValueError(f"Group '{name}': invalid start value '{start_str}'")
            if end_str == "":
                end = None
            else:
                try:
                    end = float(end_str)
                except ValueError:
                    raise ValueError(f"Group '{name}': invalid end value '{end_str}'")
                if end <= start:
                    raise ValueError(
                        f"Group '{name}': end ({end}) must be greater than start ({start})")
            result[name] = (start, end)
        if not result:
            raise ValueError("Please define at least one time group.")
        return result


class AnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laser Response Group Analysis (v2)")
        self.geometry("1020x860")
        self.resizable(True, True)

        self.main_folder: Optional[Path] = None
        self.config: dict = {}
        self.groups: dict = {}

        self._build_ui()
        self._autoload_default_config()

    def _autoload_default_config(self):
        """Pre-load <repo>/data/edf8_demo/_Group_analysis/*.json if present."""
        cfg = find_default_config()
        if cfg is None:
            self._log(f"No JSON found in {CONFIG_DIR} - use Browse... to pick one.")
            return
        self._load_config(cfg)

    def _build_ui(self):
        top_frame = tk.LabelFrame(self, text="1. Select Group JSON File", padx=8, pady=6)
        top_frame.pack(fill="x", padx=10, pady=(8, 4))

        self.path_var = tk.StringVar(value="(not selected)")
        tk.Label(top_frame, textvariable=self.path_var, anchor="w", relief="sunken", width=70, bg="white").pack(side="left", padx=(0, 6))
        tk.Button(top_frame, text="Browse...", command=self._select_json_file, bg="#4A90D9", fg="white", padx=8).pack(side="left")

        param_frame = tk.LabelFrame(self, text="2. Analysis Parameters", padx=10, pady=8)
        param_frame.pack(fill="x", padx=10, pady=4)

        tk.Label(param_frame, text="Laser zone radius (px):").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.radius_var = tk.DoubleVar(value=PARAMS["laser_zone_radius_px"])
        tk.Spinbox(param_frame, from_=5, to=200, increment=5, textvariable=self.radius_var, width=8).grid(row=0, column=1, sticky="w")

        tk.Label(param_frame, text="Orientation threshold (°):").grid(row=0, column=2, sticky="w", padx=(20, 6))
        self.orient_var = tk.DoubleVar(value=PARAMS["orientation_thresh_deg"])
        tk.Spinbox(param_frame, from_=1, to=180, increment=1, textvariable=self.orient_var, width=8).grid(row=0, column=3, sticky="w")

        tk.Label(param_frame, text="Velocity smooth window (s):").grid(row=1, column=0, sticky="w", pady=(6, 0), padx=(0, 6))
        self.smooth_var = tk.DoubleVar(value=PARAMS["speed_smooth_window_s"])
        tk.Spinbox(param_frame, from_=0.05, to=2.0, increment=0.05, format="%.2f",
                   textvariable=self.smooth_var, width=8).grid(row=1, column=1, sticky="w", pady=(6, 0))
        tk.Label(param_frame, text="(unused: velocity uses fixed roll=4 frames, matching DLCAnalysis)", fg="#aaa").grid(row=1, column=2, sticky="w", pady=(6, 0), padx=(4, 0))

        # Trajectory body part selector
        tk.Label(param_frame, text="Trajectory path uses:").grid(row=2, column=0, sticky="w", pady=(8, 0), padx=(0, 6))
        self.bodypart_var = tk.StringVar(value=PARAMS["trajectory_bodypart"])
        ttk.Combobox(param_frame, textvariable=self.bodypart_var, values=["snout", "head_center"], state="readonly", width=15).grid(row=2, column=1, sticky="w", pady=(8, 0))

        # Statistics basis selector
        tk.Label(param_frame, text="Statistics basis:").grid(row=2, column=2, sticky="w", pady=(8, 0), padx=(20, 6))
        self.stats_basis_var = tk.StringVar(
            value="Per-Mouse Average" if PARAMS["stats_basis"] == "mouse" else "Per-Epoch")
        ttk.Combobox(param_frame, textvariable=self.stats_basis_var,
                     values=["Per-Mouse Average", "Per-Epoch"], state="readonly", width=18).grid(row=2, column=3, sticky="w", pady=(8, 0))

        tk.Label(param_frame, text="Output folder:").grid(row=3, column=0, sticky="w", pady=(8, 0), padx=(0, 6))
        self.output_var = tk.StringVar(value="(auto: same folder as group JSON)")
        tk.Label(param_frame, textvariable=self.output_var, anchor="w", relief="sunken", width=55, bg="#f0f0f0", fg="#444").grid(row=3, column=1, columnspan=3, sticky="w", pady=(8, 0))

        # ── Plot resolution controls (row 4) ──────────────────────────────
        # Two linked entry fields: changing either one recalculates the other.
        #   n_points  = total interpolated points across -5~+10s window
        #   bin_size  = seconds per point = 15 / n_points
        tk.Label(param_frame, text="Plot resolution (n points):").grid(
            row=4, column=0, sticky="w", pady=(8, 0), padx=(0, 6))
        self.n_points_var = tk.IntVar(value=PARAMS["plot_n_points"])
        n_points_spin = tk.Spinbox(
            param_frame, from_=15, to=1500, increment=15,
            textvariable=self.n_points_var, width=8,
            command=self._on_n_points_changed)
        n_points_spin.grid(row=4, column=1, sticky="w", pady=(8, 0))
        n_points_spin.bind("<FocusOut>", lambda e: self._on_n_points_changed())
        n_points_spin.bind("<Return>",   lambda e: self._on_n_points_changed())

        tk.Label(param_frame, text="Time bin size (s/point):").grid(
            row=4, column=2, sticky="w", pady=(8, 0), padx=(20, 6))
        self.bin_size_var = tk.StringVar(value=f"{15.0 / PARAMS['plot_n_points']:.4f}")
        bin_entry = tk.Entry(param_frame, textvariable=self.bin_size_var, width=8)
        bin_entry.grid(row=4, column=3, sticky="w", pady=(8, 0))
        bin_entry.bind("<FocusOut>", lambda e: self._on_bin_size_changed())
        bin_entry.bind("<Return>",   lambda e: self._on_bin_size_changed())

        tk.Label(param_frame,
                 text="  300 pts = 0.050 s/pt (smooth)   |   150 pts = 0.100 s/pt   |   15 pts = 1.0 s/pt (coarse)",
                 fg="#888", font=("Arial", 8)
                 ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # ── AUC time window controls (row 6) ──────────────────────────────
        tk.Label(param_frame, text="AUC window start (s):").grid(
            row=6, column=0, sticky="w", pady=(8, 0), padx=(0, 6))
        self.auc_start_var = tk.DoubleVar(value=PARAMS["auc_t_start_s"])
        tk.Spinbox(param_frame, from_=-5.0, to=30.0, increment=0.5, format="%.1f",
                   textvariable=self.auc_start_var, width=8).grid(
            row=6, column=1, sticky="w", pady=(8, 0))

        tk.Label(param_frame, text="AUC window end (s):").grid(
            row=6, column=2, sticky="w", pady=(8, 0), padx=(20, 6))
        self.auc_end_var = tk.DoubleVar(value=PARAMS["auc_t_end_s"])
        tk.Spinbox(param_frame, from_=0.5, to=30.0, increment=0.5, format="%.1f",
                   textvariable=self.auc_end_var, width=8).grid(
            row=6, column=3, sticky="w", pady=(8, 0))

        self.traj_var = tk.BooleanVar(value=PARAMS["make_trajectory_pdf"])
        tk.Checkbutton(param_frame,
                       text="Also generate trajectory PDF (not used in the manuscript)",
                       variable=self.traj_var).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

        tk.Label(param_frame,
                 text="  AUC computed over [start, end] s relative to laser ON  (default 0 to 10 s)",
                 fg="#888", font=("Arial", 8)
                 ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(2, 0))

        # Editable time groups
        self.time_group_editor = TimeGroupEditor(self)
        self.time_group_editor.pack(fill="x", padx=10, pady=4)

        list_frame = tk.LabelFrame(self, text="5. Mouse List (from JSON)", padx=8, pady=6)
        list_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self.tree = ttk.Treeview(list_frame, columns=("group", "status"), show="headings")
        self.tree.heading("group", text="Experiment Group")
        self.tree.heading("status", text="Status")
        self.tree.column("group", width=280)
        self.tree.column("status", width=80, anchor="center")
        self.tree.pack(fill="both", expand=True, side="left")

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        self.tree.tag_configure("ok", foreground="#1A7A3A")
        self.tree.tag_configure("missing", foreground="#C0392B")

        log_frame = tk.LabelFrame(self, text="Log", padx=6, pady=4)
        log_frame.pack(fill="x", padx=10, pady=(4, 2))

        self.log_text = tk.Text(log_frame, height=7, state="disabled", font=("Courier", 9), bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill="x")

        btn_frame = tk.Frame(self, pady=6)
        btn_frame.pack(fill="x", padx=10)

        self.run_btn = tk.Button(btn_frame, text="▶  Run Analysis", command=self._run,
                                 bg="#27AE60", fg="white", font=("Arial", 11, "bold"), padx=16, pady=4)
        self.run_btn.pack(side="right")

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(btn_frame, textvariable=self.status_var, fg="#555", anchor="w").pack(side="left")

    def _on_n_points_changed(self):
        """User edited n_points → recalculate and update bin_size display."""
        try:
            n = max(1, int(self.n_points_var.get()))
            self.n_points_var.set(n)
            self.bin_size_var.set(f"{15.0 / n:.4f}")
        except (ValueError, tk.TclError):
            pass

    def _on_bin_size_changed(self):
        """User edited bin_size → recalculate and update n_points."""
        try:
            b = float(self.bin_size_var.get())
            if b <= 0:
                return
            n = max(1, round(15.0 / b))
            self.n_points_var.set(n)
            self.bin_size_var.set(f"{15.0 / n:.4f}")
        except (ValueError, tk.TclError):
            pass

    def _log(self, msg: str):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def _select_json_file(self):
        json_path_str = filedialog.askopenfilename(
            title="Select group definition JSON file",
            initialdir=str(CONFIG_DIR),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not json_path_str:
            return

        self._load_config(Path(json_path_str))

    def _load_config(self, json_path: Path):
        self.main_folder = json_path.parent
        self.path_var.set(str(json_path))

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", str(e))
            return

        base_dir = self.main_folder.parent
        self.groups = resolve_mouse_paths(self.config, base_dir)

        self._output_dir = self.main_folder
        self.output_var.set(str(self._output_dir))

        self._populate_tree()
        total = sum(len(v) for v in self.groups.values())
        exist = sum(sum(1 for m in v if m["exists"]) for v in self.groups.values())
        self.status_var.set(f"Loaded: {len(self.groups)} groups, {total} mice ({exist} found)")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for group, mice in self.groups.items():
            for m in mice:
                tag = "ok" if m["exists"] else "missing"
                status = "✓" if m["exists"] else "✗ missing"
                self.tree.insert("", "end", values=(f"[{group}]  {m['name']}", status), tags=(tag,))

    def _run(self):
        if not self.groups:
            messagebox.showwarning("Warning", "Please select a main folder first.")
            return

        output_dir = getattr(self, "_output_dir", None)
        if output_dir is None:
            messagebox.showwarning("Warning", "Please select a main folder first.")
            return

        try:
            time_groups = self.time_group_editor.get_time_groups()
        except ValueError as e:
            messagebox.showerror("Time Group Error", str(e))
            return

        stats_basis = "mouse" if self.stats_basis_var.get().startswith("Per-Mouse") else "epoch"

        self.run_btn.config(state="disabled", text="Running...")
        self.status_var.set("Running analysis...")

        radius = self.radius_var.get()
        orient_thresh = self.orient_var.get()
        bodypart = self.bodypart_var.get()
        smooth_window_sec = self.smooth_var.get()
        plot_n_points = max(1, int(self.n_points_var.get()))
        make_traj   = bool(self.traj_var.get())
        auc_t_start = float(self.auc_start_var.get())
        auc_t_end   = float(self.auc_end_var.get())
        if auc_t_end <= auc_t_start:
            messagebox.showerror("AUC Window Error",
                f"AUC end ({auc_t_end}s) must be greater than start ({auc_t_start}s).")
            self.run_btn.config(state="normal", text="▶  Run Analysis")
            self.status_var.set("Ready.")
            return

        def worker():
            try:
                run_analysis(
                    self.groups, self.config,
                    radius, orient_thresh,
                    output_dir,
                    bodypart,
                    smooth_window_sec,
                    time_groups,
                    stats_basis,
                    plot_n_points=plot_n_points,
                    auc_t_start=auc_t_start,
                    auc_t_end=auc_t_end,
                    make_trajectory_pdf=make_traj,
                    log_fn=self._log
                )
                self.status_var.set("✓ Analysis complete!")
                messagebox.showinfo("Done", f"Analysis complete!\nOutputs saved to:\n{output_dir}")
            except Exception as e:
                import traceback
                self._log(f"\n[ERROR] {e}")
                self._log(traceback.format_exc())
                self.status_var.set("✗ Error during analysis.")
                messagebox.showerror("Error", str(e))
            finally:
                self.run_btn.config(state="normal", text="▶  Run Analysis")

        threading.Thread(target=worker, daemon=True).start()


def run_headless(config_path: Optional[Path] = None, log_fn=print):
    """Run with the manuscript parameters, no GUI."""
    cfg_path = Path(config_path) if config_path else find_default_config()
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Group JSON not found (looked in {CONFIG_DIR})")

    log_fn(f"Config : {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = cfg_path.parent.parent
    groups   = resolve_mouse_paths(config, base_dir)
    out_dir  = cfg_path.parent

    run_analysis(
        groups, config,
        radius=PARAMS["laser_zone_radius_px"],
        orient_thresh=PARAMS["orientation_thresh_deg"],
        output_dir=out_dir,
        bodypart=PARAMS["trajectory_bodypart"],
        smooth_window_sec=PARAMS["speed_smooth_window_s"],
        time_groups=TIME_GROUPS,
        stats_basis=PARAMS["stats_basis"],
        plot_n_points=PARAMS["plot_n_points"],
        auc_t_start=PARAMS["auc_t_start_s"],
        auc_t_end=PARAMS["auc_t_end_s"],
        make_trajectory_pdf=PARAMS["make_trajectory_pdf"],
        log_fn=log_fn,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command line:  python 03_laser_chasing_analysis_v2.py [path/to/group.json]
        arg = sys.argv[1]
        if arg == "--gui":
            AnalysisApp().mainloop()
        else:
            run_headless(None if arg in ("--run", "-r") else Path(arg))
    elif PARAMS["use_gui"]:
        AnalysisApp().mainloop()
    else:
        run_headless()