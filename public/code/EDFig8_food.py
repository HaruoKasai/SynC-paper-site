"""
Group Analysis - Food Approaching Response  (v2)
=================================================
Time axis convention:
  t = 0   -> start_frame  (food placed)
  t = +N  -> N seconds after food placed

Distance calculation:
  For each frame, load head_center xy from dlc_data.csv and
  food xy from food_location_frames_ep{N}.csv (per-frame food position).
  Distance = sqrt((hc_x - food_x)^2 + (hc_y - food_y)^2)
  If px_to_mm is set: convert to mm; otherwise keep px.

Epochs without food_location_frames_ep{N}.csv are skipped.

Outputs (saved next to the group JSON):
  food_approaching_analysis_<part>_<scale>.pdf  - the analysis figure
  food_curves_<part>_<scale>.csv               - source data for the curves
  food_auc_<part>_<scale>.csv                  - source data for the AUC bars
  food_approaching_trajectories_<part>_<scale>.pdf  (optional, off by default)
"""

import os, json, math, warnings
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

POST_SEC_DEFAULT = 60.0
PRE_SEC_DEFAULT  = 10.0  # seconds before food placement to show
LOG_T_MIN        = 0.1   # kept for compatibility but sqrt scale used instead
N_INTERP         = 2000  # default plot resolution (GUI-adjustable)

# ===============================================================
#  ANALYSIS PARAMETERS  --  values used in the manuscript
#  These are also the defaults shown in the GUI.  Edit here (not in
#  the GUI code) if the analysis settings ever need to change.
# ===============================================================
PARAMS = {
    # -- 2. Parameters -----------------------------------------------
    "px_to_mm":              0.76,          # mm per px; None = keep px
    "speed_smooth_s":        0.20,
    "post_window_s":         90.0,
    "pre_window_s":          30.0,
    "distance_bodypart":     "head_center",
    "velocity_bodypart":     "body_center",
    "plot_points":           24,            # interpolated points per curve
    "auc_start_s":           30.0,
    "auc_end_s":             90.0,
    # -- plotting ----------------------------------------------------
    "line_plot_ymax":        200.0,         # y-axis upper limit for ALL line plots
    # -- how to launch -----------------------------------------------
    # False = run straight away with the parameters above (no window).
    # True  = open the GUI so the parameters can be adjusted by hand.
    "use_gui":               False,
    # -- optional output ---------------------------------------------
    # Trajectory PDF (one panel per food epoch, per mouse).  Not used in
    # the manuscript, so it is OFF by default.  Set to True to generate it.
    "make_trajectory_pdf":   False,
}

# -- Time groups (minutes relative to A/C; None = open-ended) --------
# NOTE: the boundaries below reproduce the original script exactly.
TIME_GROUPS_MIN = {
    "Before":     (None, 0),
    "After 0-50": (0,    50),
    "After 60+":  (60,   None),
}


def _make_time_group_fn(lo, hi):
    def _fn(s):
        if lo is not None and s < lo:
            return False
        if hi is not None and s >= hi:
            return False
        return True
    return _fn


TIME_GROUPS = {name: _make_time_group_fn(lo, hi)
               for name, (lo, hi) in TIME_GROUPS_MIN.items()}

# -- Input / output locations ---------------------------------------
# Layout assumed by this script:
#     <repo>/code/02_food_chasing_analysis_v2.py   <- this file
#     <repo>/data/edf8_demo/_Group_analysis/*.json <- config + outputs
CODE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = CODE_DIR.parent / "data" / "edf8_demo"
CONFIG_DIR = DATA_DIR / "_Group_analysis"
OUTPUT_DIR = CONFIG_DIR


def find_default_config() -> Optional[Path]:
    """Locate the food group-definition JSON inside CONFIG_DIR."""
    if not CONFIG_DIR.is_dir():
        return None
    cands = sorted(CONFIG_DIR.glob("*food*.json"))
    if not cands:
        cands = sorted(CONFIG_DIR.glob("*.json"))
    return cands[0] if cands else None

COLORS = {
    "Control": "#222222",
    "SynC":    "#E91E8C",
    "dGAP":    "#4CAF50",
}
DEFAULT_COLOR = "#9E9E9E"

METRIC_LABEL = {"distance": "distance", "speed": "velocity"}

TRAJ_COLS  = 4
PANEL_SIZE = 3.5

BODY_PARTS = ["head_center", "snout", "centroid", "body_center"]
# likelihood threshold for parts that have it
LIK_THRESH = {
    "snout":    0.8,
    "centroid": 0.8,
    # head_center and body_center: use 0.6 as before (original behaviour)
    "head_center": 0.6,
    "body_center": 0.6,
}


# ═══════════════════════════════════════════════════════════════
#  PATH HELPERS
# ═══════════════════════════════════════════════════════════════

def get_segment_folders(mouse_path: Path) -> List[Path]:
    folders = [p for p in mouse_path.iterdir()
               if p.is_dir() and not p.name.startswith("_")]
    def sort_key(p):
        try:
            return int(p.name.split("_")[0])
        except ValueError:
            return 9999
    return sorted(folders, key=sort_key)


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_mouse_paths(config: dict, base_dir: Path) -> dict:
    result = {}
    for group_name, path_list in config.get("Group", {}).items():
        mice, seen = [], set()
        for rel_path in path_list:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            rel_clean = rel_path.replace("\\", os.sep).replace("/", os.sep)
            abs_path  = base_dir / rel_clean
            mice.append({"name": abs_path.name, "path": abs_path,
                          "exists": abs_path.exists()})
        result[group_name] = mice
    return result


def csv_path_for_epoch(laser_path: Path, epoch_idx: int) -> Path:
    return laser_path.parent / f"food_location_frames_ep{epoch_idx}.csv"


# ═══════════════════════════════════════════════════════════════
#  DLC DATA
# ═══════════════════════════════════════════════════════════════

def load_dlc(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, header=[0, 1], index_col=0)
    except Exception:
        return None


def get_bodypart_xy(dlc: pd.DataFrame,
                    bodypart: str = "head_center") -> Optional[tuple]:
    """
    Extract xy for the given body part with likelihood-based interpolation.
    For snout/centroid: frames with likelihood < 0.8 are interpolated out.
    For head_center/body_center: threshold is 0.6 (original behaviour).
    """
    try:
        bp_x = dlc[bodypart]["x"].values.astype(float)
        bp_y = dlc[bodypart]["y"].values.astype(float)
    except KeyError:
        return None
    try:
        lik   = dlc[bodypart]["likelihood"].values.astype(float)
        thresh = LIK_THRESH.get(bodypart, 0.6)
        bad   = lik < thresh
        idx   = np.arange(len(bp_x))
        good  = ~bad
        if good.sum() >= 2:
            bp_x[bad] = np.interp(idx[bad], idx[good], bp_x[good])
            bp_y[bad] = np.interp(idx[bad], idx[good], bp_y[good])
        elif good.sum() == 0:
            return None   # all frames low likelihood  -  skip entirely
    except KeyError:
        pass   # no likelihood column  -  use as-is
    return bp_x, bp_y


# keep alias for backward compatibility
def get_head_center_xy(dlc):
    return get_bodypart_xy(dlc, "head_center")


def get_fps(dlc: pd.DataFrame) -> float:
    # try any body part that has a time column
    for bp in BODY_PARTS + list(dlc.columns.get_level_values(0).unique()):
        try:
            times = dlc[bp]["time"].values.astype(float)
            dt = np.nanmedian(np.diff(times))
            if dt > 0:
                return 1.0 / dt
        except Exception:
            continue
    return 20.0


def get_arena(laser_data: dict) -> Optional[dict]:
    laser   = laser_data.get("Laser", {})
    corners = {}
    for key in ["up-left", "up-right", "down-left", "down-right"]:
        if key in laser and isinstance(laser[key], (list, tuple)) \
                and len(laser[key]) == 2:
            corners[key] = laser[key]
    return corners if len(corners) == 4 else None


# ═══════════════════════════════════════════════════════════════
#  EPOCH COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _dlc_velocity(x: np.ndarray, y: np.ndarray,
                  fps: float, scale: float,
                  roll: int = 4) -> np.ndarray:
    """
    DLCAnalysis.time_series_velocity method (roll=4):
      x_avg_prev = mean of next-4 frames (shift(-roll).rolling(roll))
      x_avg_next = mean of prev-4 frames (shift(1).rolling(roll))
      velocity   = sqrt(dx^2 + dy^2) x scale / frame_time / (roll+1)
    Edge NaNs filled with bfill then ffill.
    """
    frame_time = 1.0 / fps
    xs = pd.Series(x.astype(float))
    ys = pd.Series(y.astype(float))
    xp = xs.shift(-roll).rolling(roll, min_periods=1).mean()
    xn = xs.shift(1).rolling(roll, min_periods=1).mean()
    yp = ys.shift(-roll).rolling(roll, min_periods=1).mean()
    yn = ys.shift(1).rolling(roll, min_periods=1).mean()
    vel = (np.sqrt((xp - xn)**2 + (yp - yn)**2)
           * scale / frame_time / (roll + 1))
    vel = vel.bfill().ffill()
    return vel.to_numpy()


def compute_epoch(hc_x_full: np.ndarray,
                  hc_y_full: np.ndarray,
                  food_df: pd.DataFrame,
                  start_frame: int,
                  end_frame: int,
                  px_to_mm: Optional[float],
                  fps: float,
                  smooth_sec: float,        # kept for API compat, unused
                  cross_seg_offset: int = 0,
                  pre_sec: float = 0.0,
                  spd_x_full: Optional[np.ndarray] = None,
                  spd_y_full: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    hc_x_full / hc_y_full : full head-center arrays for the segment(s).
                             For normal epochs these span the whole segment;
                             for cross-seg they are pre-concatenated.
    food_df  : DataFrame [frame, x, y]  -  0-indexed from start_frame.
    pre_sec  : seconds before start_frame to include (t<0 region).
    """
    scale = px_to_mm if px_to_mm is not None else 1.0
    unit  = "mm" if px_to_mm is not None else "px"

    # pre-window: frames before start_frame
    pre_frames = int(round(pre_sec * fps))
    pre_start  = max(0, start_frame - pre_frames)
    actual_pre = start_frame - pre_start   # may be less than requested

    # slice full arrays: pre region + post region
    n_post = end_frame - start_frame + 1
    hc_x   = hc_x_full[pre_start : start_frame + n_post]
    hc_y   = hc_y_full[pre_start : start_frame + n_post]
    n      = len(hc_x)
    if n == 0:
        return None

    # time axis: negative before food placement, 0 at start_frame
    time_axis = (np.arange(n) - actual_pre) / fps

    # food coordinates:
    #   pre-window uses food position at t=0 (first row of food_df)
    food_x_post = food_df["x"].values.astype(float)
    food_y_post = food_df["y"].values.astype(float)
    food_x0 = food_x_post[0] if len(food_x_post) > 0 else 0.0
    food_y0 = food_y_post[0] if len(food_y_post) > 0 else 0.0

    n_use_post = min(n_post, len(food_x_post))
    fx = np.concatenate([
        np.full(actual_pre, food_x0),       # pre: fixed at t=0 food pos
        food_x_post[:n_use_post],            # post: per-frame food pos
        np.full(max(0, n - actual_pre - n_use_post), food_x_post[-1])
    ])[:n]
    fy = np.concatenate([
        np.full(actual_pre, food_y0),
        food_y_post[:n_use_post],
        np.full(max(0, n - actual_pre - n_use_post), food_y_post[-1])
    ])[:n]

    dist_px  = np.sqrt((hc_x - fx)**2 + (hc_y - fy)**2)
    distance = dist_px * scale

    # speed: DLCAnalysis.time_series_velocity method
    # Compute on the FULL segment array first (avoids edge artefacts
    # at the pre/post boundary), then slice to the epoch window.
    # Use separate speed body-part arrays if provided.
    sx_full = spd_x_full if spd_x_full is not None else hc_x_full
    sy_full = spd_y_full if spd_y_full is not None else hc_y_full
    spd_full = _dlc_velocity(sx_full, sy_full, fps, scale)
    spd = spd_full[pre_start : start_frame + n_post][:n]
    if len(spd) < n:
        spd = np.pad(spd, (0, n - len(spd)), mode="edge")

    return {
        "time_axis":        time_axis,
        "distance":         distance,
        "speed":            spd,
        "duration":         time_axis[-1],        # last post-window time
        "pre_frames":       actual_pre,
        "unit":             unit,
        # trajectory: post-window (for AUC/curve panels) and pre-window separately
        "hc_x":             hc_x[actual_pre:],
        "hc_y":             hc_y[actual_pre:],
        "food_x":           fx[actual_pre:],
        "food_y":           fy[actual_pre:],
        "hc_x_pre":         hc_x[:actual_pre+1],   # include t=0 point to connect
        "hc_y_pre":         hc_y[:actual_pre+1],
        "start_frame":      start_frame,
        "end_frame":        end_frame,
        "cross_seg_split":  cross_seg_offset if cross_seg_offset > 0 else None,
    }


# ═══════════════════════════════════════════════════════════════
#  CROSS-SEG XY LOADING
# ═══════════════════════════════════════════════════════════════

def load_xy_for_epoch(seg_folders, seg_idx, start_raw, end_raw,
                      log_fn, bodypart="head_center"):
    """Return (bp_x, bp_y, fps, cross_seg_offset) or None."""
    def load_seg(sf):
        dlc = load_dlc(sf / "_DLC_analysis" / "dlc_data.csv")
        if dlc is None:
            return None, None, None
        xy = get_bodypart_xy(dlc, bodypart)
        if xy is None:
            return None, None, None
        return xy[0], xy[1], get_fps(dlc)

    seg_a = seg_folders[seg_idx]

    if start_raw != "prev" and end_raw != "next":
        dlc = load_dlc(seg_a / "_DLC_analysis" / "dlc_data.csv")
        if dlc is None:
            return None
        xy = get_bodypart_xy(dlc, bodypart)
        if xy is None:
            log_fn(f"  [WARN] bodypart '{bodypart}' not found or all "
                   f"low-likelihood: {seg_a.name}")
            return None
        bp_x, bp_y = xy
        sf, ef = int(start_raw), int(end_raw)
        # return full array so compute_epoch can access pre-window frames
        return bp_x, bp_y, get_fps(dlc), sf

    if end_raw == "next":
        if seg_idx + 1 >= len(seg_folders):
            log_fn(f"  [WARN] 'next' but no next seg: {seg_a.name}")
            return None
        seg_b   = seg_folders[seg_idx + 1]
        laser_b = load_json(seg_b / "_DLC_analysis" / "laser_timepoints.json")
        if laser_b is None:
            return None
        end_b = None
        for ft in laser_b.get("food", {}).get("food-time", []):
            if isinstance(ft, (list, tuple)) and ft[0] == "prev":
                end_b = int(ft[1])
                break
        if end_b is None:
            log_fn(f"  [WARN] no 'prev' in {seg_b.name}")
            return None

        xa, ya, fps_a = load_seg(seg_a)
        xb, yb, _     = load_seg(seg_b)
        if xa is None or xb is None:
            return None

        sf_a   = int(start_raw)
        # Concatenate FULL xa (not xa[sf_a:]) so pre-window frames are accessible.
        # compute_epoch will use sf_a as start_frame to slice correctly.
        bp_x   = np.concatenate([xa, xb[:end_b+1]])
        bp_y   = np.concatenate([ya, yb[:end_b+1]])
        ef_abs = len(xa) + end_b   # end frame index in concatenated array
        log_fn(f"  [CROSS-SEG] {seg_a.name}[{sf_a}:end]+"
               f"{seg_b.name}[0:{end_b}]  total={len(bp_x)}")
        return bp_x, bp_y, fps_a, sf_a

    return None  # standalone "prev"


# ═══════════════════════════════════════════════════════════════
#  MOUSE-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════

def analyze_mouse(mouse_path, px_to_mm, smooth_sec,
                  bodypart="head_center",
                  speed_bodypart="head_center",
                  pre_sec=PRE_SEC_DEFAULT,
                  log_fn=print):
    param = load_json(mouse_path / "_analysis_param.json")
    if param is None:
        log_fn(f"  [SKIP] no param: {mouse_path.name}")
        return None

    continuous  = param["Time"]["Continuous"]
    seg_folders = get_segment_folders(mouse_path)
    if not seg_folders:
        return None

    epochs_by_group = {g: [] for g in TIME_GROUPS}
    arena_by_group  = {g: None for g in TIME_GROUPS}

    for seg_idx, seg_folder in enumerate(seg_folders):
        if seg_idx >= len(continuous):
            break
        seg_start_min = continuous[seg_idx][0]
        group_name    = next(
            (g for g, pred in TIME_GROUPS.items() if pred(seg_start_min)), None)
        if group_name is None:
            continue

        dlc_dir    = seg_folder / "_DLC_analysis"
        laser_data = load_json(dlc_dir / "laser_timepoints.json")
        if laser_data is None:
            continue

        food_section   = laser_data.get("food", {})
        food_times     = food_section.get("food-time", [])
        food_durations = food_section.get("duration", [])
        if not food_times:
            continue

        if arena_by_group[group_name] is None:
            arena_by_group[group_name] = get_arena(laser_data)

        for epoch_idx, ft in enumerate(food_times):
            if not isinstance(ft, (list, tuple)) or len(ft) != 2:
                continue
            start_raw, end_raw = ft[0], ft[1]
            if start_raw == "prev":
                continue

            # check food location CSV exists
            csv_p = csv_path_for_epoch(dlc_dir / "laser_timepoints.json",
                                       epoch_idx)
            if not csv_p.exists():
                log_fn(f"  [SKIP] no food CSV: {csv_p.name}")
                continue

            food_df = pd.read_csv(csv_p)

            resolved = load_xy_for_epoch(
                seg_folders, seg_idx, start_raw, end_raw,
                log_fn, bodypart=bodypart)
            if resolved is None:
                continue
            hc_x, hc_y, fps, cross_offset = resolved

            # load speed body part (may differ from distance body part)
            if speed_bodypart != bodypart:
                spd_resolved = load_xy_for_epoch(
                    seg_folders, seg_idx, start_raw, end_raw,
                    log_fn, bodypart=speed_bodypart)
                spd_x = spd_resolved[0] if spd_resolved else None
                spd_y = spd_resolved[1] if spd_resolved else None
            else:
                spd_x, spd_y = None, None  # reuse hc_x/hc_y in compute_epoch

            # cross_offset = sf_a (start frame in full concatenated array)
            # For cross-seg: sf_abs = sf_a, ef_abs = len(xa)+end_b stored
            # in the returned tuple 4th element (now sf_a for both cases)
            if end_raw == "next" or str(start_raw).lower() == "next":
                sf_abs = cross_offset          # sf_a in full concat array
                ef_abs = len(hc_x) - 1        # last frame of concat array
            else:
                sf_abs = int(start_raw)        # frame index in full seg array
                ef_abs = int(end_raw)
            ep = compute_epoch(
                hc_x, hc_y, food_df,
                sf_abs, ef_abs,
                px_to_mm, fps, smooth_sec, 0,
                pre_sec=pre_sec,
                spd_x_full=spd_x, spd_y_full=spd_y)
            if ep is None:
                continue

            dur_entry = (food_durations[epoch_idx]
                         if epoch_idx < len(food_durations) else None)
            if dur_entry is not None and len(dur_entry) > 0:
                ep["eating_onset_sec"] = float(dur_entry[0])
            else:
                ep["eating_onset_sec"] = None

            epochs_by_group[group_name].append(ep)
            onset = ep["eating_onset_sec"]
            onset_str = f"  eating_onset={onset:.0f}s" if onset else ""
            log_fn(f"  [OK] seg={seg_folder.name} ep={epoch_idx} "
                   f"group={group_name} dur={ep['duration']:.1f}s "
                   f"unit={ep['unit']}{onset_str}")

    if not any(epochs_by_group.values()):
        return None

    return {
        "mouse_name":       mouse_path.name,
        "experiment_group": "",
        "epochs_by_group":  epochs_by_group,
        "arena_by_group":   arena_by_group,
    }


# ═══════════════════════════════════════════════════════════════
#  INTERPOLATION  (forward time, t=0 on left)
# ═══════════════════════════════════════════════════════════════

def interpolate_traces(traces, post_sec, n_points, log_scale,
                       pre_sec: float = 0.0):
    """
    traces   : list of (time_axis, values)  -  time may start negative (pre_sec)
    pre_sec  : seconds before t=0 to include in linear axis
    Returns (common_t, mean, sem, n_contributing)
    """
    valid = [(t, v) for t, v in traces if len(t) >= 2]
    if not valid:
        return None, None, None, None

    if log_scale:
        # sqrt-spaced from 0 to post_sec (no pre-window on log axis)
        common_t = np.linspace(0, np.sqrt(max(post_sec, 0.1)),
                               n_points) ** 2
        common_t[0] = 0.0
    else:
        common_t = np.linspace(-pre_sec, post_sec, n_points)

    arr = []
    for t_arr, v_arr in valid:
        interp = np.interp(common_t, t_arr, v_arr,
                           left=np.nan, right=np.nan)
        arr.append(interp)

    arr  = np.array(arr)
    mean = np.nanmean(arr, axis=0)
    sem  = (np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1 else np.zeros(n_points))
    n_c  = np.sum(~np.isnan(arr), axis=0)
    return common_t, mean, sem, n_c


# ═══════════════════════════════════════════════════════════════
#  CURVE PLOT PDF
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
#  AUC COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_auc(time_axis: np.ndarray,
                values: np.ndarray,
                auc_sec: float,
                auc_start: float = 0.0) -> float:
    """
    Trapezoidal AUC from auc_start to auc_sec.
    Returns NaN if insufficient data in range.
    """
    mask = (time_axis >= auc_start) & (time_axis <= auc_sec)
    t = time_axis[mask]
    v = values[mask]
    if len(t) < 2:
        return np.nan
    if t[-1] < auc_sec:
        t = np.append(t, auc_sec)
        v = np.append(v, v[-1])
    # np.trapz was removed in NumPy 2.0 and renamed to np.trapezoid.
    _trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz_fn(v, t))


def collect_auc(agg_by_exp_group: dict,
                group_name: str,
                metric: str,
                auc_sec: float,
                auc_start: float = 0.0) -> dict:
    """
    Returns {exp_group: {mouse_name: auc_value}} for a given
    time_group and metric.
    """
    result = {}
    for exp_group, mouse_list in agg_by_exp_group.items():
        result[exp_group] = {}
        for m in mouse_list:
            aucs = []
            for ep in m["epochs_by_group"].get(group_name, []):
                if metric == "speed":
                    a = compute_auc(ep["time_axis"], ep[metric],
                                    ep["duration"], 0.0)
                else:
                    a = compute_auc(ep["time_axis"], ep[metric],
                                    auc_sec, auc_start)
                if not np.isnan(a):
                    aucs.append(a)
            if aucs:
                result[exp_group][m["mouse_name"]] = float(np.mean(aucs))
    return result

def _plot_metric(ax, agg_by_exp_group, group_name, metric,
                 ylabel, log_scale, post_sec,
                 pre_sec=PRE_SEC_DEFAULT, n_points=N_INTERP):
    has_data = False
    unit     = "px"   # updated from first epoch found

    # for log scale: use max actual duration across all traces (no cap)
    if log_scale:
        all_durations = [
            ep["duration"]
            for mouse_list in agg_by_exp_group.values()
            for m in mouse_list
            for ep in m["epochs_by_group"].get(group_name, [])
        ]
        t_max = max(all_durations) if all_durations else post_sec
    else:
        t_max = post_sec

    for exp_group, mouse_list in agg_by_exp_group.items():
        color  = COLORS.get(exp_group, DEFAULT_COLOR)
        traces = []
        for m in mouse_list:
            for ep in m["epochs_by_group"].get(group_name, []):
                traces.append((ep["time_axis"], ep[metric]))
                unit = ep.get("unit", "px")
        if not traces:
            continue
        has_data = True

        t_c, mn, sem, n_c = interpolate_traces(
            traces, t_max, n_points, log_scale,
            pre_sec=0.0 if log_scale else pre_sec)
        if t_c is None:
            continue

        # individual thin lines  -  downsampled to same grid as mean curve
        t_min_show = 0.0 if log_scale else -pre_sec
        for t_arr, v_arr in traces:
            ind_v = np.interp(t_c, t_arr, v_arr,
                              left=np.nan, right=np.nan)
            valid = ~np.isnan(ind_v)
            if valid.any():
                ax.plot(t_c[valid], ind_v[valid],
                        color=color, lw=0.5, alpha=0.18)

        # sparse overlay
        n_max = n_c.max() if n_c.max() > 0 else 1
        sparse = (n_c / n_max) < 0.5
        if sparse.any():
            ylim = ax.get_ylim()
            ax.fill_between(t_c, ylim[0], ylim[1],
                            where=sparse, color="grey",
                            alpha=0.06, zorder=0)

        ax.plot(t_c, mn, color=color, lw=2.0,
                label=f"{exp_group} (n={len(traces)})")
        ax.fill_between(t_c, mn - sem, mn + sem,
                        color=color, alpha=0.2)

    if not has_data:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey", fontsize=10)
        return

    if log_scale:
        # sqrt scale: transform axis ticks manually
        ax.set_xlim(0, t_max)
        # place ticks at "nice" times, spaced evenly in sqrt domain
        nice = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100,
                121, 144, 169, 196, 225, 256, 289, 324, 361, 400]
        ticks = [v for v in nice if v <= t_max]
        if not ticks or ticks[-1] < t_max * 0.8:
            ticks.append(int(t_max))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(v) for v in ticks], fontsize=7)
        # apply sqrt transform via matplotlib scale
        ax.set_xscale("function",
                      functions=(np.sqrt, lambda x: x**2))
        ax.set_xlabel(
            f"Time after food placed (s, sqrt scale, max={t_max:.0f}s)",
            fontsize=9)
    else:
        ax.set_xlim(-pre_sec, post_sec)
        ax.axvline(0, color="green", lw=1.0, ls="--", alpha=0.5,
                   label="food placed (t=0)")
        ax.set_xlabel("Time relative to food placement (s)", fontsize=9)

    dist_ylabel = f"Distance to food ({'mm' if unit == 'mm' else 'px'})"
    rel_ylabel  = "Relative distance (norm. to t=0)"
    spd_ylabel  = f"Velocity ({'mm/s' if unit == 'mm' else 'px/s'})"
    if metric == "distance":
        ylabel = dist_ylabel
    elif metric == "rel_distance":
        ylabel = rel_ylabel
    else:
        ylabel = spd_ylabel
    ax.set_ylabel(ylabel, fontsize=10)
    # uniform y-axis upper limit for all line plots
    ax.set_ylim(bottom=0, top=PARAMS["line_plot_ymax"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, loc="upper right")

    # eating-onset markers below x-axis (linear axis only)
    # Per individual: circle (<=post_sec) or x-cross (>post_sec, clipped to post_sec)
    # Per group: mean line (<=post_sec) or line to post_sec + text annotation (>post_sec)
    if not log_scale:
        y_lo, y_hi = ax.get_ylim()
        gap        = (y_hi - y_lo) * 0.025
        seg_base   = y_lo - gap
        grp_list   = [g for g in agg_by_exp_group if agg_by_exp_group[g]]
        any_onset  = False

        for gi, exp_group in enumerate(grp_list):
            color  = COLORS.get(exp_group, DEFAULT_COLOR)
            y_seg  = seg_base - gi * gap * 1.8

            onsets = []
            for m in agg_by_exp_group[exp_group]:
                for ep in m["epochs_by_group"].get(group_name, []):
                    onset = ep.get("eating_onset_sec")
                    if onset is None:
                        continue
                    onsets.append(float(onset))
                    any_onset = True
                    if float(onset) <= post_sec:
                        ax.scatter(float(onset), y_seg, color=color,
                                   s=28, marker="o", zorder=6,
                                   alpha=0.85, facecolors="none",
                                   edgecolors=color, linewidths=1.2)
                    else:
                        # clipped: draw x at post_sec with y jitter
                        jitter_y = y_seg + np.random.uniform(-gap*0.35, gap*0.35)
                        ax.scatter(post_sec, jitter_y, color="red",
                                   s=28, marker="x", zorder=6,
                                   linewidths=1.2, alpha=0.9)

            if not onsets:
                continue

            # group mean line
            mean_onset = np.mean(onsets)
            line_end   = min(mean_onset, post_sec)
            lbl = f"{exp_group} onset (mean)"
            ax.plot([0, line_end], [y_seg, y_seg],
                    color=color, lw=2.0, alpha=0.65,
                    solid_capstyle="round", label=lbl)

            if mean_onset > post_sec:
                # annotate beyond axis
                ax.text(post_sec * 0.99, y_seg + gap * 0.4,
                        f"mean={mean_onset:.0f}s",
                        ha="right", va="bottom",
                        fontsize=6.5, color=color, style="italic")
            else:
                ax.text(mean_onset + post_sec * 0.01, y_seg + gap * 0.4,
                        f"mean={mean_onset:.0f}s",
                        ha="left", va="bottom",
                        fontsize=6.5, color=color, style="italic")

        if any_onset:
            n_grps = len(grp_list)
            bottom = seg_base - n_grps * gap * 2.2
            ax.set_ylim(bottom, y_hi)
            ax.axhline(y_lo, color="black", lw=0.3, alpha=0.25)


def _plot_auc_summary(ax, agg_by_exp_group: dict,
                      group_names: list,
                      metric: str, auc_sec: float, unit: str,
                      auc_start: float = 0.0):
    """
    AUC summary: all time groups in one chart.
    Clustered bars  -  one cluster per time_group,
    one bar per exp_group. Individual dots overlaid.
    """
    exp_groups = list(agg_by_exp_group.keys())
    n_exp = len(exp_groups)
    n_tg  = len(group_names)
    if n_exp == 0 or n_tg == 0:
        ax.axis("off")
        return

    np.random.seed(42)
    bar_w   = 0.7 / max(n_exp, 1)
    offsets = np.linspace(-(n_exp-1)/2*bar_w, (n_exp-1)/2*bar_w, n_exp)

    for ei, exp_group in enumerate(exp_groups):
        mouse_list = agg_by_exp_group[exp_group]
        color      = COLORS.get(exp_group, DEFAULT_COLOR)
        for ti, gname in enumerate(group_names):
            aucs = []
            for m in mouse_list:
                for ep in m["epochs_by_group"].get(gname, []):
                    if metric == "speed":
                        a = compute_auc(ep["time_axis"], ep[metric],
                                        ep["duration"], 0.0)
                    else:
                        a = compute_auc(ep["time_axis"], ep[metric],
                                        auc_sec, auc_start)
                    if not np.isnan(a):
                        aucs.append(a)
            if not aucs:
                continue
            x    = ti + offsets[ei]
            mean = np.mean(aucs)
            sem  = np.std(aucs) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0
            ax.bar(x, mean, width=bar_w*0.85, color=color, alpha=0.75,
                   yerr=sem, capsize=3,
                   error_kw={"elinewidth": 1.2, "ecolor": "black"},
                   label=exp_group if ti == 0 else "_nolegend_")
            jitter = np.random.uniform(-bar_w*0.25, bar_w*0.25, len(aucs))
            ax.scatter(x + jitter, aucs, color=color,
                       s=20, zorder=5, alpha=0.9,
                       edgecolors="white", linewidths=0.4)

    short = [g.replace("After ", "A.").replace("Before", "Bef.")
             for g in group_names]
    ax.set_xticks(range(n_tg))
    ax.set_xticklabels(short, fontsize=7, rotation=15, ha="right")
    dist_u = "mm" if unit == "mm" else "px"
    spd_u  = "mm/s" if unit == "mm" else "px/s"
    ylabel = (f"AUC dist. ({dist_u}\u00b7s)" if metric == "distance"
              else f"AUC vel. ({spd_u}\u00b7s)")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"AUC summary\n({auc_start:.0f}\u2013{auc_sec:.0f}s)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.6)

def generate_curve_pdf(agg_by_exp_group, output_path,
                        post_sec, pre_sec=PRE_SEC_DEFAULT,
                        n_points=N_INTERP, auc_start=0.0,
                        auc_end=None, log_fn=print):
    group_names = list(TIME_GROUPS.keys())
    # Only the linear-time rows are produced; the sqrt-time variants were
    # never used in the manuscript.
    metrics = [
        ("distance", "linear"),
        ("speed",    "linear"),
    ]

    # detect unit from first available epoch
    unit = "px"
    for ml in agg_by_exp_group.values():
        for m in ml:
            for eps in m["epochs_by_group"].values():
                if eps:
                    unit = eps[0].get("unit", "px")
                    break

    # layout: 4 cols
    #   col 0-2 : curve panels (Before / After 0-50 / After 60+)
    #   col 3   : AUC summary (all time groups, linear rows only)
    n_rows       = len(metrics)
    width_ratios = [3.5, 3.5, 3.5, 2.5]

    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(sum(width_ratios) * 1.05, 4 * n_rows),
        gridspec_kw={"width_ratios": width_ratios})
    fig.suptitle("Food Approaching Analysis  (t=0: food placed)",
                 fontsize=13, y=1.01)
    plt.subplots_adjust(hspace=0.52, wspace=0.38)

    for ri, (metric, axis_type) in enumerate(metrics):
        ls    = axis_type == "log"
        label = {"distance": "Distance", "speed": "Velocity"}.get(metric, metric)

        # cols 0-2: curve panels
        for gi, gname in enumerate(group_names):
            ax = axes[ri][gi]
            ax.set_title(f"{gname}\n{label}", fontsize=10)
            _plot_metric(ax, agg_by_exp_group, gname, metric,
                         "", ls, post_sec, pre_sec=pre_sec,
                         n_points=n_points)

        # col 3: AUC summary (linear rows only)
        ax_auc = axes[ri][3]
        if not ls:
            auc_cap = auc_end if auc_end is not None else post_sec
            _plot_auc_summary(ax_auc, agg_by_exp_group,
                              group_names, metric, auc_cap, unit,
                              auc_start=auc_start)
        else:
            ax_auc.axis("off")

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log_fn(f"[SAVED] {output_path}")


# ═══════════════════════════════════════════════════════════════
#  TRAJECTORY PDF
# ═══════════════════════════════════════════════════════════════

def _draw_arena(ax, arena):
    if arena is None:
        return
    order = ["up-left", "up-right", "down-right", "down-left", "up-left"]
    xs = [arena[k][0] for k in order]
    ys = [arena[k][1] for k in order]
    ax.plot(xs, ys, color="black", lw=1.5, zorder=2)


def _arena_limits(arena, margin=30):
    if arena is None:
        return (0, 400), (0, 420)
    all_x = [v[0] for v in arena.values()]
    all_y = [v[1] for v in arena.values()]
    return ((min(all_x) - margin, max(all_x) + margin),
            (min(all_y) - margin, max(all_y) + margin))


def _draw_epoch_panel(ax, ep, arena, ep_label):
    xlim, ylim = _arena_limits(arena)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
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

    hc_x = ep["hc_x"]; hc_y = ep["hc_y"]
    fx   = ep["food_x"]; fy  = ep["food_y"]
    n    = len(hc_x)
    if n < 2:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="grey")
        return

    # pre-food trajectory (grey dashed, t<0)
    hc_x_pre = ep.get("hc_x_pre")
    hc_y_pre = ep.get("hc_y_pre")
    if hc_x_pre is not None and len(hc_x_pre) >= 2:
        ax.plot(hc_x_pre, hc_y_pre, color="grey", lw=1.0, ls="--",
                alpha=0.6, zorder=3, label="pre-food path")
        ax.scatter(hc_x_pre[0], hc_y_pre[0], color="grey", s=14,
                   marker="^", zorder=6, alpha=0.8)   # pre-window start

    # head trajectory (colour = time)
    pts  = np.array([hc_x, hc_y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=plt.cm.cool,
                          norm=plt.Normalize(0, max(n-1, 1)),
                          linewidth=1.2, alpha=0.85, zorder=4)
    lc.set_array(np.arange(n - 1))
    ax.add_collection(lc)

    # food trajectory (dotted orange)
    ax.plot(fx, fy, color="orange", lw=0.8, ls=":",
            alpha=0.7, zorder=3, label="food path")
    ax.scatter(fx[0],  fy[0],  color="orange", s=20,
               marker="*", zorder=6)   # initial food position
    ax.scatter(fx[-1], fy[-1], color="red",    s=20,
               marker="*", zorder=6)   # final food position

    # cross-seg boundary
    split = ep.get("cross_seg_split")
    if split and 0 < split < n:
        ax.scatter(hc_x[split], hc_y[split], color="orange",
                   s=16, marker="D", zorder=6)

    ax.scatter(hc_x[0],  hc_y[0],  color="blue", s=18,
               marker="o", zorder=7)
    ax.scatter(hc_x[-1], hc_y[-1], color="navy", s=18,
               marker="s", zorder=7)

    handles = [
        Line2D([0],[0], color="grey", lw=1.0, ls="--", label="pre-food path"),
        Line2D([0],[0], color="deepskyblue", lw=1.2, label="head path"),
        Line2D([0],[0], color="orange", lw=1, ls=":", label="food path"),
        Line2D([0],[0], color="blue",  marker="o", ls="none",
               markersize=4, label="start"),
        Line2D([0],[0], color="navy",  marker="s", ls="none",
               markersize=4, label="end"),
    ]
    ax.legend(handles=handles, fontsize=4.5, loc="lower right",
              framealpha=0.6)


def generate_trajectory_pdf(all_mouse_data, output_path, log_fn=print):
    log_fn("Generating trajectory PDF...")
    group_names = list(TIME_GROUPS.keys())
    valid = [m for m in all_mouse_data
             if any(m["epochs_by_group"].get(g) for g in group_names)]
    if not valid:
        log_fn("  [SKIP] no trajectory data.")
        return

    with PdfPages(str(output_path)) as pdf:
        for mouse in valid:
            m_name = mouse["mouse_name"]
            m_grp  = mouse["experiment_group"]
            a_by_g = mouse["arena_by_group"]

            row_plan = [(g,
                         mouse["epochs_by_group"].get(g, []),
                         max(1, math.ceil(
                             len(mouse["epochs_by_group"].get(g, []))
                             / TRAJ_COLS)))
                        for g in group_names]

            gs_ratios = []
            for _, _, nr in row_plan:
                gs_ratios += [0.22] + [1.0] * nr

            fig_h = 1.2 + sum(nr for _, _, nr in row_plan) * PANEL_SIZE * 1.1
            fig   = plt.figure(figsize=(TRAJ_COLS * PANEL_SIZE, fig_h))
            fig.suptitle(f"{m_name}  [{m_grp}]",
                         fontsize=11, fontweight="bold")
            gs = gridspec.GridSpec(len(gs_ratios), TRAJ_COLS, figure=fig,
                                   height_ratios=gs_ratios,
                                   hspace=0.55, wspace=0.35)
            gs_row = 0

            for g, evs, n_pr in row_plan:
                n_ep  = len(evs)
                arena = a_by_g.get(g)

                ax_h = fig.add_subplot(gs[gs_row, :])
                ax_h.axis("off")
                ax_h.text(0.0, 0.5,
                           f"  {g}   -   {n_ep} epoch{'s' if n_ep!=1 else ''}",
                           transform=ax_h.transAxes, fontsize=9,
                           fontweight="bold", va="center", color="white",
                           bbox=dict(boxstyle="round,pad=0.3",
                                     facecolor="#444", edgecolor="none"))
                gs_row += 1

                if n_ep == 0:
                    ax_nd = fig.add_subplot(gs[gs_row, 0])
                    ax_nd.axis("off")
                    ax_nd.text(0.5, 0.5, "no data",
                               transform=ax_nd.transAxes,
                               ha="center", va="center",
                               fontsize=9, color="grey")
                else:
                    for ei, ev in enumerate(evs):
                        r = ei // TRAJ_COLS
                        c = ei %  TRAJ_COLS
                        ax = fig.add_subplot(gs[gs_row + r, c])
                        _draw_epoch_panel(ax, ev, arena, f"Epoch {ei+1}")
                    last_r    = (n_ep - 1) // TRAJ_COLS
                    remainder = n_ep % TRAJ_COLS
                    if remainder:
                        for bc in range(remainder, TRAJ_COLS):
                            fig.add_subplot(gs[gs_row+last_r, bc]).axis("off")

                gs_row += n_pr

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log_fn(f"[SAVED] {output_path}")


# ═══════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════

def export_curves_csv(agg_by_exp_group, output_path, post_sec, pre_sec,
                      n_points, log_fn=print):
    """
    Source data for the line plots (rows of the analysis PDF).

    Values are interpolated onto the same time grid the figure uses, so this
    file reproduces the plotted curves exactly.  One row per
    (exp_group, time_group, metric, trace_id, time point); trace_id is
    "<mouse>__ep<N>" for the thin individual lines and "GROUP_MEAN" for the
    thick mean curve (the `sem` and `n` columns apply to GROUP_MEAN rows).
    """
    rows = []
    for gname in TIME_GROUPS:
        for exp_group, mouse_list in agg_by_exp_group.items():
            for metric in ("distance", "speed"):
                traces, ids, unit = [], [], "px"
                for m in mouse_list:
                    for ei, ep in enumerate(m["epochs_by_group"].get(gname, [])):
                        traces.append((ep["time_axis"], ep[metric]))
                        ids.append(f"{m['mouse_name']}__ep{ei}")
                        unit = ep.get("unit", "px")
                if not traces:
                    continue

                t_c, mn, sem, n_c = interpolate_traces(
                    traces, post_sec, n_points, False, pre_sec=pre_sec)
                if t_c is None:
                    continue

                for (t_arr, v_arr), tid in zip(traces, ids):
                    vals = np.interp(t_c, t_arr, v_arr, left=np.nan, right=np.nan)
                    for t, v in zip(t_c, vals):
                        if np.isnan(v):
                            continue
                        rows.append({
                            "exp_group":  exp_group,
                            "time_group": gname,
                            "metric":     METRIC_LABEL[metric],
                            "trace_id":   tid,
                            "time_s":     round(float(t), 4),
                            "value":      round(float(v), 4),
                            "sem":        "",
                            "n":          1,
                            "unit":       unit,
                        })
                for t, m_, s_, n_ in zip(t_c, mn, sem, n_c):
                    rows.append({
                        "exp_group":  exp_group,
                        "time_group": gname,
                        "metric":     METRIC_LABEL[metric],
                        "trace_id":   "GROUP_MEAN",
                        "time_s":     round(float(t), 4),
                        "value":      "" if np.isnan(m_) else round(float(m_), 4),
                        "sem":        "" if np.isnan(s_) else round(float(s_), 4),
                        "n":          int(n_),
                        "unit":       unit,
                    })

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        log_fn(f"[SAVED] {output_path}  ({len(rows)} rows)")
    else:
        log_fn("No curve data to save.")


def export_auc_csv(all_mouse_data, output_path, post_sec,
                   auc_start=0.0, auc_end=None, log_fn=print):
    """
    Source data for the AUC summary panels, plus the eating-onset markers
    drawn under the x-axis of the line plots.  One row per epoch per metric.
    Distance AUC uses [auc_start, auc_end]; velocity AUC uses the full epoch.
    """
    a_end_default = auc_end if auc_end is not None else post_sec
    rows = []
    for mouse in all_mouse_data:
        for tg, epochs in mouse["epochs_by_group"].items():
            for ei, ep in enumerate(epochs):
                onset = ep.get("eating_onset_sec")
                for metric in ("distance", "speed"):
                    if metric == "distance":
                        a_start, a_end = auc_start, a_end_default
                    else:
                        a_start, a_end = 0.0, ep["duration"]
                    val = compute_auc(ep["time_axis"], ep[metric], a_end, a_start)
                    rows.append({
                        "mouse":            mouse["mouse_name"],
                        "exp_group":        mouse["experiment_group"],
                        "time_group":       tg,
                        "epoch_idx":        ei,
                        "metric":           METRIC_LABEL[metric],
                        "auc":              "" if np.isnan(val) else round(val, 4),
                        "auc_start_s":      a_start,
                        "auc_end_s":        round(a_end, 2),
                        "eating_onset_sec": round(onset, 1) if onset else "",
                        "unit":             ep["unit"],
                    })

    if rows:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        log_fn(f"[SAVED] {output_path}  ({len(rows)} rows)")
    else:
        log_fn("No AUC data to save.")


# ═══════════════════════════════════════════════════════════════
#  RUN ANALYSIS
# ═══════════════════════════════════════════════════════════════

def run_analysis(groups, px_to_mm, smooth_sec,
                 post_sec, output_dir,
                 bodypart="head_center",
                 speed_bodypart="head_center",
                 pre_sec=PRE_SEC_DEFAULT,
                 n_points=N_INTERP,
                 auc_start=0.0,
                 auc_end=None,
                 make_trajectory_pdf=None,
                 log_fn=print):

    agg_by_exp_group = {}
    all_mouse_data   = []

    log_fn(f"Distance body part: {bodypart}  |  Velocity body part: {speed_bodypart}")
    for exp_group, mice in groups.items():
        log_fn(f"\n[Group: {exp_group}]")
        mouse_results = []
        for m in mice:
            if not m["exists"]:
                log_fn(f"  [MISSING] {m['path']}")
                continue
            log_fn(f"  Mouse: {m['name']}")
            res = analyze_mouse(m["path"], px_to_mm, smooth_sec,
                                bodypart=bodypart,
                                speed_bodypart=speed_bodypart,
                                pre_sec=pre_sec,
                                log_fn=log_fn)
            if res is None:
                continue
            res["experiment_group"] = exp_group
            mouse_results.append(res)
            all_mouse_data.append(res)
            log_fn("    -> " + ", ".join(
                f"{g}={len(res['epochs_by_group'].get(g,[]))}"
                for g in TIME_GROUPS))
        if mouse_results:
            agg_by_exp_group[exp_group] = mouse_results

    if not agg_by_exp_group:
        log_fn("[ERROR] No valid data found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # filename suffix: bodypart + px_to_mm scale
    px_str = ("_" + f"{px_to_mm:.4f}mm".replace(".", "p") + "perpx"
              if px_to_mm is not None else "_px")
    bp_tag = bodypart if bodypart == speed_bodypart              else f"{bodypart}-spd{speed_bodypart}"
    suffix = f"_{bp_tag}{px_str}"

    generate_curve_pdf(
        agg_by_exp_group,
        output_dir / f"food_approaching_analysis{suffix}.pdf",
        post_sec, pre_sec=pre_sec, n_points=n_points,
        auc_start=auc_start, auc_end=auc_end, log_fn=log_fn)

    export_curves_csv(
        agg_by_exp_group,
        output_dir / f"food_curves{suffix}.csv",
        post_sec, pre_sec, n_points, log_fn=log_fn)

    export_auc_csv(
        all_mouse_data,
        output_dir / f"food_auc{suffix}.csv",
        post_sec, auc_start=auc_start, auc_end=auc_end, log_fn=log_fn)

    if PARAMS["make_trajectory_pdf"] if make_trajectory_pdf is None else make_trajectory_pdf:
        generate_trajectory_pdf(
            all_mouse_data,
            output_dir / f"food_approaching_trajectories{suffix}.pdf",
            log_fn)

    log_fn(f"\n[DONE] Outputs: {output_dir}")


# ═══════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Food Approaching Analysis")
        self.geometry("780x560")
        self.resizable(True, True)
        self.config_data: dict = {}
        self.groups:      dict = {}
        self.main_folder: Optional[Path] = None
        self._build_ui()
        self._autoload_default_config()

    def _build_ui(self):
        tk.Label(self, text="Food Approaching Analysis",
                 font=("Arial", 14, "bold")).pack(pady=(14, 2))

        top = tk.LabelFrame(self, text="1. Select Main Folder",
                             padx=8, pady=6)
        top.pack(fill="x", padx=12, pady=(0, 4))
        self.path_var = tk.StringVar(value="(not selected)")
        tk.Label(top, textvariable=self.path_var, anchor="w",
                 relief="sunken", width=66,
                 bg="white").pack(side="left", padx=(0,6))
        tk.Button(top, text="Browse...", command=self._sel_folder,
                  bg="#1565c0", fg="white", padx=8).pack(side="left")

        param = tk.LabelFrame(self, text="2. Parameters",
                               padx=10, pady=8)
        param.pack(fill="x", padx=12, pady=4)

        tk.Label(param, text="px -> mm (mm/px):\n(leave blank = use px)").grid(
            row=0, column=0, sticky="w", padx=(0,6))
        self.pxtomm_var = tk.StringVar(
            value="" if PARAMS["px_to_mm"] is None else str(PARAMS["px_to_mm"]))
        tk.Entry(param, textvariable=self.pxtomm_var,
                 width=12).grid(row=0, column=1, sticky="w")

        tk.Label(param, text="Velocity smooth (s):").grid(
            row=0, column=2, sticky="w", padx=(20,6))
        self.smooth_var = tk.DoubleVar(value=PARAMS["speed_smooth_s"])
        tk.Spinbox(param, textvariable=self.smooth_var,
                   from_=0.05, to=2.0, increment=0.05,
                   format="%.2f", width=8).grid(
                       row=0, column=3, sticky="w")

        tk.Label(param, text="Distance body part:").grid(
            row=2, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.bodypart_var = tk.StringVar(value=PARAMS["distance_bodypart"])
        ttk.Combobox(param, textvariable=self.bodypart_var,
                     values=["head_center", "snout",
                             "centroid", "body_center"],
                     state="readonly", width=14).grid(
                         row=2, column=1, sticky="w", pady=(6,0))

        tk.Label(param, text="Velocity body part:").grid(
            row=2, column=2, sticky="w", padx=(20,6), pady=(6,0))
        self.speed_bodypart_var = tk.StringVar(value=PARAMS["velocity_bodypart"])
        ttk.Combobox(param, textvariable=self.speed_bodypart_var,
                     values=["head_center", "snout",
                             "centroid", "body_center"],
                     state="readonly", width=14).grid(
                         row=2, column=3, sticky="w", pady=(6,0))

        tk.Label(param,
                 text="(snout/centroid: likelihood < 0.8 interpolated out)",
                 fg="#888", font=("Arial", 8)).grid(
                     row=4, column=0, columnspan=4,
                     sticky="w", padx=(0,0), pady=(4,0))

        tk.Label(param, text="Post window (s):").grid(
            row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.post_var = tk.DoubleVar(value=PARAMS["post_window_s"])
        tk.Spinbox(param, textvariable=self.post_var,
                   from_=10, to=600, increment=10, width=8).grid(
                       row=1, column=1, sticky="w", pady=(6,0))

        tk.Label(param, text="Pre window (s):").grid(
            row=1, column=2, sticky="w", padx=(20,6), pady=(6,0))
        self.pre_var = tk.DoubleVar(value=PARAMS["pre_window_s"])
        tk.Spinbox(param, textvariable=self.pre_var,
                   from_=0, to=60, increment=5, width=8).grid(
                       row=1, column=3, sticky="w", pady=(6,0))

        tk.Label(param, text="Plot points (n):").grid(
            row=3, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.npoints_var = tk.IntVar(value=PARAMS["plot_points"])
        tk.Spinbox(param, textvariable=self.npoints_var,
                   from_=6, to=5000, increment=6, width=8).grid(
                       row=3, column=1, sticky="w", pady=(6,0))
        tk.Label(param, text="(more = finer curve)",
                 fg="#888", font=("Arial", 8)).grid(
                     row=3, column=2, columnspan=2,
                     sticky="w", padx=(10,0), pady=(6,0))

        tk.Label(param, text="AUC start (s):").grid(
            row=5, column=0, sticky="w", padx=(0,6), pady=(6,0))
        self.auc_start_var = tk.DoubleVar(value=PARAMS["auc_start_s"])
        tk.Spinbox(param, textvariable=self.auc_start_var,
                   from_=0, to=300, increment=5, width=8).grid(
                       row=5, column=1, sticky="w", pady=(6,0))

        tk.Label(param, text="AUC end (s):").grid(
            row=5, column=2, sticky="w", padx=(20,6), pady=(6,0))
        self.auc_end_var = tk.DoubleVar(value=PARAMS["auc_end_s"])
        tk.Spinbox(param, textvariable=self.auc_end_var,
                   from_=10, to=600, increment=10, width=8).grid(
                       row=5, column=3, sticky="w", pady=(6,0))
        tk.Label(param, text="AUC range",
                 fg="#888", font=("Arial", 8)).grid(
                     row=6, column=0, columnspan=4,
                     sticky="w", padx=(0,0), pady=(2,0))

        self.traj_var = tk.BooleanVar(value=PARAMS["make_trajectory_pdf"])
        tk.Checkbutton(param,
                       text="Also generate trajectory PDF (not used in the manuscript)",
                       variable=self.traj_var).grid(
                           row=7, column=0, columnspan=4,
                           sticky="w", pady=(6, 0))

        lst = tk.LabelFrame(self, text="3. Mouse List",
                             padx=8, pady=6)
        lst.pack(fill="both", expand=True, padx=12, pady=4)
        self.tree = ttk.Treeview(lst, columns=("group","status"),
                                  show="headings", height=8)
        self.tree.heading("group",  text="Experiment Group / Mouse")
        self.tree.heading("status", text="Status")
        self.tree.column("group",  width=420)
        self.tree.column("status", width=80, anchor="center")
        self.tree.tag_configure("ok",      foreground="#1a7a3a")
        self.tree.tag_configure("missing", foreground="#c0392b")
        sb = ttk.Scrollbar(lst, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self.log_text = tk.Text(self, height=5, state="disabled",
                                 font=("Courier New", 8),
                                 bg="#1e1e1e", fg="#d4d4d4")
        self.log_text.pack(fill="x", padx=12, pady=(2,2))

        bf = tk.Frame(self, pady=6)
        bf.pack(fill="x", padx=12)
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(bf, textvariable=self.status_var,
                 fg="#555", anchor="w").pack(side="left")
        self.run_btn = tk.Button(
            bf, text="▶  Run Analysis",
            command=self._run,
            bg="#27ae60", fg="white",
            font=("Arial", 11, "bold"),
            padx=16, pady=4, relief="groove")
        self.run_btn.pack(side="right")

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
        self.update_idletasks()

    def _autoload_default_config(self):
        """Pre-load <repo>/data/edf8_demo/_Group_analysis/*food*.json if present."""
        cfg = find_default_config()
        if cfg is None:
            self._log(f"No JSON found in {CONFIG_DIR} - use Browse... to pick one.")
            return
        self._load_config(cfg)

    def _sel_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder containing group JSON")
        if not folder:
            return
        folder = Path(folder)
        p = folder / "_group_analysis.json"
        jsons = [p] if p.exists() else sorted(folder.glob("*.json"))
        if not jsons:
            messagebox.showerror("Error", "No JSON file found.")
            return
        self._load_config(jsons[0])

    def _load_config(self, json_path: Path):
        self.main_folder = json_path.parent
        self.path_var.set(str(json_path))
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.config_data = json.load(f)
        except Exception as e:
            messagebox.showerror("JSON Error", str(e))
            return
        base_dir    = self.main_folder.parent
        self.groups = resolve_mouse_paths(self.config_data, base_dir)
        self._populate_tree()
        total  = sum(len(v) for v in self.groups.values())
        exists = sum(sum(1 for m in v if m["exists"])
                     for v in self.groups.values())
        self.status_var.set(
            f"Loaded: {len(self.groups)} groups, "
            f"{total} mice ({exists} found)")

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for group, mice in self.groups.items():
            for m in mice:
                tag    = "ok" if m["exists"] else "missing"
                status = "[OK]" if m["exists"] else "x missing"
                self.tree.insert("", "end",
                                 values=(f"[{group}]  {m['name']}", status),
                                 tags=(tag,))

    def _run(self):
        if not self.groups:
            messagebox.showwarning("Warning",
                                   "Please select a folder first.")
            return

        # parse px_to_mm (blank = None = use px)
        raw = self.pxtomm_var.get().strip()
        try:
            px_to_mm = float(raw) if raw else None
        except ValueError:
            messagebox.showerror("Parameter error",
                                 "px->mm value must be a number or blank.")
            return

        smooth_sec = self.smooth_var.get()
        post_sec   = self.post_var.get()
        pre_sec    = self.pre_var.get()
        n_points   = self.npoints_var.get()
        output_dir = self.main_folder

        self.run_btn.config(state="disabled", text="Running...")
        self.status_var.set("Running...")

        def worker():
            try:
                run_analysis(self.groups, px_to_mm, smooth_sec,
                             post_sec, output_dir,
                             bodypart=self.bodypart_var.get(),
                             speed_bodypart=self.speed_bodypart_var.get(),
                             pre_sec=self.pre_var.get(),
                             n_points=self.npoints_var.get(),
                             auc_start=self.auc_start_var.get(),
                             auc_end=self.auc_end_var.get(),
                             make_trajectory_pdf=bool(self.traj_var.get()),
                             log_fn=self._log)
                self.status_var.set("[OK] Done!")
                messagebox.showinfo(
                    "Done",
                    f"Output saved to:\n{output_dir}\n\n"
                    "  food_approaching_analysis_<part>_<scale>.pdf\n"
                    "  food_curves_<part>_<scale>.csv\n"
                    "  food_auc_<part>_<scale>.csv")
            except Exception as e:
                import traceback
                self._log(f"\n[ERROR] {e}")
                self._log(traceback.format_exc())
                self.status_var.set("x Error.")
                messagebox.showerror("Error", str(e))
            finally:
                self.run_btn.config(state="normal",
                                    text="▶  Run Analysis")

        threading.Thread(target=worker, daemon=True).start()


def run_headless(config_path: Optional[Path] = None, log_fn=print):
    """Run with the manuscript parameters, no GUI."""
    cfg_path = Path(config_path) if config_path else find_default_config()
    if cfg_path is None or not cfg_path.exists():
        raise FileNotFoundError(f"Group JSON not found (looked in {CONFIG_DIR})")

    log_fn(f"Config : {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    groups = resolve_mouse_paths(config_data, cfg_path.parent.parent)

    run_analysis(
        groups,
        px_to_mm=PARAMS["px_to_mm"],
        smooth_sec=PARAMS["speed_smooth_s"],
        post_sec=PARAMS["post_window_s"],
        output_dir=cfg_path.parent,
        bodypart=PARAMS["distance_bodypart"],
        speed_bodypart=PARAMS["velocity_bodypart"],
        pre_sec=PARAMS["pre_window_s"],
        n_points=PARAMS["plot_points"],
        auc_start=PARAMS["auc_start_s"],
        auc_end=PARAMS["auc_end_s"],
        make_trajectory_pdf=PARAMS["make_trajectory_pdf"],
        log_fn=log_fn,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Command line:  python 02_food_chasing_analysis_v2.py [path/to/group.json]
        arg = sys.argv[1]
        if arg == "--gui":
            App().mainloop()
        else:
            run_headless(None if arg in ("--run", "-r") else Path(arg))
    elif PARAMS["use_gui"]:
        App().mainloop()
    else:
        run_headless()