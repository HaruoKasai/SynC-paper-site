#!/usr/bin/env python3
"""EEG/EMG, behaviour, and autonomic-signal preprocessing pipeline.

This refactor preserves the original analysis parameters and output contract.
It also provides a lightweight ``--demo`` mode for the public synthetic data
distributed with the SynC manuscript resources.
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing
import os
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.backends.backend_pdf import PdfPages
except (ImportError, AttributeError):  # pragma: no cover - optional for --make-demo
    plt = gridspec = PdfPages = None

try:
    import scipy.signal as scipy_signal
    from scipy.signal import find_peaks, stft
except ImportError:  # pragma: no cover - optional for --make-demo
    scipy_signal = find_peaks = stft = None

# These packages are only required by the raw-data pipeline. Keeping the
# imports optional lets readers run the public NPZ demo without proprietary
# Blackrock files or the laboratory's separate behavioural-analysis package.
try:
    import h5py
except ImportError:  # pragma: no cover - depends on the selected workflow
    h5py = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on the selected workflow
    pd = None

try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - depends on the selected workflow
    Parallel = delayed = None

try:
    from neo.io import BlackrockIO
except ImportError:  # pragma: no cover - depends on the selected workflow
    BlackrockIO = None

try:
    from skimage.measure import EllipseModel
except ImportError:  # pragma: no cover - depends on the selected workflow
    EllipseModel = None

try:
    import lib.DLCAnalysis as dlc_analysis
except ImportError:  # pragma: no cover - external laboratory package
    dlc_analysis = None

if plt is not None:
    plt.rcParams.update({"axes.titlesize": 14, "axes.labelsize": 12})

ROTARY_SAMPLING_RATE_HZ = 2_000
ROTARY_RADIUS_MM = 50
ROTARY_RESOLUTION = 100
IMMOBILITY_THRESHOLD_MM_S = 10
MIN_IMMOBILITY_DURATION_S = 10
STFT_EPOCH_S = 2
STFT_MAX_FREQUENCY_HZ = 100
MISSING_DATA = np.array([np.nan])
DEMO_REQUIRED_ARRAYS = {
    "schema_version",
    "synthetic",
    "sampling_rate_hz",
    "time_s",
    "eeg_uv",
    "emg_uv",
    "movement_mm_s",
}
DEMO_RANDOM_SEED = 404
DEMO_SAMPLING_RATE_HZ = 2_000
DEMO_DURATION_S = 60


def parse_args(argv=None):
    """Parse command-line options while retaining the original GUI default."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the Fig. 4c EEG/EMG preprocessing pipeline or render the "
            "lightweight public demo."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--demo",
        type=Path,
        metavar="NPZ",
        help="render the public synthetic NPZ demo instead of raw Blackrock data",
    )
    source.add_argument(
        "--data-folder",
        type=Path,
        help="process a raw-data folder without opening the folder-selection GUI",
    )
    source.add_argument(
        "--make-demo",
        type=Path,
        metavar="NPZ",
        help="regenerate the deterministic synthetic public demo archive",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Fig4c_EEG_demo_output.png"),
        help="output PNG for --demo (default: %(default)s)",
    )
    return parser.parse_args(argv)


def _require_raw_dependencies(dlc_type):
    """Report optional packages needed for the requested raw-data workflow."""
    dependencies = {
        "matplotlib": plt,
        "scipy": scipy_signal,
        "h5py": h5py,
        "pandas": pd,
        "neo": BlackrockIO,
    }
    if dlc_type == "Pupillometry":
        dependencies.update(
            {"joblib": Parallel, "scikit-image": EllipseModel}
        )
    elif dlc_type == "Openfield":
        dependencies["lib.DLCAnalysis"] = dlc_analysis

    missing = [name for name, module in dependencies.items() if module is None]
    if missing:
        raise RuntimeError(
            "Missing dependencies for the raw-data workflow: " + ", ".join(missing)
        )


def generate_demo_data(npz_path, seed=DEMO_RANDOM_SEED):
    """Generate deterministic, non-biological signals for workflow testing."""
    rng = np.random.default_rng(seed)
    sampling_rate = DEMO_SAMPLING_RATE_HZ
    time_s = np.arange(DEMO_DURATION_S * sampling_rate) / sampling_rate

    quiet = time_s < 20
    active = (time_s >= 20) & (time_s < 40)
    recovery = time_s >= 40

    eeg_uv = rng.normal(0, 5.0, time_s.size)
    eeg_uv += quiet * 32 * np.sin(2 * np.pi * 2.0 * time_s)
    eeg_uv += active * 23 * np.sin(2 * np.pi * 8.0 * time_s)
    eeg_uv += active * 7 * np.sin(2 * np.pi * 40.0 * time_s)
    eeg_uv += recovery * 18 * np.sin(2 * np.pi * 4.5 * time_s)
    eeg_uv += recovery * (
        10
        * (0.5 + 0.5 * np.sin(2 * np.pi * 0.18 * time_s))
        * np.sin(2 * np.pi * 35.0 * time_s)
    )

    emg_envelope = np.where(quiet, 2.5, np.where(active, 12.0, 5.0))
    emg_uv = rng.normal(0, 1, time_s.size) * emg_envelope
    emg_uv += active * 3.0 * np.sin(2 * np.pi * 70.0 * time_s)

    smoothing_samples = sampling_rate // 4
    kernel = np.ones(smoothing_samples) / smoothing_samples
    movement_noise = np.convolve(rng.normal(0, 1, time_s.size), kernel, mode="same")
    movement_mm_s = np.where(quiet, 1.0, np.where(active, 65.0, 12.0))
    movement_mm_s = np.maximum(0, movement_mm_s + 25 * movement_noise)

    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        schema_version=np.array(1, dtype=np.int16),
        synthetic=np.array(True),
        random_seed=np.array(seed, dtype=np.int32),
        figure=np.array("Fig. 4c"),
        description=np.array(
            "Synthetic EEG, EMG, and movement signals for workflow testing only; "
            "not manuscript source data."
        ),
        sampling_rate_hz=np.array(sampling_rate, dtype=np.float32),
        time_s=time_s.astype(np.float32),
        eeg_uv=eeg_uv.astype(np.float32),
        emg_uv=emg_uv.astype(np.float32),
        movement_mm_s=movement_mm_s.astype(np.float32),
    )
    return npz_path


def load_demo_data(npz_path):
    """Load and validate the compact, synthetic public demo archive."""
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Demo data not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(DEMO_REQUIRED_ARRAYS - set(archive.files))
        if missing:
            raise ValueError("Demo archive is missing: " + ", ".join(missing))
        demo = {key: np.asarray(archive[key]) for key in DEMO_REQUIRED_ARRAYS}

    if int(np.ravel(demo["schema_version"])[0]) != 1:
        raise ValueError("Unsupported demo schema_version")
    if not bool(np.ravel(demo["synthetic"])[0]):
        raise ValueError("Public demo must be explicitly marked as synthetic")
    sampling_rate = float(np.ravel(demo["sampling_rate_hz"])[0])
    time_s = np.ravel(demo["time_s"]).astype(float)
    signals = {
        name: np.ravel(demo[name]).astype(float)
        for name in ("eeg_uv", "emg_uv", "movement_mm_s")
    }

    if sampling_rate <= 2 * STFT_MAX_FREQUENCY_HZ:
        raise ValueError("sampling_rate_hz must exceed 200 Hz")
    if time_s.size < int(STFT_EPOCH_S * sampling_rate * 2):
        raise ValueError("Demo must contain at least two STFT epochs")
    if any(values.size != time_s.size for values in signals.values()):
        raise ValueError("All demo signals must have the same length as time_s")
    if not np.all(np.isfinite(time_s)) or any(
        not np.all(np.isfinite(values)) for values in signals.values()
    ):
        raise ValueError("Demo arrays must contain only finite values")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")

    measured_rate = 1.0 / np.median(np.diff(time_s))
    if not np.isclose(measured_rate, sampling_rate, rtol=1e-3):
        raise ValueError(
            "sampling_rate_hz is inconsistent with the time_s spacing"
        )
    return sampling_rate, time_s, signals


def render_demo(npz_path, output_path):
    """Render a compact EEG/EMG/spectrogram demonstration figure."""
    missing = [
        name
        for name, module in {"matplotlib": plt, "scipy": stft}.items()
        if module is None
    ]
    if missing:
        raise RuntimeError(
            "Missing dependencies for --demo: " + ", ".join(missing)
        )
    sampling_rate, time_s, signals = load_demo_data(npz_path)
    eeg = signals["eeg_uv"]
    emg = signals["emg_uv"]
    movement = signals["movement_mm_s"]

    nperseg = int(STFT_EPOCH_S * sampling_rate)
    frequencies, stft_time, coefficients = stft(
        eeg,
        fs=sampling_rate,
        nperseg=nperseg,
        noverlap=nperseg // 2,
    )
    frequency_mask = frequencies <= STFT_MAX_FREQUENCY_HZ
    power_db = 10 * np.log10(np.abs(coefficients[frequency_mask]) ** 2 + 1e-10)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2, 1, 1]},
        constrained_layout=True,
    )
    axes[0].plot(time_s, eeg, color="#17223b", linewidth=0.45)
    axes[0].set_ylabel("EEG (uV)")
    axes[0].set_title("Fig. 4c EEG analysis: synthetic workflow demo")

    mesh = axes[1].pcolormesh(
        stft_time + time_s[0],
        frequencies[frequency_mask],
        power_db,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_ylim(0, STFT_MAX_FREQUENCY_HZ)
    colorbar = fig.colorbar(mesh, ax=axes[1], pad=0.01)
    colorbar.set_label("Power (dB)")

    axes[2].plot(time_s, emg, color="#147d64", linewidth=0.45)
    axes[2].set_ylabel("EMG (uV)")
    axes[3].plot(time_s, movement, color="#bd4b4b", linewidth=0.7)
    axes[3].set_ylabel("Velocity\n(mm/s)")
    axes[3].set_xlabel("Time (s)")

    for axis in axes:
        axis.grid(False)
        axis.margins(x=0)
    axes[-1].set_xlim(time_s[0], time_s[-1])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _hdf_data(value):
    """Represent unavailable arrays exactly as in the original HDF5 output."""
    return MISSING_DATA if value is None else value


def choose_data_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(
        title="Select the 'data' directory", initialdir=str(Path.cwd())
    )
    root.destroy()
    return folder_path


def decode_rotary_encoder(
    channel_a,
    channel_b,
    channel_z,
    experiment_duration,
    time_bin=1,
    radius=ROTARY_RADIUS_MM,
    resolution=ROTARY_RESOLUTION,
):
    """Convert quadrature-encoder transitions to binned linear velocity."""
    del channel_z  # Kept in the interface because the source data include it.

    sampling_rate = ROTARY_SAMPLING_RATE_HZ
    num_rows = int(experiment_duration * sampling_rate)
    encoder_state = np.zeros((num_rows, 2), dtype=int)

    channel_a_indices = (np.asarray(channel_a) * sampling_rate).astype(int)
    channel_b_indices = (np.asarray(channel_b) * sampling_rate).astype(int)

    encoder_state[channel_a_indices, 0] = 1
    encoder_state[channel_b_indices, 1] = 1

    def toggle_pattern(array, toggle_points):
        toggle_state = np.zeros_like(array)
        toggle_state[toggle_points] = 1
        cumulative_toggles = np.cumsum(toggle_state)
        return cumulative_toggles % 2

    encoder_state[:, 0] = toggle_pattern(encoder_state[:, 0], channel_a_indices)
    encoder_state[:, 1] = toggle_pattern(encoder_state[:, 1], channel_b_indices)

    channel_a_changes = (np.diff(encoder_state[:, 0], prepend=0) == 1).astype(int)
    channel_b_values = encoder_state[:, 1]
    bin_size = int(time_bin * sampling_rate)
    transition_indices = np.where(channel_a_changes == 1)[0]
    direction_at_transitions = channel_b_values[transition_indices]

    bin_edges = np.arange(0, num_rows + 1, bin_size)
    bin_indices = np.digitize(transition_indices, bin_edges) - 1

    bin_counts = np.bincount(
        bin_indices,
        weights=1 - 2 * direction_at_transitions,
        minlength=len(bin_edges) - 1,
    )
    time_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / sampling_rate

    circumference = 2 * np.pi * radius
    mm_per_resolution = circumference / resolution

    velocities = bin_counts * mm_per_resolution / time_bin
    if velocities.sum() <= 0:
        velocities = -velocities
    return velocities, time_centers


def plot_binned_timeseries(data, time_centers, ylabel, ax):
    ax.grid(False)
    ax.plot(time_centers, data, lw=1)
    xticks = np.arange(0, time_centers[-1], 30)
    xtick_labels = np.arange(0, time_centers[-1], 30)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-400, 400)
    bin = time_centers[1] - time_centers[0]
    ax.set_xlim(time_centers[0] - bin / 2, time_centers[-1] + bin / 2)
    ax.margins(x=0)
    return ax


def _channel_pairs(channel_map):
    return [
        [channel - 1 if channel is not None else None for channel in pair]
        for pair in channel_map.values()
    ]


def _referenced_signals(raw_signal, channel_map, sample_count):
    signals = []
    for recording_channel, reference_channel in _channel_pairs(channel_map):
        recorded = raw_signal[:, recording_channel].magnitude.flatten()
        reference = (
            raw_signal[:, reference_channel].magnitude.flatten()
            if reference_channel is not None
            else np.zeros_like(recorded)
        )
        signals.append(recorded - reference)
    return np.asarray(signals)[:, :sample_count]


def load_blackrock_signals(
    file_path,
    eeg_channels,
    emg_channels,
    analog_channels,
    time_range,
):
    """Load and reference Blackrock EEG/EMG channels and auxiliary signals."""
    reader = BlackrockIO(filename=file_path)
    block = reader.read_block()
    raw_signals = [seg.analogsignals[0] for seg in block.segments]
    primary_signal = raw_signals[0]
    sampling_rate = int(primary_signal[:, 0].sampling_rate.magnitude)
    experiment_duration = (time_range[1] - time_range[0]) * 60
    sample_count = int(experiment_duration * sampling_rate)

    eeg_signals = _referenced_signals(primary_signal, eeg_channels, sample_count)
    emg_signals = _referenced_signals(primary_signal, emg_channels, sample_count)

    start_time = raw_signals[0].t_start.magnitude
    digital_input_data = None
    channel_times = {}
    breathe = None
    tem = None

    try:
        analog_input_signals = [seg.analogsignals[1] for seg in block.segments]
        breathe = analog_input_signals[0][
            :, analog_channels["Breathing"] - 1
        ].magnitude.flatten()
        tem = analog_input_signals[0][
            :, analog_channels["Temperature"] - 1
        ].magnitude.flatten()
        breathe = breathe[:sample_count]
        tem = tem[:sample_count]

    except Exception:
        pass

    for segment in block.segments:
        for event_array in segment.events:
            if event_array.name == "digital_input_port":
                digital_input_data = event_array
    try:
        times = digital_input_data.times.rescale("s").magnitude
        labels = digital_input_data.labels.astype(int)
        num_channels = 16
        bitwise_states = np.array(
            [[(label >> i) & 1 for i in range(num_channels)] for label in labels]
        )
        channel_times = {f"channel_{i}": [] for i in range(num_channels)}
        for i in range(num_channels):
            changes = np.where(np.diff(bitwise_states[:, i]) != 0)[0] + 1
            channel_times[f"channel_{i}"] = times[changes] - start_time
        channel_times = {
            key: values[values <= experiment_duration]
            for key, values in channel_times.items()
        }
    except Exception:
        pass
    return (
        emg_signals,
        eeg_signals,
        sampling_rate,
        breathe,
        tem,
        channel_times,
        digital_input_data,
    )


def sum_band_power(
    freqs, power_spectrum, lower_bound, upper_bound, normalize=False, to_db=False
):

    band_mask = (freqs >= lower_bound) & (freqs < upper_bound)
    band_power = np.sum(power_spectrum[band_mask])

    if normalize:
        band_power = band_power / np.sum(power_spectrum) * 100
    if to_db:
        band_power = 10 * np.log10(band_power + 1e-10)
    return band_power


def plot_band_power_timeseries(
    eeg, analog_tp, sampling_rate, time_bin, ax_list, lw, legend=True, dB=True
):
    columns = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-12 Hz)",
        "Beta (12-30 Hz)",
        "Gamma (30-80 Hz)",
        "High gamma (60-100Hz)",
        "Low gamma (30-60Hz)",
    ]
    powers = {col: [] for col in columns}
    time_bins = int(len(eeg) / (sampling_rate * time_bin))

    for t in range(time_bins):
        start = t * sampling_rate * time_bin
        end = start + sampling_rate * time_bin
        segment = eeg[start:end]
        freqs, power_spectrum = fft_power_spectrum(segment, sampling_rate)

        powers["Delta (0.5-4 Hz)"].append(
            sum_band_power(freqs, power_spectrum, 0.5, 4, to_db=False)
        )
        powers["Theta (4-8 Hz)"].append(
            sum_band_power(freqs, power_spectrum, 4, 8, to_db=False)
        )
        powers["Alpha (8-12 Hz)"].append(
            sum_band_power(freqs, power_spectrum, 8, 12, to_db=False)
        )
        powers["Beta (12-30 Hz)"].append(
            sum_band_power(freqs, power_spectrum, 12, 30, to_db=False)
        )
        powers["Gamma (30-80 Hz)"].append(
            sum_band_power(freqs, power_spectrum, 30, 80, to_db=False)
        )
        powers["High gamma (60-100Hz)"].append(
            sum_band_power(freqs, power_spectrum, 60, 100, to_db=False)
        )
        powers["Low gamma (30-60Hz)"].append(
            sum_band_power(freqs, power_spectrum, 30, 60, to_db=False)
        )

    df_linear = pd.DataFrame(powers)

    if dB:
        df = 10 * np.log10(df_linear + 1e-10)
    else:
        df = df_linear
    for c, col in enumerate(columns[:5]):
        ax = ax_list[c]
        if ax is not None:
            ax.plot(
                analog_tp[0] + time_bin / 2 + np.arange(len(df)) * time_bin,
                df[col],
                label=col,
                lw=lw,
            )
            ax.set_title("Band Power Time Series")
            ax.set_ylabel("Power (dB)")
            ax.set_ylim(55, 85)
            if legend:
                ax.legend(loc="upper right")

    return powers


def fft_power_spectrum(eeg, sampling_rate):
    n = len(eeg)
    eeg -= np.mean(eeg)
    freqs = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_vals = np.fft.fft(eeg)
    power_spectrum = np.abs(fft_vals) ** 2 / n
    return freqs[: n // 2], power_spectrum[: n // 2]


def plot_heatmap(ax, t, f, power, title, ylabel, freq_limit, cmap, power_range):
    ax.pcolormesh(
        t,
        f,
        power,
        shading="gouraud",
        rasterized=True,
        cmap=cmap,
        vmin=power_range[0],
        vmax=power_range[1],
    )
    ax.set_title(title)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, freq_limit)


def adaptive_threshold(signal, window_size, sigma_factor):
    if len(signal) < window_size:
        return np.full_like(signal, np.mean(signal))
    rolling_std = (
        pd.Series(signal).rolling(window=window_size, min_periods=1).std().to_numpy()
    )
    return np.mean(signal) + sigma_factor * rolling_std


def compute_breathing_frequency(breathe, sampling_rate, window_size=2):
    breathe = breathe - np.mean(breathe)

    nyquist = 0.5 * sampling_rate
    b, a = scipy_signal.butter(2, [1 / nyquist, 10 / nyquist], btype="band")
    filtered_signal = scipy_signal.filtfilt(b, a, breathe)

    threshold_values = adaptive_threshold(
        filtered_signal, int(sampling_rate * 2), sigma_factor=0.25
    )

    min_peak_distance = int(sampling_rate * 0.1)
    peaks, _ = scipy_signal.find_peaks(filtered_signal, distance=min_peak_distance)

    valid_peaks = peaks[filtered_signal[peaks] > threshold_values[peaks]]
    if len(valid_peaks) < 2:
        return np.array([]), np.array([])

    peak_intervals = np.diff(valid_peaks) / sampling_rate
    breathing_rates = 60 / peak_intervals
    timestamps = valid_peaks[1:] / sampling_rate

    df = pd.DataFrame({"timestamp": timestamps, "breathing_rate": breathing_rates})
    df["window"] = (df["timestamp"] // window_size).astype(int)
    result = df.groupby("window")["breathing_rate"].mean().reset_index()
    all_windows = pd.DataFrame(
        {"window": range(df["window"].min(), df["window"].max() + 1)}
    )
    result = all_windows.merge(result, on="window", how="left").fillna(0)
    result["timestamp"] = result["window"] * window_size

    return result["timestamp"].to_numpy(), result["breathing_rate"].to_numpy()


def calculate_heart_rate(ecg_data, sampling_rate, bin_size):
    min_distance = int(0.05 * sampling_rate)
    sd = np.std(ecg_data)
    r_peaks, _ = find_peaks(ecg_data, distance=min_distance, prominence=sd * 3)

    r_peak_times = r_peaks / sampling_rate
    rr_intervals = np.diff(r_peak_times)
    rr_times = r_peak_times[:-1] + rr_intervals / 2
    hr_values = 60.0 / rr_intervals

    total_duration = len(ecg_data) / sampling_rate
    bin_edges = np.arange(0, total_duration + bin_size, bin_size)
    bin_indices = np.digitize(rr_times, bin_edges) - 1

    num_bins = len(bin_edges) - 1
    sum_hr = np.bincount(bin_indices, weights=hr_values, minlength=num_bins)
    count_hr = np.bincount(bin_indices, minlength=num_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        hr_timeseries = sum_hr / count_hr
        hr_timeseries[count_hr == 0] = np.nan

    return hr_timeseries, bin_edges[:-1] + bin_size / 2


def extract_params(json_dir):
    dlc_json = os.path.join(json_dir, "_analysis_param.json")
    with open(dlc_json, "r", encoding="utf-8") as file:
        data = json.load(file)
        dlc_type = data["DLC"].get("type", None)
        dlc_dir = data["DLC"].get("dir", None)
        EEG_ch_dict = data["EEG"]
        EMG_ch_dict = data["EMG"]
        analog_dict = data["Analog"]
        contime = data["Time"]["Continuous"]
    return dlc_type, dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, contime


def pupillometry(dir):
    csv = glob.glob(os.path.join(dir, "*filtered.csv"))[0]
    df = pd.read_csv(csv, low_memory=False)

    num_points = 8
    x_indices = [1 + i * 3 for i in range(num_points)]
    y_indices = [x + 1 for x in x_indices]
    l_indices = [y + 1 for y in y_indices]

    time_steps = len(df) - 2
    coords_x = df.iloc[2:, x_indices].astype(float).values
    coords_y = df.iloc[2:, y_indices].astype(float).values
    likelihood = np.mean(df.iloc[2:, l_indices].astype(float).values, axis=1)

    def fit_ellipse(t):
        ellipse = EllipseModel()
        if ellipse.estimate(np.column_stack([coords_x[t], coords_y[t]])):
            _, _, a, b, _ = ellipse.params
            return np.pi * (a / 2) * (b / 2)
        return np.nan

    max_jobs = min(multiprocessing.cpu_count(), 63)
    areas = Parallel(n_jobs=max_jobs, backend="threading")(
        delayed(fit_ellipse)(t) for t in range(time_steps)
    )
    areas = np.array(areas)

    df = pd.read_csv(os.path.join(dir, "_timestamp.csv"))
    df["time"] = pd.to_datetime(df["time"])
    real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)
    real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)

    return areas, real_frame_time, likelihood


def analyze_open_field(
    index, exp_dir, dlc_dir, event_df, start_time, velocity_boundary
):
    try:
        arena_mm_per_pix = 0.6
        dlc_exp_dir = glob.glob(os.path.join(dlc_dir, "day*"))[index]
        dlc_h5_path = os.path.join(dlc_exp_dir, "dlc_raw.h5")
        param_ind = os.path.join(dlc_exp_dir, "param_individual.json")
        df = pd.read_hdf(dlc_h5_path, key="dlc_data")
        dlc_output_dir = os.path.join(exp_dir, "_DLC_analysis")
        if not os.path.exists(dlc_output_dir):
            os.makedirs(dlc_output_dir)
        df.to_csv(os.path.join(dlc_output_dir, "dlc_data.csv"))

        real_frame_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / (len(df) - 1)
        real_frame_time = round(pd.to_timedelta(real_frame_time).total_seconds(), 5)
        velocity, cumulative_distance, frames_extracted_by_velocity, likelihood = (
            dlc_analysis.time_series_velocity(
                df, real_frame_time, arena_mm_per_pix, "centroid", velocity_boundary
            )
        )

        for v in range(len(frames_extracted_by_velocity)):
            event_name = "Centroid~" + str(velocity_boundary[v]) + "mm_per_s"
            event_df = dlc_analysis.frame_to_sec(
                frames_extracted_by_velocity[v],
                real_frame_time,
                event_df,
                event_name,
                start_time,
                tolerable_frame_drop=2,
                min_duration=10,
            )

        arena_coordinate = dlc_analysis.get_roi_coordinate(
            "arena_box", param_ind=param_ind
        )
        distance_to_center, _, _ = dlc_analysis.time_series_distance_to_object(
            df,
            arena_coordinate,
            real_frame_time,
            arena_mm_per_pix,
            body_part="centroid",
            distance_to_boundary_mm=100,
        )

        event_df.to_csv(os.path.join(dlc_output_dir, "event.csv"))

        return (
            real_frame_time,
            velocity,
            cumulative_distance,
            distance_to_center,
            event_df,
            likelihood,
        )
    except Exception:
        print("Something's wrong in processing openfield")
        return None, None, None, None, None, None


def plot_timeseries(
    tp, data, window_size, ax, color, lw, title, ylabel, ylim, label=None, alpha=1
):
    """
    tp, data: numpy
    window_size: int
    """
    valid_len = (len(data) // window_size) * window_size
    ave_data = data[:valid_len].reshape(-1, window_size).mean(axis=1)
    ave_tp = tp[:valid_len].reshape(-1, window_size).mean(axis=1)
    ax.set_ylim(ylim)
    ax.plot(ave_tp, ave_data, lw=lw, color=color, label=label, alpha=alpha)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(loc="upper right")


def binning(data, window_size):
    valid_len = (len(data) // window_size) * window_size
    ave_data = data[:valid_len].reshape(-1, window_size).mean(axis=1)
    return ave_data


def save_analysis_pdf(
    dlc_type,
    analog_tp,
    eeg,
    t_stft,
    f_stft,
    linear_power,
    dB_power,
    emg,
    ecg,
    sampling_rate,
    heartrate,
    hr_tp,
    breathe,
    tem,
    breathing_rate,
    pupil_size,
    pupil_tp,
    table_velocities,
    table_tp,
    OF_tp,
    velocity,
    cumulative_distance,
    distance,
    velocity_boundary,
    likelihood,
    event_df,
    output_dir,
    pdf_name,
    figsize,
):

    pdf_path = os.path.join(output_dir, pdf_name)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(10, 1, height_ratios=[0.4, 1, 1, 1, 1, 0.4, 1, 1, 1, 1])
    plt.subplots_adjust(hspace=0.5)

    total_time = len(eeg) / sampling_rate

    axes = [fig.add_subplot(gs[i, 0]) for i in range(10)]
    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = axes

    plot_timeseries(
        analog_tp,
        eeg,
        window_size=1,
        ax=ax0,
        color="#555555",
        lw=0.1,
        title=pdf_name,
        ylabel="Amplitude (µV)",
        ylim=(-500, 500),
    )

    plot_heatmap(
        ax1,
        t_stft,
        f_stft,
        linear_power,
        "STFT Linear Power",
        "Frequency (Hz)",
        100,
        "rainbow",
        [-10, 100],
    )

    plot_heatmap(
        ax2,
        t_stft,
        f_stft,
        dB_power,
        "STFT dB Power",
        "Frequency (Hz)",
        80,
        "rainbow",
        [-10, 33],
    )

    plot_heatmap(
        ax3,
        t_stft,
        f_stft,
        dB_power,
        "STFT dB Power (0-20 Hz)",
        "Frequency (Hz)",
        20,
        "rainbow",
        [-10, 33],
    )

    plot_band_power_timeseries(
        eeg,
        analog_tp,
        sampling_rate,
        time_bin=2,
        ax_list=[ax4, ax4, ax4, ax4, ax4],
        lw=0.4,
    )

    plot_timeseries(
        analog_tp,
        emg,
        window_size=1,
        ax=ax5,
        color="#555555",
        lw=0.1,
        title="Raw EMG Signal",
        ylabel="Amplitude (µV)",
        ylim=(-1000, 1000),
    )

    if dlc_type == "Pupillometry":
        plot_timeseries(
            analog_tp,
            breathe,
            window_size=4,
            ax=ax7,
            color="k",
            lw=0.1,
            title="Breathe",
            ylabel="Amplitude (µV)",
            ylim=(1800, 2400),
        )

        ax7_2 = ax7.twinx()
        window_size = total_time / len(breathing_rate)
        time = np.arange(window_size / 2, total_time, window_size) + analog_tp[0]
        ax7_2.plot(
            time,
            breathing_rate,
            label="Breathing_rate (BPM)",
            color="red",
            linewidth=0.5,
        )
        ax7_2.set_ylabel("Breathing_rate (BPM)", color="red")
        ax7_2.tick_params(axis="y", labelcolor="red")
        ax7_2.set_ylim([40, 420])

        plot_timeseries(
            pupil_tp,
            pupil_size,
            window_size=4,
            ax=ax8,
            color="green",
            lw=0.2,
            title="Pupil size",
            ylabel="(a.u.)",
            ylim=(200, 3500),
        )
        plot_timeseries(
            pupil_tp,
            likelihood,
            window_size=4,
            ax=ax8.twinx(),
            color="gray",
            lw=0.25,
            title="Pupil size",
            ylabel="likelihood",
            ylim=(0, 1),
        )

        if table_velocities is not None and table_tp is not None:
            plot_binned_timeseries(
                table_velocities, table_tp, ylabel="Velocity (mm/s)", ax=ax6
            )
        else:
            ax6.set_title("No Rotary Encoder Data Available")
            ax6.set_ylabel("Velocity (mm/s)")

    elif dlc_type == "Openfield":
        plot_timeseries(
            OF_tp,
            velocity,
            window_size=4,
            ax=ax6,
            color="blue",
            lw=0.25,
            title="Velocity (Centroid)",
            ylabel="Velocity (mm/s)",
            ylim=(0, 200),
        )
        plot_timeseries(
            OF_tp,
            velocity,
            window_size=20 * 60,
            ax=ax6,
            color="olive",
            lw=0.5,
            title="Velocity (Centroid)",
            ylabel="Velocity (mm/s)",
            ylim=(0, 200),
        )
        plot_timeseries(
            OF_tp,
            likelihood,
            window_size=20,
            ax=ax6.twinx(),
            color="gray",
            lw=0.25,
            title="Velocity (Centroid)",
            ylabel="likelihood",
            ylim=(0, 1),
        )

        plot_timeseries(
            OF_tp,
            distance,
            window_size=1,
            ax=ax7,
            color="green",
            lw=0.5,
            title="Distance to Center (Centroid)",
            ylabel="(mm)",
            ylim=(0, 300),
        )

        for y in velocity_boundary:
            ax6.axhline(y=y, color="gray", linewidth=0.2)

    if ecg is not None:
        plot_timeseries(
            hr_tp,
            heartrate,
            window_size=4,
            ax=ax9,
            color="#1f77b4",
            lw=0.5,
            title="ECG",
            ylabel="BPM",
            ylim=(200, 1000),
        )

    added_labels = set()
    event_list = event_df["event_name"].unique().tolist()
    for e, event in enumerate(event_list):
        df = event_df[event_df["event_name"] == event]
        df = df.reset_index(drop=True)
        for ep in range(len(df)):
            label = event if event not in added_labels else ""
            ax6.axvspan(
                df.loc[ep, "start_time"],
                df.loc[ep, "end_time"],
                color=plt.get_cmap("tab10")(e),
                alpha=0.3,
                linewidth=0,
                label=label,
            )
            added_labels.add(event)
    ax6.legend(fontsize=8)

    for ax in axes:
        xticks = list(range(int(analog_tp[0]), int(analog_tp[-1]) + 1, 1 * 60))
        ax.set_xticks(xticks)
        xtick_labels = [
            str(x) + "(" + str(int(x / 60)) + ")" if i % 5 == 0 else ""
            for i, x in enumerate(xticks)
        ]
        ax.set_xticklabels(xtick_labels)
        ax.set_xlabel("Time (sec)")
        ax.set_xlim(min(xticks), max(xticks))
        ax.margins(x=0)

    plt.tight_layout()

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, dpi=300)
    plt.close(fig)

    print(f"Saved to  {pdf_path}")
    return pdf_path


def add_immobility_events(
    table_tp, table_velocities, vmax, min_dur_sec, tolerable_drop_sec, event_df
):
    valid_indices = np.where(np.abs(table_velocities) < 10)[0]
    event_name = "turntable_velocity_~" + str(vmax) + "mm_per_s"
    if len(valid_indices) > 0:
        start, end = valid_indices[0], valid_indices[0]
        tp_width = table_tp[1] - table_tp[0]
        for i in range(1, len(valid_indices)):
            if (
                valid_indices[i]
                <= valid_indices[i - 1] + 1 + tolerable_drop_sec / tp_width
            ):
                end = valid_indices[i]
            else:
                if (end + 1 - start) * tp_width > min_dur_sec and start * tp_width >= 0:
                    event_df.loc[len(event_df)] = [
                        table_tp[start],
                        table_tp[end],
                        event_name,
                    ]
                start, end = valid_indices[i], valid_indices[i]
        if (end + 1 - start) * tp_width > min_dur_sec and start * tp_width > 0:
            event_df.loc[len(event_df)] = [
                table_tp[start],
                table_tp[min(end, len(table_tp) - 1)],
                event_name,
            ]

    return event_df


def process_recording(
    file_path,
    index,
    dlc_type,
    dlc_dir,
    EEG_ch_dict,
    EMG_ch_dict,
    analog_dict,
    cont_time,
    result_list,
    lock,
):
    """Process one .ns3/.ns2 recording; safe to call in a worker process."""
    root_dir = os.path.dirname(file_path)
    print(root_dir)
    output_dir = os.path.join(root_dir, "results")
    os.makedirs(output_dir, exist_ok=True)
    exp_dur = (cont_time[1] - cont_time[0]) * 60

    emg, eeg, sampling_rate, breathe, tem, channel_times, digital_input_data = (
        load_blackrock_signals(
            file_path, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time
        )
    )
    analog_tp = cont_time[0] * 60 + np.arange(len(emg[0])) / sampling_rate

    epoch_length = STFT_EPOCH_S
    nperseg = int(epoch_length * sampling_rate)
    f_stft, t_stft, Zxx = stft(
        eeg, fs=sampling_rate, nperseg=nperseg, noverlap=nperseg // 2
    )

    t_stft = t_stft[:-1]

    Zxx = Zxx[:, :, :-1]

    cutoff = np.argmax(f_stft > STFT_MAX_FREQUENCY_HZ)
    f_stft = f_stft[:cutoff]
    Zxx = Zxx[:, :cutoff, :]
    t_stft += cont_time[0] * 60
    linear_power = np.abs(Zxx) ** 2
    dB_power = 10 * np.log10(linear_power + 1e-10)

    breathing_rate = pupil_size = pupil_tp = None
    table_velocities = table_tp = None
    OF_tp = velocity = cumulative_distance = None
    distance_to_center = velocity_boundary = None

    event_df = pd.DataFrame(columns=["start_time", "end_time", "event_name"])
    digital_timer = None

    if digital_input_data is not None:
        Timer_channel, A_channel, B_channel, Z_channel = 0, 4, 6, 8
        digital_timer, A, B, Z = (
            np.asarray(channel_times.get(f"channel_{Timer_channel}", []), dtype=float),
            np.asarray(channel_times.get(f"channel_{A_channel}", []), dtype=float),
            np.asarray(channel_times.get(f"channel_{B_channel}", []), dtype=float),
            np.asarray(channel_times.get(f"channel_{Z_channel}", []), dtype=float),
        )

        table_velocities, table_tp = decode_rotary_encoder(
            A, B, Z, experiment_duration=len(eeg[0]) / sampling_rate
        )
        table_tp += cont_time[0] * 60

        event_df = add_immobility_events(
            table_tp,
            table_velocities,
            IMMOBILITY_THRESHOLD_MM_S,
            MIN_IMMOBILITY_DURATION_S,
            0,
            event_df,
        )

    if dlc_type == "Pupillometry":
        pupil_size, video_frame_time, likelihood = pupillometry(
            os.path.join(root_dir, "raw_video")
        )
        pupil_size = pupil_size[: int(exp_dur / video_frame_time)]
        likelihood = likelihood[: int(exp_dur / video_frame_time)]
        pupil_tp = cont_time[0] * 60 + np.arange(len(pupil_size)) * video_frame_time

        _, breathing_rate = compute_breathing_frequency(breathe, sampling_rate)

    elif dlc_type == "Openfield":
        velocity_boundary = [IMMOBILITY_THRESHOLD_MM_S]
        (
            video_frame_time,
            velocity,
            cumulative_distance,
            distance_to_center,
            event_df,
            likelihood,
        ) = analyze_open_field(
            index,
            os.path.dirname(file_path),
            dlc_dir,
            event_df,
            cont_time[0] * 60,
            velocity_boundary,
        )
        velocity = velocity[: int(exp_dur / video_frame_time)]
        likelihood = likelihood[: int(exp_dur / video_frame_time)]
        distance_to_center = distance_to_center[: int(exp_dur / video_frame_time)]
        OF_tp = cont_time[0] * 60 + np.arange(len(velocity)) * video_frame_time

    ecg = emg[1] if len(emg) > 1 else None
    heartrate = None
    hr_tp = None
    if ecg is not None:
        bin_size = 0.25
        heartrate, hr_tp = calculate_heart_rate(ecg, sampling_rate, bin_size)
        hr_tp += cont_time[0] * 60

    with lock:
        result_list.append(
            (
                index,
                analog_tp,
                eeg,
                emg,
                sampling_rate,
                breathe,
                tem,
                breathing_rate,
                pupil_size,
                pupil_tp,
                table_velocities,
                table_tp,
                OF_tp,
                velocity,
                cumulative_distance,
                distance_to_center,
                t_stft,
                f_stft,
                linear_power,
                dB_power,
                event_df,
                velocity_boundary,
                likelihood,
                heartrate,
                hr_tp,
                digital_timer,
            )
        )

    mouse_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    if eeg is not None and emg is not None:
        for i, electrode in enumerate(EEG_ch_dict):
            pdf_name = mouse_name + "_____" + electrode + ".pdf"
            save_analysis_pdf(
                dlc_type,
                analog_tp,
                eeg[i],
                t_stft,
                f_stft,
                linear_power[i],
                dB_power[i],
                emg[0],
                ecg,
                sampling_rate,
                heartrate,
                hr_tp,
                breathe,
                tem,
                breathing_rate,
                pupil_size,
                pupil_tp,
                table_velocities,
                table_tp,
                OF_tp,
                velocity,
                cumulative_distance,
                distance_to_center,
                velocity_boundary,
                likelihood,
                event_df,
                output_dir=output_dir,
                pdf_name=pdf_name,
                figsize=(15, 25),
            )


def process_data_folder(data_folder):
    dlc_type, dlc_dir, EEG_ch_dict, EMG_ch_dict, analog_dict, cont_time = (
        extract_params(data_folder)
    )
    _require_raw_dependencies(dlc_type)

    output_dir = os.path.join(data_folder, "_Combined")
    os.makedirs(output_dir, exist_ok=True)

    file_list = []
    for root_dir, _, files in os.walk(data_folder):
        if not os.path.basename(root_dir).startswith("_"):
            file_list.extend(
                [
                    os.path.join(root_dir, f)
                    for f in files
                    if f.endswith((".ns3", ".ns2"))
                ]
            )
    print(file_list)

    if not file_list:
        print("No .ns3 or .ns2 files found.")
        return

    manager = multiprocessing.Manager()
    result_list = manager.list()
    lock = manager.Lock()

    num_workers = min(multiprocessing.cpu_count(), len(file_list))
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(
            process_recording,
            zip(
                file_list,
                range(len(file_list)),
                [dlc_type] * len(file_list),
                [dlc_dir] * len(file_list),
                [EEG_ch_dict] * len(file_list),
                [EMG_ch_dict] * len(file_list),
                [analog_dict] * len(file_list),
                [cont_time] * len(file_list),
                [result_list] * len(file_list),
                [lock] * len(file_list),
            ),
        )

    sorted_results = sorted(result_list, key=lambda x: x[0])

    all_analog_tp = np.concatenate([res[1] for res in sorted_results], axis=0)
    all_eeg = np.concatenate([res[2] for res in sorted_results], axis=1)
    all_emg = np.concatenate([res[3] for res in sorted_results], axis=1)
    sampling_rate = sorted_results[0][4]
    all_breathe = (
        None
        if any(res[5] is None for res in sorted_results)
        else np.concatenate([res[5] for res in sorted_results], axis=0)
    )
    all_tem = (
        None
        if any(res[6] is None for res in sorted_results)
        else np.concatenate([res[6] for res in sorted_results], axis=0)
    )
    all_b_rate = (
        None
        if any(res[7] is None for res in sorted_results)
        else np.concatenate([res[7] for res in sorted_results], axis=0)
    )
    all_pupil = (
        None
        if any(res[8] is None for res in sorted_results)
        else np.concatenate([res[8] for res in sorted_results], axis=0)
    )
    all_pupil_tp = (
        None
        if any(res[9] is None for res in sorted_results)
        else np.concatenate([res[9] for res in sorted_results], axis=0)
    )
    all_table_v = (
        None
        if any(res[10] is None for res in sorted_results)
        else np.concatenate([res[10] for res in sorted_results], axis=0)
    )
    all_table_tp = (
        None
        if any(res[11] is None for res in sorted_results)
        else np.concatenate([res[11] for res in sorted_results], axis=0)
    )
    all_OF_tp = (
        None
        if any(res[12] is None for res in sorted_results)
        else np.concatenate([res[12] for res in sorted_results], axis=0)
    )
    all_v = (
        None
        if any(res[13] is None for res in sorted_results)
        else np.concatenate([res[13] for res in sorted_results], axis=0)
    )
    all_cum_d = (
        None
        if any(res[14] is None for res in sorted_results)
        else np.concatenate([res[14] for res in sorted_results], axis=0)
    )
    all_distance = (
        None
        if any(res[15] is None for res in sorted_results)
        else np.concatenate([res[15] for res in sorted_results], axis=0)
    )
    all_t_stft = (
        None
        if any(res[16] is None for res in sorted_results)
        else np.concatenate([res[16] for res in sorted_results], axis=0)
    )
    f_stft = sorted_results[0][17]
    all_linear_power = np.concatenate([res[18] for res in sorted_results], axis=2)
    all_dB_power = np.concatenate([res[19] for res in sorted_results], axis=2)
    all_event_df = (
        None
        if any(res[20] is None for res in sorted_results)
        else pd.concat([res[20] for res in sorted_results], axis=0, ignore_index=True)
    )
    velocity_boundary = sorted_results[0][21]
    all_OF_likelihood = (
        None
        if any(res[22] is None for res in sorted_results)
        else np.concatenate([res[22] for res in sorted_results], axis=0)
    )
    all_hr = (
        None
        if any(res[23] is None for res in sorted_results)
        else np.concatenate([res[23] for res in sorted_results], axis=0)
    )
    all_hr_tp = (
        None
        if any(res[24] is None for res in sorted_results)
        else np.concatenate([res[24] for res in sorted_results], axis=0)
    )

    all_digital_timer = (
        None
        if any(res[25] is None for res in sorted_results)
        else [res[25] for res in sorted_results]
    )

    all_event_df.to_csv(os.path.join(output_dir, "event_combined.csv"))
    h5_name = os.path.join(output_dir, "data.h5")
    with h5py.File(h5_name, "w") as f:
        f.create_dataset("all_analog_tp", data=all_analog_tp)
        f.create_dataset("all_eeg", data=all_eeg)
        f.create_dataset("all_emg", data=all_emg)
        f.create_dataset("sampling_rate", data=sampling_rate)

        optional_datasets = {
            "all_breathe": all_breathe,
            "all_tem": all_tem,
            "all_b_rate": all_b_rate,
            "all_pupil": all_pupil,
            "all_pupil_tp": all_pupil_tp,
            "all_hr": all_hr,
            "all_hr_tp": all_hr_tp,
            "all_table_v": all_table_v,
            "all_table_tp": all_table_tp,
            "all_OF_tp": all_OF_tp,
            "all_v": all_v,
            "all_cum_d": all_cum_d,
            "all_distance": all_distance,
            "all_t_stft": all_t_stft,
        }
        for dataset_name, value in optional_datasets.items():
            f.create_dataset(dataset_name, data=_hdf_data(value))

        f.create_dataset("f_stft", data=f_stft)

        f.create_dataset("all_linear_power", data=all_linear_power)
        f.create_dataset("all_dB_power", data=all_dB_power)

        f.create_dataset(
            "velocity_boundary", data=velocity_boundary
        ) if velocity_boundary is not None and len(velocity_boundary) > 0 else None

        if all_digital_timer is None:
            f.create_dataset("all_digital_timer", data=MISSING_DATA)
        else:
            dt = h5py.vlen_dtype(np.float64)

            all_digital_timer_obj = np.empty(len(all_digital_timer), dtype=object)
            for i, arr in enumerate(all_digital_timer):
                all_digital_timer_obj[i] = arr
            f.create_dataset("all_digital_timer", data=all_digital_timer_obj, dtype=dt)

        if all_event_df is not None:
            all_event_df.to_hdf(h5_name, key="event_df", mode="a", format="table")

    mouse_name = os.path.basename(data_folder)
    all_ecg = all_emg[1] if len(all_emg) > 1 else None
    for i, electrode in enumerate(EEG_ch_dict):
        pdf_name = mouse_name + "_____" + electrode + "_COMBINED.pdf"
        save_analysis_pdf(
            dlc_type,
            all_analog_tp,
            all_eeg[i],
            all_t_stft,
            f_stft,
            all_linear_power[i],
            all_dB_power[i],
            all_emg[0],
            all_ecg,
            sampling_rate,
            all_hr,
            all_hr_tp,
            all_breathe,
            all_tem,
            all_b_rate,
            all_pupil,
            all_pupil_tp,
            all_table_v,
            all_table_tp,
            all_OF_tp,
            all_v,
            all_cum_d,
            all_distance,
            velocity_boundary,
            all_OF_likelihood,
            all_event_df,
            output_dir=output_dir,
            pdf_name=pdf_name,
            figsize=(45, 25),
        )


def main(argv=None):
    args = parse_args(argv)
    if args.make_demo is not None:
        demo_path = generate_demo_data(args.make_demo)
        print(f"Saved synthetic demo data to {demo_path}")
        return
    if args.demo is not None:
        output_path = render_demo(args.demo, args.output)
        print(f"Saved synthetic demo figure to {output_path}")
        return

    data_folder = args.data_folder
    if data_folder is None:
        selected_folder = choose_data_folder()
        if not selected_folder:
            print("No data folder selected.")
            return
        data_folder = Path(selected_folder)
    if not data_folder.is_dir():
        raise NotADirectoryError(f"Data folder not found: {data_folder}")
    process_data_folder(str(data_folder))


if __name__ == "__main__":
    main()
