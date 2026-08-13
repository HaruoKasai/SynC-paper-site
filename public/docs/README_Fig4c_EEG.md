## Fig. 4c EEG/EMG analysis

This package publishes the preprocessing and visualisation pipeline used for
the EEG/EMG analysis associated with Fig. 4c, together with a compact public
demo that exercises the time-series and short-time Fourier transform (STFT)
workflow.

### Files

- `Fig4c_EEG_analysis.py`: raw Blackrock EEG/EMG preprocessing pipeline plus a
  self-contained public demo mode.
- `Fig4c_EEG_demo.npz`: deterministic synthetic EEG, EMG and movement signals
  for testing the public workflow (1,504,742 bytes; approximately 1.44 MiB).

### Important data-status note

`Fig4c_EEG_demo.npz` is **synthetic demonstration data**. It contains no animal,
recording-session or personal identifiers and is not a subset of the
manuscript's biological data. It is suitable for checking installation, input
validation, STFT processing and figure generation, but it must not be used to
recalculate or verify the quantitative result shown in Fig. 4c.

The raw Blackrock `.ns2`/`.ns3` recordings are intentionally not included in
the website bundle. Raw electrophysiology files are typically much larger than
is appropriate for a static website or ordinary Git repository. A compact,
explicitly labelled synthetic archive also avoids publishing an arbitrary
single-animal trace as though it were the supporting dataset for the panel.

### Demo contents

The NumPy archive uses schema version 1 and contains:

| Name | Type/shape | Meaning |
| --- | --- | --- |
| `schema_version` | scalar integer | Public demo schema (`1`) |
| `synthetic` | scalar Boolean | Always `True` for this archive |
| `random_seed` | scalar integer | Deterministic generator seed (`404`) |
| `sampling_rate_hz` | scalar float | Sampling rate (`2,000 Hz`) |
| `time_s` | 120,000 floats | Time points for the 60-s demo |
| `eeg_uv` | 120,000 floats | Synthetic EEG voltage in microvolts |
| `emg_uv` | 120,000 floats | Synthetic EMG voltage in microvolts |
| `movement_mm_s` | 120,000 floats | Synthetic movement velocity in mm/s |

The three 20-s intervals have deliberately different spectral and movement
content so that a successful run has an easy visual check: a 2-Hz-dominant
quiet interval, an 8-Hz/40-Hz active interval with increased EMG and movement,
and a 4.5-Hz interval with modulated 35-Hz activity.

### Run the public demo

Python 3.10 or newer is recommended. Install the three demo dependencies:

```bash
python -m pip install numpy scipy matplotlib
```

Place the Python and NPZ files in the same directory, then run:

```bash
python Fig4c_EEG_analysis.py --demo Fig4c_EEG_demo.npz \
  --output Fig4c_EEG_demo_output.png
```

The command validates the archive and generates a four-panel PNG containing
the EEG trace, 0-100-Hz spectrogram, EMG trace and movement velocity. The
spectrogram uses the analysis constants in the script: a 2-s STFT epoch, 50%
overlap and a 100-Hz display limit.

The distributed demo can be regenerated exactly from the documented fixed
seed and generator embedded in the script:

```bash
python Fig4c_EEG_analysis.py --make-demo Fig4c_EEG_demo.npz
```

### Run the raw-data pipeline

The original folder-selection interface remains the default:

```bash
python Fig4c_EEG_analysis.py
```

For non-interactive use, pass the folder containing `_analysis_param.json` and
one or more `.ns2`/`.ns3` recordings:

```bash
python Fig4c_EEG_analysis.py --data-folder PATH_TO_RECORDING_FOLDER
```

The raw-data workflow additionally requires `pandas`, `h5py` and `neo`. The
pupillometry path requires `joblib` and `scikit-image`; the open-field path
also uses the laboratory `lib.DLCAnalysis` module. Those optional dependencies
are not required for the public NPZ demo.

The raw pipeline writes per-recording PDF summaries to `results/` and combined
outputs to `_Combined/`, including `data.h5`, `event_combined.csv` and combined
PDF summaries.

### Integrity

SHA-256 checksums for this release:

```text
107713C5758742BE14D94D5494122C2255AD8A185FF77505A33AF2B717B1611C  Fig4c_EEG_analysis.py
83EB97B0B77399B4994F66632401413634D74925C7BDE1A236ED31D1EEFBCC0C  Fig4c_EEG_demo.npz
```
