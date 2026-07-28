import numpy as np
from scipy.optimize import curve_fit
from suite2p.extraction import dcnv
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd

import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed



def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
    root.destroy()
    return folder_path


# --- 指数減衰モデル ---
def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

# --- 静的（quiet）期間のマスクを作る ---
def get_quiet_mask(trace, sigma=100, percentile_threshold=25):
    smoothed = gaussian_filter1d(trace, sigma=sigma)
    dy = np.abs(np.gradient(smoothed))
    threshold = np.percentile(dy, percentile_threshold)
    return dy < threshold

# --- 各トレースに対して指数減衰補正を適用 ---
def correct_trace_bleach(trace, time, mean_level=True):
    if np.any(np.isnan(trace)):
        trace = np.nan_to_num(trace, nan=np.nanmedian(trace))
    quiet_mask = get_quiet_mask(trace)
    t_quiet = time[quiet_mask]
    y_quiet = trace[quiet_mask]

    try:
        popt, _ = curve_fit(exp_decay, t_quiet, y_quiet, p0=(1, 1e-4, y_quiet[-1]))
        fitted = exp_decay(time, *popt)
        if mean_level:
            corrected = trace / fitted * np.mean(y_quiet)
        else:
            corrected = trace / fitted
        return corrected
    except Exception as e:
        return trace  # フィットに失敗した場合は元の信号を使う

# --- パイプライン全体 ---
def run_bleach_correction_pipeline(data_folder, ops_path):
    f_path = os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "F.npy")
    fneu_path = os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "Fneu.npy")
    save_dir = os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected")
    os.makedirs(save_dir, exist_ok=True)

    # --- データ読み込み ---
    ops = np.load(ops_path, allow_pickle=True).item()
    F = np.load(f_path)  # shape: (n_cells, n_frames)
    Fneu = np.load(fneu_path)
    Fc_raw = F - ops['neucoeff'] * Fneu
    n_cells, n_frames = F.shape
    time = np.arange(n_frames)

    # --- ブリーチ補正 ---
    F_corrected = np.array(
        Parallel(n_jobs=-1)(
            delayed(correct_trace_bleach)(Fc_raw[i, :], time)
            for i in range(n_cells)
        )
    )

    # --- Suite2p dcnv処理 ---
    Fc = dcnv.preprocess(
        F=F_corrected,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )
    spks_corrected = dcnv.oasis(
        F=Fc,
        batch_size=ops['batch_size'],
        tau=ops['tau'],
        fs=ops['fs']
    )

    # --- 保存 ---
    np.save(os.path.join(save_dir, 'F_corrected.npy'), F_corrected)
    np.save(os.path.join(save_dir, 'spks_corrected.npy'), spks_corrected)

    print("補正とスパイク推定が完了しました。保存先:", save_dir)
    return F_corrected, spks_corrected


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250718_z254-1_IRES-2x_GCaMP-5e12_soma_imaging_EEG"  # for development
    run_bleach_correction_pipeline(data_folder, r"X:\Behavior\Ca_imaging\ops_roundtrip_2025-07-21.npy")

if __name__ == "__main__":
    main()