import numpy as np
from scipy.optimize import curve_fit
from suite2p.extraction import dcnv
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from scipy.ndimage import percentile_filter
from joblib import Parallel, delayed



def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
    root.destroy()
    return folder_path


def process_folder(data_folder):
    # -- パス設定 --
    f_path = os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "F.npy")
    fneu_path = os.path.join(data_folder, "_GCaMP", "suite2p", "plane0", "Fneu.npy")
    save_dir = os.path.join(data_folder, "_GCaMP", "suite2p_bleach_corrected")
    event_df = pd.read_csv(os.path.join(data_folder, "_Combined", "event_combined.csv"))
    frame2p_df = pd.read_csv(os.path.join(data_folder, "_Combined", "2p_frame_time_combined.csv"))
    os.makedirs(save_dir, exist_ok=True)

    # -- データ読み込み --
    ops = np.load(r"X:\Behavior\Ca_imaging\ops_roundtrip_2025-07-21.npy", allow_pickle=True).item()
    F = np.load(f_path)  # shape: (n_cells, n_frames)
    print("F shape", F.shape)
    Fneu = np.load(fneu_path)
    Fc_raw = F - ops['neucoeff'] * Fneu
    n_cells, n_frames = F.shape
    time = np.arange(n_frames)

    # window_sec = 60  # 秒単位のウィンドウ幅
    percentile = 20
    fs = ops['fs']  # サンプリングレート [Hz]
    # win_frames = int(window_sec * fs)
    win_frames = 8192#2の累乗だと速くなるらしい。window_secは120秒強くらい

    # -- 出力用 --

    def correct_trace(trace, percentile, win_frames):
        if np.any(np.isnan(trace)):
            trace = np.nan_to_num(trace, nan=np.nanmedian(trace))
        baseline = percentile_filter(trace, percentile=percentile, size=win_frames, mode='reflect')
        baseline[baseline == 0] = 1e-6
        return (trace - baseline)/baseline

    F_corrected = np.array(
        Parallel(n_jobs=-1)(  # n_jobs=-1でCPU全コア使用
            delayed(correct_trace)(Fc_raw[i, :], percentile, win_frames)
            for i in range(n_cells)
        )
    )

    Fc = dcnv.preprocess(
        F=F_corrected,
        baseline=ops['baseline'],
        win_baseline=ops['win_baseline'],
        sig_baseline=ops['sig_baseline'],
        fs=ops['fs'],
        prctile_baseline=ops['prctile_baseline']
    )
    spks_corrected = dcnv.oasis(F=Fc, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
    # -- 保存 --
    np.save(os.path.join(save_dir, 'F_corrected.npy'), F_corrected)
    np.save(os.path.join(save_dir, 'spks_corrected.npy'), spks_corrected)



def main():
    # data_folder = select_folder()
    data_folder = r"X:\Behavior\Ca_imaging\20250718_z254-1_IRES-2x_GCaMP-5e12_soma_imaging_EEG"  # for development
    process_folder(data_folder)

if __name__ == "__main__":
    main()