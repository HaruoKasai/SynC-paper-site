import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk
from tkinter import filedialog
import os

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select F.npy file",
        filetypes=[("NumPy array", "*.npy")])
    root.destroy()
    return file_path

def parse_range(text):
    text = text.replace('~', '-')
    start, end = [int(x.strip()) for x in text.split('-')]
    if end <= start:
        raise ValueError("frame_range end must be > start.")
    return start, end

def bin_time_1d(x, bin_size=3):
    n = len(x) // bin_size
    if n == 0:
        return np.array([])
    X = x[:n * bin_size].reshape(n, bin_size)
    return X.mean(axis=1)

def normalize_1d(x):
    mu = np.mean(x)
    sd = np.std(x)
    return (x - mu) / sd if sd > 0 else x * 0

def plot_traces(F, cells, frame_range, save_path, bin_size=3):
    start, end = frame_range
    traces = []
    for c in cells:
        tr = F[c, start:end]
        tr_b = bin_time_1d(tr, bin_size)
        if tr_b.size == 0:
            continue
        tr_z = normalize_1d(tr_b)
        traces.append(tr_z)

    if not traces:
        raise ValueError("No valid traces in range.")

    plt.figure(figsize=(8, 16))
    cmap = cm.get_cmap("tab20", len(traces))
    sds = [np.std(t) for t in traces]
    offset = np.median(sds) * 2 if sds else 2.0

    for i, t in enumerate(traces):
        plt.plot(t + i * offset, color=cmap(i), lw=0.9)

    # === scale bar ===
    bar_len = 1  # 1 SD
    y0 = -offset * 0.5
    x0 = len(traces[0]) * 1.02
    plt.plot([x0, x0], [y0, y0 + bar_len], color='k', lw=2)
    plt.text(x0 + 3, y0 + bar_len / 2, "1 SD", va='center', fontsize=10)

    plt.xlabel(f"Frames (binned every {bin_size})")
    plt.ylabel("ΔF/F (z-score, offset)")
    plt.title("Suite2p Traces (3-frame binning, z-score)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
    plt.close()
    print(f"Saved to: {save_path}")

def main():
    file_path = select_file()
    if not file_path:
        print("No file selected.")
        return

    F = np.load(file_path)
    print(f"Loaded F.npy: shape {F.shape}")
    folder = os.path.dirname(file_path)
    iscell = np.load(os.path.join(os.path.dirname(folder), "suite2p", "plane0", "iscell.npy"))
    cell_indices = np.where(iscell[:, 0] == 1)[0]
    F = F[cell_indices]


    # ===== ユーザー設定 =====
    cells = list(range(0, 177)) #+ [113, 151]
    frame_range = [117000, 123000]
    bin_size = 3


    save_path = os.path.join(folder, f"traces_{frame_range[0]}_{frame_range[1]}_bin{bin_size}_with_scalebar.pdf")

    plot_traces(F, cells, frame_range, save_path, bin_size)

if __name__ == "__main__":
    main()