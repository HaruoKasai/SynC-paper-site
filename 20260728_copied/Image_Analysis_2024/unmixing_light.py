import os
import glob
import sys
import pathlib
import gc

import numpy as np
import tifffile as tiff
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir / "Lib"))

from ImageJRoiReader import *   # original package
from get_flat_area import *     # original package


coefficient_list = [
    [0, 0.1576 * 2 / 3],   # 0 2023.01.10
    [0, 0.2431],           # 1 2023.02.13
    [0, 0.1626],           # 2 2023.01.17 cLTP
    [0, 0.0541],           # 3 2023.02.20 GEF-dead
    [0, 0.0264],           # 4 2023.02.21 PSD95 UTR
    [0, 0.0541 * 3 / 2],   # 5 2023.02.27 woRapa
    [0, 0.0264 * 3 / 2 * 2],  # 6 2023.03.06 Rapa->cLTP
    [0, 0.0264 / 2],       # 7 2023.03.07 PSD95-UTR , 2023.03.13PSD95-UTR_Vehicle
    [0, 0.0541 * 3 * 2 / 3],  # 8 2023.03.07 w_wo_APV, 20230313_cL100_APV
    [0, 0.0541 * 2 / 3],   # 9 2023.03.14 cLTP
    [0, 0.0541 * 3],       # 10 2023.04.10 cLTD control
    [0, 0.05],             # 11 temp
    [0.00081, 0],          # 12 230514 c-fos imaging mVenus
    [0.00291, 0],          # 13 230514 c-fos imaging 633
    [0.000154, 0],         # 14 240429 ~ paCaMKII imaging_zoom5
    [0.00044, 0],          # 15 240429~ soma_spine_zoom0.75
    [0.00047, 0],          # 16 240624~_A144_conc_zoom0.75_514nm_1.0%
    [0.0023, 0],           # 17 zoom3 514nm 1.0%
    [0.00116, 0],          # 18 zoom3 514nm 0.5%
    [0.0011, 0],           # 19 230514 c-fos imaging mVenus_strong
    [0.000696, 0],         # 20 zoom3 514nm 0.5%, 561 nm laser 3.0%
    [0.000232, 0],         # 21 zoom3 514nm 0.5%, 561 nm laser 1.0%
    [0.00084, 0],          # 22 zoom3 lipofection target protein 561nm 5.0%, 514nm 0.5%
    [0.00129, 0],          # 23 zoom3 lipofection target protein 561 nm 3.0%, 514 nm 0.5%
    [0.001756, 0],         # 24 zoom3 lipofection target protein 561 nm 2.0%, 514 nm 0.5%
    [0.00318, 0],          # 25 zoom3 lipofection target protein 561 nm 1.0%, 514 nm 0.5%
    [0.000104, 0],         # 26 zoom3 lipofection target protein 561nm 5.0%, 514nm 0.3%
    [0.069487, 0]          # 27 2P imaging zoom8 A163 2.0E13, A140 6.7E8, HV500, HV520
]


def unmix_one_plane(img_cyx, param=20):
    """
    img_cyx : shape (C, Y, X)
    return  : shape (2, Y, X) float32
    """
    if img_cyx.ndim != 3:
        raise ValueError(f"Expected (C, Y, X), got {img_cyx.shape}")

    if img_cyx.shape[0] < 2:
        raise ValueError(f"Need at least 2 channels, got shape {img_cyx.shape}")

    a, b = coefficient_list[param]

    Y = img_cyx[0].astype(np.float32, copy=False)
    R = img_cyx[1].astype(np.float32, copy=False)

    BaseR = get_flat_area(R)[0]
    BaseY = get_flat_area(Y)[0]

    V = (Y - b * R + b * BaseR - BaseY) / (1 - b * a)
    R_unmixed = R - a * V - BaseR

    out = np.empty((2, Y.shape[0], Y.shape[1]), dtype=np.float32)
    out[0] = V
    out[1] = R_unmixed
    return out


def infer_output_shape_and_mode(shape, axes):
    """
    入力 axes を解釈して、出力 shape と取り出しモードを返す。

    対応:
      TCYX -> 出力 TCYX
      CTYX -> 出力 TCYX
      ZCYX -> 出力 ZCYX
      CZYX -> 出力 ZCYX
    """
    if axes == "TCYX":
        T, C, Y, X = shape
        return (T, C, Y, X), "TCYX", "TCYX"

    if axes == "CTYX":
        C, T, Y, X = shape
        return (T, C, Y, X), "CTYX", "TCYX"

    if axes == "ZCYX":
        Z, C, Y, X = shape
        return (Z, C, Y, X), "ZCYX", "ZCYX"

    if axes == "CZYX":
        C, Z, Y, X = shape
        return (Z, C, Y, X), "CZYX", "ZCYX"

    raise ValueError(
        f"Unsupported axes={axes}, shape={shape}. "
        f"Expected one of: TCYX, CTYX, ZCYX, CZYX."
    )


def get_plane_cyx(imgs, i, original_axes):
    """
    i 番目の T または Z を取り出して (C, Y, X) を返す
    """
    if original_axes == "TCYX":
        return imgs[i]

    if original_axes == "CTYX":
        return imgs[:, i, :, :]

    if original_axes == "ZCYX":
        return imgs[i]

    if original_axes == "CZYX":
        return imgs[:, i, :, :]

    raise ValueError(f"Unsupported original_axes={original_axes}")


def process_tiff_file(file_path, output_dir, param=20):
    print("=" * 100)
    print(f"Processing: {file_path}")

    basename_wo_ext = os.path.splitext(os.path.basename(file_path))[0]
    output_fname = os.path.join(output_dir, basename_wo_ext + "_unmix.tif")

    with tiff.TiffFile(file_path) as tif:
        series = tif.series[0]
        in_shape = series.shape
        in_dtype = series.dtype
        in_axes = series.axes

    print(f"Input shape : {in_shape}")
    print(f"Input dtype : {in_dtype}")
    print(f"Input axes  : {in_axes}")

    (N, C, Y, X), original_axes, out_axes = infer_output_shape_and_mode(in_shape, in_axes)

    if C < 2:
        raise ValueError(f"Need at least 2 channels, got C={C}")

    imgs = tiff.memmap(file_path)
    print("Opened input TIFF as memmap")

    out = tiff.memmap(
        output_fname,
        shape=(N, 2, Y, X),
        dtype=np.float32,
        imagej=True,
        bigtiff=True,
        metadata={'axes': out_axes}
    )

    print(f"Created output memmap: {output_fname}")
    print(f"Output shape: {(N, 2, Y, X)} dtype=float32 axes={out_axes}")

    axis_name = "Timepoint" if out_axes == "TCYX" else "Z-slice"

    for i in range(N):
        print(f"  {axis_name} {i + 1}/{N}")

        img_cyx = get_plane_cyx(imgs, i, original_axes)
        unmixed = unmix_one_plane(img_cyx, param=param)

        out[i, :, :, :] = unmixed

        if (i + 1) % 5 == 0 or (i + 1) == N:
            out.flush()

        del img_cyx
        del unmixed
        gc.collect()

    out.flush()
    del out
    del imgs
    gc.collect()

    print(f"Saved: {output_fname}")


def main():
    root = tk.Tk()
    root.withdraw()

    img_dir = tkinter.filedialog.askdirectory(
        initialdir=r"\\Synology\arima\raw\2photon",
        title="Select folder containing TIFF files"
    )

    if not img_dir:
        print("No folder selected. Exit.")
        return

    output_dir = img_dir
    img_files = glob.glob(os.path.join(img_dir, "*.tif"))

    print(f"Number of files: {len(img_files)}")

    if len(img_files) == 0:
        tkinter.messagebox.showinfo("Info", "No .tif files found.")
        return

    for file_path in img_files:
        try:
            process_tiff_file(file_path, output_dir, param=20)
        except Exception as e:
            print(f"ERROR while processing {file_path}")
            print(str(e))

    print("Finished")
    tkinter.messagebox.showinfo("Finished", "All processing finished.")


if __name__ == "__main__":
    main()