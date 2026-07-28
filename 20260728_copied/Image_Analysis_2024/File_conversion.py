import bioformats
import javabridge
import tifffile
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import glob



def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select the 'data' directory", initialdir=r"X:\Behavior\Ca_imaging")
    root.destroy()
    return folder_path

def oir_to_multipage_tiff(input_path,  output_path):

    with bioformats.ImageReader(input_path) as reader:
        metadata = bioformats.get_omexml_metadata(path=input_path)
        ome = bioformats.OMEXML(metadata)
        size_z = ome.image().Pixels.SizeZ
        size_t = ome.image().Pixels.SizeT
        size_c = ome.image().Pixels.SizeC

        images = []
        for t in range(size_t):
            for z in range(size_z):
                for c in range(size_c):
                    img = reader.read(c=c, z=z, t=t, rescale=False)
                    images.append(img)

        # 保存 (uint16などのまま保存)
        tifffile.imwrite(output_path, np.array(images), photometric='minisblack')

    print(f"Saved multipage TIFF to {output_path}")

def process_folder(mouse_dir):
    all_dirs = glob.glob(os.path.join(mouse_dir, "*"))  # 全てのディレクトリ/ファイルを取得
    exp_list = [d for d in all_dirs if os.path.basename(d)[0] != "_" and os.path.basename(d)[0] == "0"]
    output_dir = os.path.join(mouse_dir, "_GCaMP")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for exp in exp_list:
        print("Processing", os.path.basename(exp))
        oir_path = glob.glob(os.path.join(exp, "*.oir"))[0]
        oir_to_multipage_tiff(oir_path, os.path.join(output_dir, os.path.basename(exp)[:2]+".tif"))


def main():
    data_folder = select_folder()
    # data_folder = r"X:\Behavior\Ca_imaging\20250707_z251-2_SynC-GCaMP"
    javabridge.start_vm(class_path=bioformats.JARS)
    try:
        process_folder(data_folder)
    finally:
        # 最後にVMをkill
        javabridge.kill_vm()

if __name__ == "__main__":
    main()


# 使用例
# oir_to_multipage_tiff(r"X:\Behavior\Ca_imaging\20250707_z251-2_SynC-GCaMP\after\20250707_z251-2_soma_after-55000cycles.oir", output_dir, output_name)