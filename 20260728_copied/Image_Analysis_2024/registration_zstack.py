# import logging
import javabridge
import bioformats
# logging.getLogger("bioformats").setLevel(logging.WARNING)
# logging.getLogger("javabridge").setLevel(logging.WARNING)
javabridge.start_vm(class_path=bioformats.JARS)
import os
import time
import math
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import tifffile
from registration_translation_sequential_from_center import registration_translation
import glob
from xml.etree import ElementTree as ETree
import json
import tifffile as tiff


def convert_oir(fname):
    print("##################################################################################")
    image = bioformats.load_image(fname)
    print(image)
    print(image.shape)
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # md = bioformats.get_omexml_metadata(fname)
    # rdr = bioformats.ImageReader(fname, perform_init=True)
    # # time.sleep(4)  # threadなので時間かかる→joinできないのか
    # treeroot = ETree.fromstring(md)
    # series = []
    for e in treeroot.getchildren():
        if "ID" in e.attrib and e.attrib["ID"][:5] == "Image":
            # e.getchildren():[<Element '{http://www.openmicroscopy.org/Schemas/OME/2016-06}InstrumentRef' at 0x0000021228748548>, <Element '{http://www.openmicroscopy.org/Schemas/OME/2016-06}ObjectiveSettings' at 0x0000021228748598>, <Element '{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels' at 0x00000212287486D8>]
            # pixelsにimage_param入っている
            # ファイル拡張子によってpixelsの番号が異なる
            image_param = e.getchildren()[3].attrib
            series.append(image_param)

    for sidx, s in enumerate(series):
        znum = int(s["SizeZ"])
        tnum = int(s["SizeT"])
        print("##################################################################################")
        print("##################################################################################")
        print("##################################################################################")
        print("##################################################################################")
        print("##################################################################################")
        print(znum)
        print(tnum)
        base = np.zeros((int(s['SizeT']), int(s["SizeY"]), int(s["SizeX"]), int(s["SizeC"]), int(s["SizeZ"]), 1),
                        dtype=np.uint16)
        # 1Ch imageだとrdr.readのshapeが変わりc=0とする対応が必要
        if int(s['SizeC']) == 1:
            for z in range(0, znum):
                for t in range(0, tnum):
                    base[t, :, :, 0, z, 0] = rdr.read(c=0, t=t, z=z, rescale=False)
        else:
            for z in range(0, znum):
                for t in range(0, tnum):
                    base[t, :, :, :, z, 0] = rdr.read(z=z, t=t, rescale=False)
        # base = base.transpose(5, 4, 3, 1, 2, 0)
        base = base.transpose(0, 4, 3, 1, 2, 5)
        # tifffileでimageJでhyperstackになるように保存するためにはdimensions in TZCYXS order XYの順番よくわからないけどとりあえずこの順番でうまくいく
        # 次元変換の順番間違えていたがsとtが同じなため今まで問題がしょうじていなかったのを修正210201

        # tiff.imsave(out_fname, base, imagej=True,
        #             resolution=(1. / float(s['PhysicalSizeX']), 1. / float(s['PhysicalSizeY'])),
        #             metadata={'unit': 'um'})

        base = base.reshape(base.shape[1], base.shape[2],base.shape[3],base.shape[4],)
        return base



root = tk.Tk()
root.withdraw()
dir = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\sawada\raw\S")
f_list = glob.glob(os.path.join(dir, "*.tif"))
for fname in f_list:
    if "beads" not in fname and "lowmag" not in fname:
        print(fname)

        # #ch数を取得
        # def submit():
        #     global ch
        #     ch = entry.get()  # 入力されたテキストを取得
        #     root.quit()  # ウィンドウを閉じる
        # root = tk.Tk()  # Tkインスタンスを作成
        # root.title("Ch　number")  # ウィンドウのタイトルを設定
        # entry = tk.Entry(root)  # Entry（テキストボックス）を作成
        # entry.pack()
        # submit_button = tk.Button(root, text="Submit", command=submit)  # "Submit"という名前のボタンを作成。ボタンが押されるとsubmit関数が呼び出される
        # submit_button.pack()
        # root.mainloop()  # GUIを表示
        # cnum = int(ch)
        img = tiff.imread(fname).astype(np.float32)
        if img.ndim ==3:
            img = np.transpose(img, (0, 1, 2))
        else:
            img = np.transpose(img, (1,0,2,3))
        print(img.shape)
        reg_img = registration_translation(img, ref_ch=0)
        sum = np.sum(reg_img, axis=0, dtype=np.float32)


        output_dir = os.path.join(os.path.dirname(fname), "sum")
        os.makedirs(output_dir, exist_ok=True)
        basename_wo_ext = os.path.splitext(os.path.basename(fname))[0]

        if len(reg_img.shape) == 3:
            tiff.imwrite(os.path.join(output_dir, basename_wo_ext + "_reg.tif"), reg_img, imagej=True, metadata={'axes': 'ZYX'})
        else:
            tiff.imwrite(os.path.join(output_dir, basename_wo_ext + "_reg.tif"), reg_img, imagej=True,
                         metadata={'axes': 'ZCYX'})
        #tiff.imwrite(os.path.join(output_dir, basename_wo_ext + "_reg_sum.tif"), sum, imagej=True, metadata={'axes': 'CYX'})