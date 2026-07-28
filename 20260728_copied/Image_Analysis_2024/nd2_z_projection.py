import os
# from nd2reader import ND2Reader
import nd2reader
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tifffile

root = tk.Tk()
root.withdraw()

# フォルダの選択ダイアログを表示します
folder_path = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data")

# フォルダ内をクロールして、nd2ファイルを検索します
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.nd2'):
            file_path = os.path.join(root, file)
            print(file_path)
            # print(f'Converting {file_path}...')

            # nd2readerライブラリを使用して、nd2ファイルを読み込みます
            with nd2reader.ND2Reader(file_path) as images: #https://github.com/Open-Science-Tools/nd2reader/issues/24


                ndim = len(images.sizes)
                keys = images.sizes.keys()
                print(images)
                if "v" in keys:
                    images.iter_axes = 'v'
                    if ndim ==6:
                        images.bundle_axes = 'tczyx'
                        axis_z=2
                        zsum_axes='TCYX'
                    elif ndim==5:
                        images.bundle_axes = 'tzyx'
                        axis_z=1
                        zsum_axes = 'TYX'
                    for v in range(len(images)):  # v: field of view
                        tiff_name = "SUM_" + file[:-4] + "_series%s.tiff" % (v + 1)
                        print("** Processing %s" % tiff_name)
                        im = images[v]
                        sum = np.sum(im, axis=axis_z, dtype=np.float32)
                        tifffile.imwrite(os.path.join(root,tiff_name), sum, imagej=True, metadata={'axes': zsum_axes})

                else:
                    if ndim==5:
                        images.bundle_axes = 'tczyx'
                        axis_z = 2
                        zsum_axes = 'TCYX'
                    elif ndim==4:
                        images.bundle_axes = 'tzyx'
                        axis_z = 1
                        zsum_axes = 'TYX'
                    tiff_name = "SUM_" + file[:-4] + ".tiff"
                    print("** Processing %s" % tiff_name)
                    im = images[0]
                    sum = np.sum(im, axis=axis_z, dtype=np.float32)
                    tifffile.imwrite(os.path.join(root, tiff_name), sum, imagej=True, metadata={'axes': zsum_axes})


                # tp_num = images.shape[1]
                    #　もしデータが大きすぎて分割する場合は以下の感じ？
                    # for ts in range(int(tp_num/10)): #dataが大きすぎるときはtを分割する
                    #     if tp_num<10:
                    #         tiff_name = "SUM_"+file[:-4]+"_cell%s.tiff" %v
                    #     else:
                    #         tiff_name = "SUM_" + file[:-4] + "_cell%s_t%s-%s.tiff" %(v,ts*10,ts*10+9)
                    #     print("******* Processing %s"%tiff_name)
                    #     im = images[v][ts * 10: ts * 10 + 10]
                    #     sum = np.sum(im, axis=2, dtype=np.float32)
                    #     tifffile.imwrite(os.path.join(root,tiff_name), sum, imagej=True, metadata={'axes': 'TCYX'})

