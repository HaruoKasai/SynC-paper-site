# import logging
import os
import time
import math
import sys
import tkinter as tk
from tkinter import filedialog
import numpy as np
import itk
import pandas as pd
import tifffile
# from registration_translation_sequential_from_center import registration_translation
import glob
from xml.etree import ElementTree as ETree
import json
import tifffile as tiff


def registration_translation_sliding_reference(
        imgs,
        ref_ch=0,
        window_radius=5,
        max_shift_px=50,
        min_corr=0.25
):
    """
    imgs:
        3D: ZYX
        4D: ZCYX

    各Zを、その周辺Zの平均画像にregistrationする。
    bad ZはQCで判定する。
    """

    z_n = imgs.shape[0]
    imgs_aligned = imgs.copy()
    qc_list = []

    is_4d = imgs.ndim == 4

    for z in range(z_n):
        z0 = max(0, z - window_radius)
        z1 = min(z_n, z + window_radius + 1)

        if is_4d:
            ref_stack = imgs[z0:z1, ref_ch, :, :]
            moving_ref = imgs[z, ref_ch, :, :]
        else:
            ref_stack = imgs[z0:z1, :, :]
            moving_ref = imgs[z, :, :]

        # 自分自身も含む周辺Z平均
        ref_img = np.mean(ref_stack, axis=0).astype(np.float32)
        moving_img = moving_ref.astype(np.float32)

        fixedImage = itk.GetImageFromArray(ref_img)
        movingImage = itk.GetImageFromArray(moving_img)

        PixelType = itk.ctype('float')
        Dimension = fixedImage.GetImageDimension()
        FixedImageType = itk.Image[PixelType, Dimension]
        MovingImageType = itk.Image[PixelType, Dimension]

        TransformType = itk.TranslationTransform[itk.D, Dimension]
        initialTransform = TransformType.New()

        optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
            LearningRate=4,
            MinimumStepLength=0.0001,
            RelaxationFactor=0.5,
            NumberOfIterations=200
        )

        metric = itk.MeanSquaresImageToImageMetricv4[
            FixedImageType, MovingImageType
        ].New()

        registration = itk.ImageRegistrationMethodv4.New(
            FixedImage=fixedImage,
            MovingImage=movingImage,
            Metric=metric,
            Optimizer=optimizer,
            InitialTransform=initialTransform
        )

        movingInitialTransform = TransformType.New()
        initialParameters = movingInitialTransform.GetParameters()
        initialParameters[0] = 0
        initialParameters[1] = 0
        movingInitialTransform.SetParameters(initialParameters)
        registration.SetMovingInitialTransform(movingInitialTransform)

        identityTransform = TransformType.New()
        identityTransform.SetIdentity()
        registration.SetFixedInitialTransform(identityTransform)

        registration.SetNumberOfLevels(1)
        registration.SetSmoothingSigmasPerLevel([0])
        registration.SetShrinkFactorsPerLevel([1])

        try:
            registration.Update()

            transform = registration.GetTransform()
            finalParameters = transform.GetParameters()
            dx = float(finalParameters.GetElement(0))
            dy = float(finalParameters.GetElement(1))

            shift_norm = math.sqrt(dx ** 2 + dy ** 2)

            CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
            outputCompositeTransform = CompositeTransformType.New()
            outputCompositeTransform.AddTransform(movingInitialTransform)
            outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

            # ref channelで相関チェック用にregistration
            resampler = itk.ResampleImageFilter.New(
                Input=movingImage,
                Transform=outputCompositeTransform,
                UseReferenceImage=True,
                ReferenceImage=fixedImage
            )
            resampler.SetDefaultPixelValue(0)
            resampler.Update()

            aligned_ref = itk.GetArrayFromImage(resampler.GetOutput()).astype(np.float32)

            corr = np.corrcoef(ref_img.ravel(), aligned_ref.ravel())[0, 1]
            if np.isnan(corr):
                corr = -1

            is_good = (shift_norm <= max_shift_px) and (corr >= min_corr)

            # 全channelへ同じtransformを適用
            if is_4d:
                for ch in range(imgs.shape[1]):
                    moving_ch = itk.GetImageFromArray(imgs[z, ch, :, :].astype(np.float32))

                    resampler_ch = itk.ResampleImageFilter.New(
                        Input=moving_ch,
                        Transform=outputCompositeTransform,
                        UseReferenceImage=True,
                        ReferenceImage=fixedImage
                    )
                    resampler_ch.SetDefaultPixelValue(0)
                    resampler_ch.Update()

                    aligned_ch = itk.GetArrayFromImage(resampler_ch.GetOutput())
                    imgs_aligned[z, ch, :, :] = aligned_ch.astype(imgs.dtype)
            else:
                imgs_aligned[z, :, :] = aligned_ref.astype(imgs.dtype)

            status = "good" if is_good else "bad"

        except Exception as e:
            dx = np.nan
            dy = np.nan
            shift_norm = np.nan
            corr = np.nan
            is_good = False
            status = "failed"
            print(f"Registration failed at z={z}: {e}")

        qc_list.append({
            "z": z,
            "z0_ref": z0,
            "z1_ref": z1,
            "dx": dx,
            "dy": dy,
            "shift_norm": shift_norm,
            "corr": corr,
            "is_good": is_good,
            "status": status
        })

        print(
            f"z={z}/{z_n-1}, dx={dx:.2f}, dy={dy:.2f}, "
            f"shift={shift_norm:.2f}, corr={corr:.3f}, {status}"
        )

    qc_df = pd.DataFrame(qc_list)
    return imgs_aligned, qc_df

def convert_oir(fname):
    print("##################################################################################")
    image = bioformats.load_image(fname)
    # print(image)
    # print(image.shape)
    # print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    # print("::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
    md = bioformats.get_omexml_metadata(fname)
    rdr = bioformats.ImageReader(fname, perform_init=True)
    # time.sleep(4)  # threadなので時間かかる→joinできないのか
    treeroot = ETree.fromstring(md)
    series = []
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
        # print("##################################################################################")
        # print("##################################################################################")
        # print("##################################################################################")
        # print("##################################################################################")
        # print("##################################################################################")
        # print(znum)
        # print(tnum)
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
root.attributes("-topmost", True)
dir = filedialog.askdirectory(initialdir=r"\\Synology\arima\raw\2photon\iAS_imaging\20250423")

root.destroy()

import javabridge
import bioformats
# logging.getLogger("bioformats").setLevel(logging.WARNING)
# logging.getLogger("javabridge").setLevel(logging.WARNING)
javabridge.start_vm(class_path=bioformats.JARS)

f_list = glob.glob(os.path.join(dir, "*.oir"))
for fname in f_list:
    if "beads" not in fname and "lowmag" not in fname:
        print(fname)
        img = convert_oir(fname).astype(np.float32)
        print("###############")
        print(img.shape)

        output_dir = os.path.join(os.path.dirname(fname), "sum")
        basename_wo_ext = os.path.splitext(os.path.basename(fname))[0]
        print("basename_wo_ext")
        print(basename_wo_ext)

        reg_img, qc_df = registration_translation_sliding_reference(
            img,
            ref_ch=0,
            window_radius=5,
            max_shift_px=50,
            min_corr=0.25
        )

        sum_all = np.sum(reg_img, axis=0, dtype=np.float32)

        good_mask = qc_df["is_good"].values
        reg_img_good = reg_img[good_mask]
        sum_good = np.sum(reg_img_good, axis=0, dtype=np.float32)

        tiff.imwrite(
            os.path.join(output_dir, basename_wo_ext + "_reg_sliding.tif"),
            reg_img,
            imagej=True,
            metadata={"axes": "ZCYX"}
        )

        tiff.imwrite(
            os.path.join(output_dir, basename_wo_ext + "_reg_sliding_sum.tif"),
            sum_all,
            imagej=True,
            metadata={"axes": "CYX"}
        )

        tiff.imwrite(
            os.path.join(output_dir, basename_wo_ext + "_reg_sliding_good_z.tif"),
            reg_img_good,
            imagej=True,
            metadata={"axes": "ZCYX"}
        )

        tiff.imwrite(
            os.path.join(output_dir, basename_wo_ext + "_reg_sliding_sum_good_z.tif"),
            sum_good,
            imagej=True,
            metadata={"axes": "CYX"}
        )

        qc_df.to_csv(
            os.path.join(output_dir, basename_wo_ext + "_registration_qc.csv"),
            index=False
        )





