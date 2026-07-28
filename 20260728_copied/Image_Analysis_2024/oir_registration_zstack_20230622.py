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
import itk
import pandas as pd
import tifffile
# from registration_translation_sequential_from_center import registration_translation
import glob
from xml.etree import ElementTree as ETree
import json
import tifffile as tiff


def registration_translation(imgs, ref_ch=0):
    timepoints_n = imgs.shape[0]  # z registrationでも同じ
    imgs_aligned = np.empty_like(imgs)
    plus_minus = [-1, 1]
    print(imgs.shape)
    print("imgs.ndim ==" + str(imgs.ndim))
    if imgs.ndim == 3:
        for f in plus_minus:
            ref_img = imgs[int(timepoints_n / 2), :,
                      :]  # z=center+1はcenterに、center+2はcenter+1に、Center+3はcenter+2に、、、という感じで順次そろえていく
            for i in range(int(timepoints_n / 2)):
                print(int(timepoints_n / 2) + f * i)
                fixedImage = itk.GetImageFromArray(ref_img)
                PixelType = itk.ctype('float')
                # PixelType = itk.ctype('int')
                Dimension = fixedImage.GetImageDimension()
                FixedImageType = itk.Image[PixelType, Dimension]
                MovingImageType = itk.Image[PixelType, Dimension]

                TransformType = itk.TranslationTransform[itk.D, Dimension]
                initialTransform = TransformType.New()

                optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
                    # LearningRate=4,
                    LearningRate=4,
                    MinimumStepLength=0.0001,
                    RelaxationFactor=0.5,
                    NumberOfIterations=200)

                # optimizer.SetGradientMagnitudeTolerance(10)

                img = imgs[int(timepoints_n / 2) + f * i, :, :]
                movingImage = itk.GetImageFromArray(img)

                metric = itk.MeanSquaresImageToImageMetricv4[
                    FixedImageType, MovingImageType].New()

                registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixedImage,
                                                                 MovingImage=movingImage,
                                                                 Metric=metric,
                                                                 Optimizer=optimizer,
                                                                 InitialTransform=initialTransform)
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

                registration.Update()

                transform = registration.GetTransform()
                finalParameters = transform.GetParameters()
                translationAlongX = finalParameters.GetElement(0)
                translationAlongY = finalParameters.GetElement(1)

                numberOfIterations = optimizer.GetCurrentIteration()

                bestValue = optimizer.GetValue()

                print("Result = ")
                print(" Translation X = " + str(translationAlongX))
                print(" Translation Y = " + str(translationAlongY))
                print(" Iterations    = " + str(numberOfIterations))
                print(" Metric value  = " + str(bestValue))

                CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
                outputCompositeTransform = CompositeTransformType.New()
                outputCompositeTransform.AddTransform(movingInitialTransform)
                outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

                # リサンプリング
                resampler = itk.ResampleImageFilter.New(Input=movingImage,
                                                        Transform=outputCompositeTransform,
                                                        UseReferenceImage=True,
                                                        ReferenceImage=fixedImage)

                # 結果表示
                resampler.SetDefaultPixelValue(100)

                OutputPixelType = itk.ctype('unsigned char')
                OutputImageType = itk.Image[OutputPixelType, Dimension]

                resampler.Update()
                result = resampler.GetOutput()

                imgs_aligned[int(timepoints_n / 2) + f * i] = itk.GetArrayFromImage(result)
                ref_img = imgs_aligned[int(timepoints_n / 2) + f * i]

    if imgs.ndim == 4:  ############################TODO sequential式に書き直していない   #################################
        ch_n = imgs.shape[1]
        for f in plus_minus:
            ref_img = imgs[int(timepoints_n / 2), ref_ch, :, :]
            for i in range(int(timepoints_n / 2)):
                fixedImage = itk.GetImageFromArray(ref_img)
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
                    NumberOfIterations=200)

                img = imgs[int(timepoints_n / 2) + f * i, ref_ch, :, :]
                movingImage = itk.GetImageFromArray(img)

                metric = itk.MeanSquaresImageToImageMetricv4[
                    FixedImageType, MovingImageType].New()

                registration = itk.ImageRegistrationMethodv4.New(FixedImage=fixedImage,
                                                                 MovingImage=movingImage,
                                                                 Metric=metric,
                                                                 Optimizer=optimizer,
                                                                 InitialTransform=initialTransform)
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

                registration.Update()
                transform = registration.GetTransform()
                finalParameters = transform.GetParameters()
                translationAlongX = finalParameters.GetElement(0)
                translationAlongY = finalParameters.GetElement(1)

                numberOfIterations = optimizer.GetCurrentIteration()

                bestValue = optimizer.GetValue()

                # print("Result = ")
                # print(" Translation X = " + str(translationAlongX))
                # print(" Translation Y = " + str(translationAlongY))
                # print(" Iterations    = " + str(numberOfIterations))
                # print(" Metric value  = " + str(bestValue))
                # print(" current frame  = " + str(int(timepoints_n/2)+f*i))

                CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
                outputCompositeTransform = CompositeTransformType.New()
                outputCompositeTransform.AddTransform(movingInitialTransform)
                outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

                # リサンプリング
                for ch in range(ch_n):
                    img_ch = imgs[int(timepoints_n / 2) + f * i, ch, :, :]
                    movingImage_ch = itk.GetImageFromArray(img_ch)
                    resampler = itk.ResampleImageFilter.New(Input=movingImage_ch,
                                                            Transform=outputCompositeTransform,
                                                            UseReferenceImage=True,
                                                            ReferenceImage=fixedImage)

                    # 結果表示
                    resampler.SetDefaultPixelValue(100)

                    OutputPixelType = itk.ctype('unsigned char')
                    OutputImageType = itk.Image[OutputPixelType, Dimension]

                    resampler.Update()
                    result = resampler.GetOutput()

                    imgs_aligned[int(timepoints_n / 2) + f * i][ch] = itk.GetArrayFromImage(result)
                    if ch == ref_ch:
                        ref_img = imgs_aligned[int(timepoints_n / 2) + f * i][ch]

        # imgs_aligned = np.reshape(imgs_aligned, (timepoints_n, ch_n, :,:))

    return imgs_aligned

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
dir = filedialog.askdirectory(initialdir=r"\\Synology\arima\raw\2photon\iAS_imaging\20250423")
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

        reg_img = registration_translation(img, ref_ch=0)
        sum = np.sum(reg_img, axis=0, dtype=np.float32)

        tiff.imwrite(os.path.join(output_dir, basename_wo_ext + "_reg.tif"), reg_img, imagej=True, metadata={'axes': 'ZCYX'})
        tiff.imwrite(os.path.join(output_dir, basename_wo_ext + "_reg_sum.tif"), sum, imagej=True, metadata={'axes': 'CYX'})




