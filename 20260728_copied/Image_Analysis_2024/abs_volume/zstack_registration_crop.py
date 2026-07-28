import pandas as pd
import numpy as np
import os
import itk
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams["font.family"]= "Arial"
import matplotlib.pyplot as plt
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib


tp=2
img_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1\zstack"
ref_dir = r"N:\SynC_invitro_test\EDF1c_fluctuation\SynC1"

def registration_translation(imgs, ref_ch=1):
    z_num = imgs.shape[0]
    imgs_aligned = np.empty_like(imgs)

    ref_img = imgs[0, ref_ch, :, :]

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

    #zstackをrefにregistrationして、そのXY移動を、zstackに当てはめる
    img = imgs[1, ref_ch, :, :]
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
    imgs_aligned[0] = imgs[0]
    for ch in range(2):
        for i in range(1, z_num):
            img = imgs[i,ch, :,:]
            movingImage = itk.GetImageFromArray(img)
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

            imgs_aligned[i][ch] = itk.GetArrayFromImage(result)
    return imgs_aligned





######################################################
img_files = glob.glob(os.path.join(img_dir, "*(series*.tif"))
for file in img_files:
    img = tiff.imread(file)[tp,:,:,:,:]
    exp_name = os.path.basename(file)[:5]
    dend = os.path.basename(file)[-12:-5]
    print(exp_name)
    sum = np.sum(img, axis=0)
    print(dend)
    #該当するcrop済み-ROI指定済みの画像にregistrationする
    # date_list = glob.glob(os.path.join(ref_dir, "[!_]*"))
    # for date in date_list:
    #     ref_file_list = glob.glob(os.path.join(date, "*"+dend+"*.tif"))
    #     if len(ref_file_list)>0:
    #         ref_file = ref_file_list[0]
    #         print(ref_file)
    ref_file = glob.glob(os.path.join(ref_dir, "*"+exp_name+"*"+dend+"*.tif"))[0]
    print(ref_file)
    ref_crop = tiff.imread(ref_file)[tp,:,:,:]
    print(ref_crop.shape)
    y, x = ref_crop.shape[1], ref_crop.shape[2]
    ref = np.zeros((2,1024,1024))
    ref[:, :y, :x] = ref_crop
    stack = np.vstack([np.expand_dims(ref, axis=0), np.expand_dims(sum, axis=0),img]).astype(np.float32)

    crop = registration_translation(stack)[2:,:, :y, :x]
    print(crop.shape)

    tiff.imwrite(file.split(".")[0]+"_"+dend+"_reg_cr.tif", crop,imagej=True, metadata={'axes': 'ZCYX'})


print("Finished")

