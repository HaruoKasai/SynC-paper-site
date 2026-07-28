import pandas as pd
import numpy as np
import os
import tifffile as tiff
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from read_roi import read_roi_zip, read_roi_file
import glob
import tkinter.filedialog
import tkinter.messagebox
import tkinter as tk
import itk
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/Lib')
from ImageJRoiReader import * #original package
from get_flat_area import * #original package

#ref_ch: reginstrationに使うchを指定(指定されたchでregistrationに使う変換行列を計算し、他のChにも同一の変換行列を適用する）


def registration_trasnlation(imgs, ref_ch):
    timepoints_n = imgs.shape[0] #z registrationでも同じ
    imgs_aligned = np.empty_like(imgs)


    if imgs.ndim == 3: #1 color
        ref_img = imgs[0, :, :]
        for i in range(timepoints_n):
            fixedImage = itk.GetImageFromArray(ref_img)
            PixelType = itk.ctype('float')
            Dimension = fixedImage.GetImageDimension()
            FixedImageType = itk.Image[PixelType, Dimension]
            MovingImageType = itk.Image[PixelType, Dimension]

            TransformType = itk.TranslationTransform[itk.D, Dimension]
            initialTransform = TransformType.New()

            optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
                LearningRate=10,
                MinimumStepLength=0.0001,
                RelaxationFactor=0.05,
                NumberOfIterations=500)

            img = imgs[i,:,:]
            movingImage = itk.GetImageFromArray(img)
            metric = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType].New()

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

            imgs_aligned[i] = itk.GetArrayFromImage(result)


    if imgs.ndim == 4:
        ch_n = imgs.shape[1]
        ref_img = imgs[0, ref_ch, :, :]
        print("!!!")
        for i in range(timepoints_n):
            fixedImage = itk.GetImageFromArray(ref_img)
            PixelType = itk.ctype('float')
            Dimension = fixedImage.GetImageDimension()
            FixedImageType = itk.Image[PixelType, Dimension]
            MovingImageType = itk.Image[PixelType, Dimension]
            TransformType = itk.TranslationTransform[itk.D, Dimension]
            initialTransform = TransformType.New()

            optimizer = itk.RegularStepGradientDescentOptimizerv4.New(
                # LearningRate=4, #SYNCit-K論文ではこのパラメタで一度も問題なかったが、iAS Arima実験でうまくいかないことがありパラメタ変更
                # MinimumStepLength=0.0001,
                # RelaxationFactor=0.5,
                # NumberOfIterations=200)
                LearningRate=4,
                MinimumStepLength=0.0001,
                RelaxationFactor=0.5,
                NumberOfIterations=200)
            img = imgs[i,ref_ch,:,:]
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
            print(" current frame  = " + str(i))

            CompositeTransformType = itk.CompositeTransform[itk.D, Dimension]
            outputCompositeTransform = CompositeTransformType.New()
            outputCompositeTransform.AddTransform(movingInitialTransform)
            outputCompositeTransform.AddTransform(registration.GetModifiableTransform())

            # リサンプリング
            for ch in range(ch_n):
                img_ch = imgs[i, ch, :, :]
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

                imgs_aligned[i][ch] = itk.GetArrayFromImage(result)
                # if ch ==ref_ch:
                #     ref_img = imgs_aligned[i][ch] #TODO 確認　不要？

        # imgs_aligned = np.reshape(imgs_aligned, (timepoints_n, ch_n, :,:))
    print(imgs_aligned.shape)
    return imgs_aligned

###############crawl####################################
root = tkinter.Tk()
root.withdraw()
# tkinter.messagebox.showinfo('Directory Chooser','Please choose an directory with imgs')
#
root.title("Choose dir")
img_dir = tkinter.filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data")
# tkinter.messagebox.showinfo('Directory Chooser','Please choose an output directory for the aligned image')
# output_dir = tkinter.filedialog.askdirectory()
output_dir = img_dir

img_files = glob.glob(os.path.join(img_dir, "*.tif*"))
# print("number of files: %s" % len(img_files))
ref_ch =0
for idx,file in enumerate(img_files):
    if "_al" in file:
        continue
    print("filename: %s" % file)
    basename_wo_ext = os.path.splitext(os.path.basename(file))[0]
    imgs = tiff.imread(file)
    if idx==0 and imgs.ndim==4:
        # ref chを取得
        def submit():
            global ch
            ch = entry.get()  # 入力されたテキストを取得
            root.quit()  # ウィンドウを閉じる
        root = tk.Tk()  # Tkインスタンスを作成
        root.title("Ref Ch")  # ウィンドウのタイトルを設定
        entry = tk.Entry(root)  # Entry（テキストボックス）を作成
        entry.pack()
        submit_button = tk.Button(root, text="Submit", command=submit)  # "Submit"という名前のボタンを作成。ボタンが押されるとsubmit関数が呼び出される
        submit_button.pack()
        root.mainloop()  # GUIを表示
        ref_ch = int(ch)
    imgs_registered = registration_trasnlation(imgs, ref_ch)
    output_fname = os.path.join(output_dir, basename_wo_ext + "_al.tif")
    # tiff.imsave(output_fname, imgs_registered )
    if imgs.ndim==3:
        tiff.imwrite(output_fname, imgs_registered, imagej=True, metadata={'axes': 'TYX'})
    elif imgs.ndim==4:
        tiff.imwrite(output_fname, imgs_registered, imagej=True, metadata={'axes': 'TCYX'})
########################################################

print("Finished")

