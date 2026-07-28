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


class LinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.sample_size = len(x)
        # 回帰係数
        self.x_mean = np.mean(self.x)
        self.s_xx = np.sum((self.x - self.x_mean) ** 2)
        self.y_mean = np.mean(self.y)
        self.s_xy = np.sum((self.x - self.x_mean) * (self.y - self.y_mean))
        self.coef = self.s_xy / self.s_xx
        self.intercept = self.y_mean - self.coef * self.x_mean

        # 不変標本分散
        s2 = np.sum((self.y - self.intercept - self.coef * self.x) ** 2) / (self.sample_size - 2)
        self.s = np.sqrt(s2)

        # t分布(自由度N-2)の上側2.5%点
        self.t = stats.t.ppf(1 - 0.025, df=self.sample_size - 2)

        # 決定係数, 相関係数
        sy2 = np.var(self.y, ddof=0)
        d = self.y - self.coef * self.x - self.intercept
        syx2 = np.mean(d ** 2)
        sr2 = sy2 - syx2
        self.r2 = sr2 / sy2
        self.r = np.sqrt(self.r2)

    # prediction
    def predict(self, x):
        return self.intercept + self.coef * x

    # 95%信頼区間
    def calc_confidence_interval(self, x):
        band = self.t * self.s * np.sqrt(1 / self.sample_size + (x - self.x_mean) ** 2 / self.s_xx)
        upper_confidence = self.predict(x) + band
        lower_confidence = self.predict(x) - band

        return (lower_confidence, upper_confidence)

    # 95%予測区間
    def calc_prediction_interval(self, x):
        band = self.t * self.s * np.sqrt(1 + 1 / self.sample_size + (x - self.x_mean) ** 2 / self.s_xx)
        upper_confidence = self.predict(x) + band
        lower_confidence = self.predict(x) - band

        return (lower_confidence, upper_confidence)
