import pandas as pd
import numpy as np
import os
from scipy import stats
import glob
import tkinter.filedialog
import tkinter.messagebox
import sys
import pathlib
import itertools
from scikit_posthocs import posthoc_dunn



root = tkinter.Tk()
root.withdraw()
csv = tkinter.filedialog.askopenfilename(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\_stats\revision_1st")
df = pd.read_csv(csv)
columns = df.columns



# #Mann-Whitney (すべての組み合わせに対してやるスクリプトになっているが本来必要ない)
print("############## Mann Whitney U ########################")
combinations = itertools.combinations(columns, 2)
for combination in combinations:
    print(combination)
    group1, group2 = df[combination[0]].values,df[combination[1]].values
    group1 = group1[np.logical_not(np.isnan(group1))]
    group2 = group2[np.logical_not(np.isnan(group2))]

    statistic, p_value = stats.mannwhitneyu(group1, group2)
    print("statistic")
    print("p value")
    print(statistic)
    print(p_value)