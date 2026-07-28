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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


root = tkinter.Tk()
root.withdraw()
csv = tkinter.filedialog.askopenfilename(initialdir=r"\\DESKTOP-WS2\data\sawada\CID_Analysis\_stats")
data = pd.read_csv(csv)
# columns = df.columns

#TODO　一般化したスクリプトに書き直したい

# データを長い形式に変換
long_data = (pd.melt(data,
                    # id_vars=['Mouse', 'group'],
                    id_vars=['cell', 'group'],
                    # value_vars=['-150', '-125', '-100', '-75', '-50', '-25', '0', '25', '50', '75', '100', '125', '150', '175', '200', '225', '250'],
                    # value_vars=['Trial1', 'Trial2', 'Trial3', 'Day2'],
                    value_vars=["-45",	"-25",	"-5",	"15",	"35",	"55",	"78",	"98",	"118",	"138",	"158",	"178",	"198",	"218",	"238",	"258"],
                    # value_vars=['-150', '-125', '-100', '-75', '-50', '-25'],
                    # value_vars=['-150', '-125', '-100', '-75', '-50', '-25', '0'],
                    var_name='current', value_name='value'))

# 繰り返し測定のANOVAを実行
formula = 'value ~ C(group) + C(current) + C(group):C(current)'
model = ols(formula, data=long_data).fit()
anova_table = anova_lm(model, typ=2)

print(anova_table)