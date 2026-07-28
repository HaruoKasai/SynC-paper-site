import pandas as pd
import re
import numpy as np
import os
from sklearn import linear_model
import glob
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.ticker as ticker
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import h5py
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# cell_nameを正規表現で抽出する関数
def extract_cell_name(cell_path):
    basename = os.path.basename(cell_path)
    match = re.search(r"N\d+", basename)  # "N"に続く数字を検索
    if match:
        return match.group(0)  # "N" + 数字 の部分を返す
    else:
        return None  # マッチしない場合はNoneを返す

def gaussian(x, a, b, c):
    return a * np.exp(-((x-b)**2) / (2 * c **2))

def calculate_AS_filler_ratio(back_filler, back_AS, prop_path, date, cell_name, df_all):
    df = pd.read_csv(prop_path)
    df = df[df['analysis'] == True]
    df = df[['label', 'area', 'mean_intensity-0', 'mean_intensity-1']][:-1]


    area = df["area"].values
    label = df['label']
    filler = (df['mean_intensity-0'].values - back_filler) * area  # 最終行を除外 (dendrite)
    AS = (df['mean_intensity-1'].values - back_AS) * area

    valid_indices = (filler > 0) & (AS > 0)
    filler = filler[valid_indices]
    AS = AS[valid_indices]
    area = area[valid_indices]
    label = label[valid_indices]



    #scoreの値を計算
    AS_filler_ratio = AS / filler

    # 体積は平均で正規化
    filler_norm = filler / np.mean(filler)

    part_name = prop_path.split('_region_')[1].split('.tif')[0]

    df_to_add = pd.DataFrame({
        'date': date,
        'cell': cell_name,
        'part': part_name,
        'label': label.values.flatten(),  # labelもflattenしておく
        'area' : area,
        'filler': filler,
        'filler_norm': filler_norm,
        'filler_mean': np.mean(filler),
        'AS': AS,
        'AS_filler_ratio': AS_filler_ratio,
    })
    df_all = pd.concat([df_all, df_to_add], ignore_index=True)
    return df_all

def get_background(h5_path, image):
    with h5py.File(h5_path, 'r') as hdf:
        bg_mean = hdf["images/"+os.path.basename(image)+"/bg_mean"] #240708_3channel_c-fos_iAS_correlation_N22_al_unmix_region_right_upper.tif/bg_mean"][:])
        filler_back = bg_mean[:][0]
        AS_back = bg_mean[:][1]
        return filler_back, AS_back

def plot_AS_filler_ratio(df, gs, ax, threshold):
    ax = fig.add_subplot(gs[ax])
    cell_name = df["cell"].unique()[0]
    X = df["filler_norm"].values
    Y = df["AS_filler_ratio"].values
    s=8
    #ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(X,Y,s=s,alpha=0.8)
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler')
    ax.set_ylim(-0.005,0.08)
    ax.set_title(cell_name)
    ax.set_xlim(0, 6)

def subtraction_AS_filler_ratio(date, cell, df_cell, df_soma, df_neurons, df_kde_all):
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]
    if df_temp.empty:
        print(f"No matching data found for date: {date}, cell: {cell}")
        return df_neurons, df_kde_all
    spine_intensity_subtract_median = (df_cell['AS'].sum() / len(df_cell['AS'])) - df_cell['AS'].median()
    data_as = df_cell['AS'].values
    kde = stats.gaussian_kde(data_as)
    x_vals = np.linspace(min(data_as), max(data_as), 1000)
    kde_vals = kde(x_vals)
    mode_value_kde = x_vals[np.argmax(kde_vals)]
    spine_intensity_mode_subtracted = df_cell['AS'].sum() / len(df_cell['AS']) - mode_value_kde
    spine_intensity = df_cell['AS'].sum() / len(df_cell['AS'])
    spine_intensity_subtract_sm = df_cell['AS'].sum() / len(df_cell['AS']) - ((df_temp['mVenus'].values[0]/df_temp['filler'].values[0]) * df_cell['filler'].mean())
    spine_AS_filler_ratio = df_cell['AS'].sum() / df_cell['filler'].sum()

    df_to_add = pd.DataFrame({
        'date': [date],
        'cell': [cell],
        'AS_soma': df_temp['mVenus'].values,
        'filler_soma': df_temp['filler'].values,
        'spine_median': df_cell['AS'].median(),
        'spine_intensity_subtract_median': [spine_intensity_subtract_median],
        'mode_value': [mode_value_kde],
        'spine_intensity_mode_subtracted': [spine_intensity_mode_subtracted],
        'spine_intensity': [spine_intensity],
        'spine_intensity_subtract_sm': [spine_intensity_subtract_sm],
        'spine_AS_filler_ratio': [spine_AS_filler_ratio],
        'spine_number': len(df_cell['AS']),
        'c-fos_nucleus': df_temp['c-fos_nucleus'].values,
        'c-fos_nucleus_norm_max': df_temp['c-fos_nucleus_norm_max'].values,
        'c-fos_nucleus_norm_median': df_temp['c-fos_nucleus_norm_median'].values,
        'c-fos_nucleus_norm_mean': df_temp['c-fos_nucleus_norm_mean'].values
    })

    df_neurons = pd.concat([df_neurons, df_to_add], ignore_index=True)

    df_kde = pd.DataFrame({
        'date':[date] * len(x_vals),
        'cell':[cell] * len(x_vals),
        'x_vals': x_vals,
        'kde_vals': kde_vals
    })

    df_kde_all = pd.concat([df_kde_all, df_kde], ignore_index=True)


    return df_neurons, df_kde_all

def analyse_AS_positive (date, cell, threshold, df_cell, df_soma, df_neurons):
    as_positive_df = df_cell[df_cell['AS_filler_ratio_mode_normalized'] > threshold]
    ASpositive_intensity_sum = as_positive_df['AS_filler_ratio_mode_normalized'].sum()
    ASpositive_percentage = len(as_positive_df) / len(df_cell) *100
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]
    for index in df_neurons.index:
        if df_neurons.loc[index, "date"] == date and df_neurons.loc[index, "cell"] == cell:
            df_neurons.loc[index, "AS(+)_intensity_sum"] = ASpositive_intensity_sum
            df_neurons.loc[index, "AS(+)_percentage"] = ASpositive_percentage
    return df_neurons

def plot_AS_filler_ratio(df, gs, ax, threshold):
    ax = fig.add_subplot(gs[ax])
    cell_name = df["cell"].unique()[0]
    X = df["filler_norm"].values
    Y = df["AS_filler_ratio_mode_normalized"].values
    s=8
    ax.axhline(y=threshold, color='k', linestyle='--')
    ax.scatter(X,Y,s=s,alpha=0.8)
    ax.set_xlabel('V (a.u)')
    ax.set_ylabel('AS/filler_normalized')
    ax.set_ylim(-0.5,10.5)
    ax.set_title(cell_name)
    ax.set_xlim(0, 6)


# 読み込んだファイルの処理
h5_path = r"\\Synology\arima\hdf5\as_proj.h5"
soma_data_path = r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_c-fos_iAS_correlatoin_correct_1slice_ROI.csv"
df_soma = pd.read_csv(soma_data_path)
day_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation", "[!_]*"))

# 全てのデータを保存するデータフレーム
df_all = pd.DataFrame(
    columns=['date', 'cell', 'part', 'label', 'area', 'filler', 'filler_norm', 'AS', 'AS_filler_ratio'])

# with h5py.File(h5_path, 'r') as hdf:
#     def print_hdf5_structure(name, obj):
#         print(name)
#
#
#     # HDF5ファイルの構造を出力
#     hdf.visititems(print_hdf5_structure)


# 各細胞ごとにループして計算を実行
for d, day in enumerate(day_list):
    dir = os.path.join(day, "_unmixing_mVenus_c-fos", "okazaki_analysis_crop")
    prop_list = glob.glob(os.path.join(dir, "*tif_props.csv"))
    cell_list = [p.split('_al_unmix')[0] for p in prop_list]
    cell_list = list(set(cell_list))

    for c, cell in enumerate(cell_list):
        date = os.path.basename(cell)[:6]
        cell_name = extract_cell_name(cell)  # 修正した部分

        # cell_nameが取得できなかった場合はスキップ
        if cell_name is None:
            print(f"cell_nameが抽出できなかったためスキップ: {cell}")
            continue

        cell_prop_list = glob.glob(cell + "_al_*unmix_*.tif_props.csv")

        # 各セルのプロパティを処理
        for prop in cell_prop_list:
            image = prop.split('_props')[0]
            back_filler, back_AS = get_background(h5_path, image)
            df_all = calculate_AS_filler_ratio(back_filler, back_AS, prop, date, cell_name, df_all)

df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all2_spine.csv")

df_neurons = pd.DataFrame(columns=['date', 'cell', 'AS_soma', 'filler_soma','spine_median', 'spine_intensity_subtract_median'
    , 'mode_value','spine_intensity_mode_subtracted', 'spine_intensity', 'spine_intensity_subtract_sm', 'spine_AS_filler_ratio', 'spine_number', 'c-fos_nucleus', 'c-fos_nucleus_norm_max', 'c-fos_nucleus_norm_median', 'c-fos_nucleus_norm_mean'])

df_kde_all = pd.DataFrame(columns=['date', 'cell', 'x_vals', 'kde_vals'])

date_list  = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]

        df_neurons, df_kde_all = subtraction_AS_filler_ratio(date, cell, df_cell, df_soma, df_neurons, df_kde_all)



#
# date と cell 毎に filler_mean を一意に取得
unique_filler_means = df_all.groupby(['date', 'cell', 'part'])['filler_mean'].mean().reset_index()
# 全体の filler_mean 平均を計算
all_filler_mean = unique_filler_means['filler_mean'].mean()
print(f"all_filler_mean: {all_filler_mean}")
# across_cell_filler_coefficient を計算して unique_filler_means に追加
unique_filler_means['across_cell_filler_coefficient'] = unique_filler_means['filler_mean'] / all_filler_mean
# 元の df_all に across_cell_filler_coefficient をマージ
df_all = df_all.merge(unique_filler_means[['date', 'cell','part', 'across_cell_filler_coefficient']], on=['date', 'cell', 'part'], how='left')
# across_spine_filler を計算して追加
df_all['across_spine_filler'] = df_all['filler'] / df_all['across_cell_filler_coefficient']
# AS_filler_ratio_filler_normalized を計算して追加
df_all['AS_filler_ratio_filler_normalized'] = df_all['AS'] / df_all['across_spine_filler']

group_iAS = df_all.groupby(['date', 'cell'])
mode_list = []
normalized_iAS_mode_list = []

for (date, cell), group in group_iAS:
    data_as = group['AS_filler_ratio_filler_normalized'].values

    kde = stats.gaussian_kde(data_as)
    x_vals = np.linspace(min(data_as), max(data_as), 1000)
    kde_vals = kde(x_vals)
    mode_value_kde = x_vals[np.argmax(kde_vals)]

    normalized_iAS_values = group['AS_filler_ratio_filler_normalized'] - mode_value_kde

    mode_list.append({
        'date': date,
        'cell': cell,
        'mode_value_kde' : mode_value_kde
    })

    group_normalized = group.copy()
    group_normalized['AS_filler_ratio_mode_normalized'] = normalized_iAS_values
    normalized_iAS_mode_list.append(group_normalized)

mode_list_df = pd.DataFrame(mode_list)
df_neurons = df_neurons.merge(mode_list_df, on=['date', 'cell'], how='left')

df_all = pd.concat(normalized_iAS_mode_list, ignore_index=True)






# データフレームを保存
df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all2_spine_normalized.csv", index=False)

df_kde_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_kde_values_summary.csv", index=False)

# 結果を CSV として保存
df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base_subtraction.csv")

# 結果をプロット
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(4, 4)
plt.rc('font', size=5)
colors = np.random.rand(len(df_neurons))

c_fos_threshold = 1
iAS_threshold = 0.4

ax = fig.add_subplot(gs[(0, 0)])
ax.scatter(df_neurons['spine_intensity_subtract_median'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity_subtract_median')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 1)])
ax.scatter(df_neurons['spine_intensity_mode_subtracted'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity_mode_subtracted')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 2)])
ax.scatter(df_neurons['spine_intensity'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('spine_intensity')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 3)])
ax.scatter(df_neurons['spine_intensity_subtract_sm'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('spine_intensity_subtract_sm')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 0)])
ax.scatter(df_neurons['c-fos_nucleus_norm_max'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('c-fos (a.u.) max_norm')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 1)])
ax.scatter(df_neurons['c-fos_nucleus_norm_median'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('c-fos (a.u.) median_norm')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(1, 2)])
ax.scatter(df_neurons['c-fos_nucleus_norm_mean'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.axhline(y=iAS_threshold, linestyle='--', lw=0.3)
ax.axvline(x=c_fos_threshold, linestyle='--', lw=0.3)
ax.set_xlabel('c-fos (a.u.) mean_norm')
ax.set_ylabel('Somatic AS intensity')

grouped = df_all.groupby(['date', 'cell'])
ax = fig.add_subplot(gs[2,1])
ratio_min = df_all['AS_filler_ratio_filler_normalized'].min()
ratio_max = df_all['AS_filler_ratio_filler_normalized'].max()

for (date, cell), group in grouped:
    plt.hist(group['AS_filler_ratio_filler_normalized'], bins=100, alpha=0.3)
    print(group['AS_filler_ratio_filler_normalized'].max())
ax.set_title('histogram ratio filler normalized')
ax.set_ylabel("Frequency")
ax.set_xlabel("AS/filler ratio")
ax.set_xlim(-0.02, 0.3)

ax = fig.add_subplot(gs[2,2])
hist, bin_edges = np.histogram(df_all['AS_filler_ratio_mode_normalized'], bins=100)
hist_max = hist.max()
hist_normalized = hist / hist_max
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

popt, _ =curve_fit(gaussian, bin_centers, hist, p0=[1, 0.5, 0.1])
a_fit, b_fit, c_fit = popt
gaussian_peak = gaussian(b_fit, *popt)
scaling_factor = 1 / gaussian_peak
a_fit_scaled = a_fit * scaling_factor

x_fit = np.linspace(0, 1, 1000)
y_fit = gaussian(x_fit, a_fit_scaled, b_fit, c_fit)
x_folded = 2 - x_fit
y_folded = gaussian(x_fit, a_fit_scaled, b_fit, c_fit)

weights = np.ones_like(df_all['AS_filler_ratio_mode_normalized']) / len(df_all['AS_filler_ratio_mode_normalized'])
ax.bar(bin_centers, hist_normalized, width=bin_edges[1] - bin_edges[0], alpha=0.7)
ax.plot(x_fit, y_fit, color='blue')
ax.plot(x_folded, y_folded, color='blue')
ax.set_title('histogram ratio mode_normalized')
ax.set_ylabel("Frequency")
ax.set_xlabel("AS/filler ratio mode normalized")
ax.set_xlim(-0.02, 0.3)

ax = fig.add_subplot(gs[2,3])
global_min = df_all['AS'].min()
global_max = df_all['AS'].max()

num_bins = 100
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
for (date, cell), group in grouped:
    ax.hist(group['AS'], bins=bin_edges, alpha=0.3, label=f'date: {date}, cell: {cell}')
ax.set_title('histogram binning across cells setting')
ax.set_ylabel("Frequency")
ax.set_xlabel("AS spine intensity (a.u.)")
ax.set_xlim(-300, 10000)

inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
inset_ax.set_xlim(4000, 10000)
inset_ax.set_ylim(0,2)
for (date, cell), group in grouped:
    inset_ax.hist(group['AS'], bins=bin_edges, alpha=0.3)

#AS(+)の基準を決める
base =
std = c_fit
threshold = 1 + 3*std
print(threshold)


date_list = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]
        df_neurons = analyse_AS_positive(date, cell, threshold, df_cell, df_soma, df_neurons)


# 結果を CSV として保存
df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base_subtraction.csv")

ax = fig.add_subplot(gs[(3,0)])
ax.scatter(df_neurons['AS(+)_intensity_sum'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine sum intensity')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(3, 1)])
ax.scatter(df_neurons['AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine (%)')
ax.set_ylabel('Somatic AS intensity')


df_neurons['c-fos_category'] = np.where(df_neurons['c-fos_nucleus_norm_mean'] < c_fos_threshold, 'c-fos(-)', 'c-fos(+)')
df_neurons['iAS_category'] = np.where(df_neurons['AS_soma'] < iAS_threshold, 'iAS(-)', 'iAS(+)')
category_counts = df_neurons.groupby(['c-fos_category', 'iAS_category']).size().unstack(fill_value=0)
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

ax = fig.add_subplot(gs[(2, 0)])
data_points = np.arange(len(category_percentages.index))
bottom_data = pd.Series(np.zeros(len(category_percentages.index)), index=category_percentages.index.tolist())
colors = {'iAS(-)': 'lightgray', 'iAS(+)': 'blue'}

for i, column in enumerate(category_percentages.columns):
    bar_list = ax.bar(data_points, category_percentages[column], bottom=bottom_data, label=column)

    for bar in bar_list:
        bar.set_color(colors[column])

    bottom_data += category_percentages[column]

ax.set_xticks(data_points)
ax.set_xticklabels(category_percentages.index)

ax.set_ylabel('Percentage (%)')
ax.legend(title='iAS Category')

# グラフ表示
plt.tight_layout()



# 結果をファイルに保存
plt.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_stacked_bar_chart_100percent.pdf", dpi=300, transparent=True)

df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\combined_c-fos_iAS_data.csv", index=False)


plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_subtraction_summary.pdf", dpi=300,
            transparent=True)


#全細胞　AS/fillerのplot
fig2 = plt.figure(figsize=(130, 50))
gs2 = gridspec.GridSpec(10, 25)
plt.rc('font', size=15)
date_list  = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]
        df_neurons = analyse_AS_positive(date, cell, threshold, df_cell, df_soma, df_neurons)
        plot_AS_filler_ratio(df_cell, gs=gs2, ax=(d,c), threshold=threshold)


plt.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_AS_filler_ratio_normalized.pdf", dpi=300, transparent=True)


