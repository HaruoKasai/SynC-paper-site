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
from scipy.stats import linregress
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.cluster import KMeans
import h5py
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42


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

def analyse_AS_positive (date, cell, filler_norm_threshold, df_cell, df_soma, df_neurons):
    as_positive_filler_norm_df = df_cell[df_cell['AS_filler_ratio_filler_normalized'] > filler_norm_threshold]
    ASpositive_filler_norm_intensity_sum = as_positive_filler_norm_df['AS'].sum()
    ASpositive_filler_norm_percentage = len(as_positive_filler_norm_df) / len(df_cell) *100
    as_positive_subtraction_zscore_above_gaussian_fit_df = df_cell[df_cell['z_scores'] > subtraction_mode_threshold]
    ASpositive_subtraction_zscore_above_gaussian_fit_percentage = len(as_positive_subtraction_zscore_above_gaussian_fit_df) / len(df_cell) * 100
    ASpositive_subtraction_zscore_above_gaussian_fit_sum = as_positive_subtraction_zscore_above_gaussian_fit_df['AS'].sum()
    as_positive_subtraction_zscore_above_3SD_df = df_cell[df_cell['z_scores'] > z_score_3SD_threshold]
    ASpositive_subtraction_zscore_above_3SD_percentage = len(as_positive_subtraction_zscore_above_3SD_df) / len(df_cell) * 100
    df_soma['date'] = df_soma['date'].astype(str)
    df_temp = df_soma[(df_soma['date'] == date) & (df_soma['cell'] == cell)]


    for index in df_neurons.index:
        if df_neurons.loc[index, "date"] == date and df_neurons.loc[index, "cell"] == cell:
            df_neurons.loc[index, "filler_norm_AS(+)_intensity_sum"] = ASpositive_filler_norm_intensity_sum
            df_neurons.loc[index, "filler_norm_AS(+)_percentage"] = ASpositive_filler_norm_percentage
            df_neurons.loc[index, "subtraction_zscore_above_gaussian_fit_ratio_AS(+)_percentage"] = ASpositive_subtraction_zscore_above_gaussian_fit_percentage
            df_neurons.loc[index, "subtraction_zscore_above_3SD_ratio_AS(+)_percentage"] = ASpositive_subtraction_zscore_above_3SD_percentage
            df_neurons.loc[index, "subtraction_zscore_above_gaussian_fit_ratio_AS(+)_sum"] = ASpositive_subtraction_zscore_above_gaussian_fit_sum
    return df_neurons

def plot_AS_filler_ratio(df, gs, ax, threshold):
    ax = fig.add_subplot(gs[ax])
    cell_name = df["cell"].unique()[0]
    X = df["filler_norm"].values
    Y = df["AS_filler_ratio_filler_normalized"].values
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

# mean_across_cell_filler = (
#     unique_filler_means.groupby(['date', 'cell'])['across_cell_filler_coefficient'].mean().reset_index().rename(
#         columns={'across_cell_filler_coefficient': 'mean_across_cell_filler_coefficient'}
#     )
# )
# df_neurons = df_neurons.merge(mean_across_cell_filler, on=['date', 'cell'], how='left')
# df_neurons['AS/filler_ratio_soma_across_cell'] = (df_neurons['AS_soma'] / (df_neurons['filler_soma'] / df_neurons['mean_across_cell_filler_coefficient']))


group_iAS = df_all.groupby(['date', 'cell'])
mode_list = []
subtraction_mode_list = []

for (date, cell), group in group_iAS:
    data_as = group['AS_filler_ratio_filler_normalized'].values

    kde = stats.gaussian_kde(data_as)
    x_vals = np.linspace(min(data_as), max(data_as), 100)
    kde_vals = kde(x_vals)
    AS_filler_ratio_mode_value_kde = x_vals[np.argmax(kde_vals)]


    subtraction_mode_AS_intensity = group['AS'].values - (group['across_spine_filler'].values * AS_filler_ratio_mode_value_kde)
    subtraction_mode_AS_filler_ratio = subtraction_mode_AS_intensity / group['across_spine_filler'].values
    print(f"subtraction_mode_AS_intensity shape: {subtraction_mode_AS_intensity.shape}")
    print(f"subtraction_mode_AS_filler_ratio shape: {subtraction_mode_AS_filler_ratio.shape}")



    mode_list.append({
        'date': date,
        'cell': cell,
        'after_filler_normalized_ratio_mode': AS_filler_ratio_mode_value_kde
    })

    for label, part, intensity, ratio in zip(group['label'], group['part'], subtraction_mode_AS_intensity, subtraction_mode_AS_filler_ratio):
        subtraction_mode_list.append({
            'date': date,
            'cell': cell,
            'label': label,
            'part': part,
            'subtraction_mode_AS_intensity': intensity,
            'subtraction_mode_AS_filler_ratio': ratio
        })


mode_list_df = pd.DataFrame(mode_list)
df_neurons = df_neurons.merge(mode_list_df, on=['date', 'cell'], how='left')
subtraction_mode_df = pd.DataFrame(subtraction_mode_list)
print(df_all[['date', 'cell', 'part', 'label']].duplicated().sum())
print(subtraction_mode_df[['date', 'cell', 'part', 'label']].duplicated().sum())
df_all = df_all.merge(subtraction_mode_df, on=['date', 'cell', 'part', 'label'], how='left')


df_kde_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_kde_values_summary.csv", index=False)

# 結果を CSV として保存
df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base_subtraction.csv")

# 結果をプロット
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(4, 4)
plt.rc('font', size=5)
colors = np.random.rand(len(df_neurons))

c_fos_threshold = 0.3
iAS_threshold = 0.2


ax = fig.add_subplot(gs[(0, 0)])
ax.scatter(df_neurons['spine_intensity_mode_subtracted'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity_mode_subtracted')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 1)])
ax.scatter(df_neurons['spine_intensity'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
           cmap='viridis')
ax.set_xlabel('spine_intensity')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 2)])
ax.scatter(df_neurons['filler_soma'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Somatic filler intensity')
ax.set_ylabel('Somatic AS intensity')

ax = fig.add_subplot(gs[(0, 3)])
# ax.scatter(df_neurons['c-fos_nucleus_norm_mean'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors,
#            cmap='viridis')
# y_lines = [0.2]
# for y in y_lines:
#     ax.axhline(y=y, linestyle='--', lw=0.3)
# ax.axvline(x=c_fos_threshold, linestyle='--', lw=0.3)
# ax.set_xlabel('c-fos (a.u.)')
# ax.set_ylabel('Somatic AS intensity')
# x_tick = [0, 0.3, 0.9, 2.7]
# ax.set_xticks(x_tick)
# y_tick = [0.2, 0.4, 0.6]
# ax.set_yticks(y_tick)

# ax = fig.add_subplot(gs[(1,0)])
X = df_neurons['c-fos_nucleus_norm_mean']
Y = df_neurons['AS_soma']
pearson_corr, pearson_p_value = pearsonr(X,Y)
spearman_corr, spearman_p_value = spearmanr(X,Y)
print(f"ピアソン相関係数: {pearson_corr:.4f}, p値: {pearson_p_value:.4e}")
print(f"スピアマン順位相関係数: {spearman_corr:.4f}, p値: {spearman_p_value:.4e}")
# 線形回帰を計算
slope, intercept, r_value, p_value, std_err = linregress(X, Y)

# 回帰線を描画するための x 値の範囲を決定
x_vals = np.linspace(min(X), max(X), 1000)
y_vals = slope * x_vals + intercept
ax.plot(x_vals, y_vals,linestyle='--', color='red', label=f'Linear fit: y = {slope:.3f}x + {intercept:.3f}\n$r^2$ = {r_value**2:.3f}')
ax.axvline(x=c_fos_threshold, linestyle='--', lw=0.3)
y_lines = [0.2]
for y in y_lines:
    ax.axhline(y=y, linestyle='--', lw=0.3)
#ax.text(0.05, 0.9, f'$R^2$ = {r_value**2:.3f}', transform=ax.transAxes, fontsize=10, color='red')
ax.scatter(X,Y, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('c-fos (a.u.)')
ax.set_ylabel('Somatic AS intensity')
x_tick = [0, 0.3, 0.9, 2.7]
ax.set_xticks(x_tick)
y_tick = [0.2, 0.4, 0.6]
ax.set_yticks(y_tick)

grouped = df_all.groupby(['date', 'cell'])
ax = fig.add_subplot(gs[1,1])
global_min = df_all['AS_filler_ratio_filler_normalized'].min()
global_max = df_all['AS_filler_ratio_filler_normalized'].max()

num_bins = 100
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
for (date, cell), group in grouped:
    ax.hist(group['AS_filler_ratio_filler_normalized'], bins=bin_edges, alpha=0.3, label=f'date: {date}, cell: {cell}')
ax.set_title('histogram AS/filler ratio')
ax.set_ylabel("Frequency")
ax.set_xlabel("AS/filler ratio (filler norm)")
#ax.set_xlim(-300, 10000)

inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
#inset_ax.set_xlim(4000, 10000)
inset_ax.set_ylim(0,10)
for (date, cell), group in grouped:
    inset_ax.hist(group['AS_filler_ratio_filler_normalized'], bins=bin_edges, alpha=0.3)


ax = fig.add_subplot(gs[1,2])
global_min = df_all['subtraction_mode_AS_filler_ratio'].min()
global_max = df_all['subtraction_mode_AS_filler_ratio'].max()

num_bins = 100
bin_edges = np.linspace(global_min, global_max, num_bins + 1)
for (date, cell), group in grouped:
    ax.hist(group['subtraction_mode_AS_filler_ratio'], bins=bin_edges, alpha=0.3, label=f'date: {date}, cell: {cell}')
ax.set_title('histogram subtraction AS/filler ratio')
ax.set_ylabel("Frequency")
ax.set_xlabel("subtraction_mode_AS_filler_ratio")
#ax.set_xlim(-300, 10000)

inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
#inset_ax.set_xlim(4000, 10000)
inset_ax.set_ylim(0,10)
for (date, cell), group in grouped:
    inset_ax.hist(group['subtraction_mode_AS_filler_ratio'], bins=bin_edges, alpha=0.3)
    inset_ax.set_xlim(0.05, 0.28)
    inset_ax.set_ylim(0, 5)

ax = fig.add_subplot(gs[1,3])
kde = stats.gaussian_kde(df_all['subtraction_mode_AS_filler_ratio'])
x_vals = np.linspace(min(df_all['subtraction_mode_AS_filler_ratio']), max(df_all['subtraction_mode_AS_filler_ratio']), 1000)
kde_vals = kde(x_vals)
mode_value_kde = x_vals[np.argmax(kde_vals)]

hist_all, bin_edges_all = np.histogram(df_all['subtraction_mode_AS_filler_ratio'], bins=300, density=True)
hist_max_all = hist_all.max()
hist_normalized_all = hist_all / hist_max_all
bin_centers_all = 0.5 * (bin_edges_all[1:] + bin_edges_all[:-1])

# 左側データを抽出
filtered_data = df_all['subtraction_mode_AS_filler_ratio'][df_all['subtraction_mode_AS_filler_ratio'] <= mode_value_kde]
fold_data = mode_value_kde + np.abs(mode_value_kde - filtered_data)
symmetric_data = np.concatenate([filtered_data, fold_data])
# ヒストグラムの計算
hist, bin_edges = np.histogram(symmetric_data, bins=300, density=True)
hist_max = hist.max()
hist_normalized = hist / hist_max
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
print(max(hist))

# ガウスフィッティング
popt, _ = curve_fit(gaussian, bin_centers, hist, p0=[max(hist), mode_value_kde, np.std(symmetric_data)])
a_fit, b_fit, c_fit = popt

gaussian_peak = gaussian(b_fit, *popt)
scaling_factor = 1 / gaussian_peak
a_fit_scaled = a_fit * scaling_factor
print(mode_value_kde)

# 左側フィッティング結果
x_fit = np.linspace(min(symmetric_data), max(df_all['subtraction_mode_AS_filler_ratio']), 300)
y_fit = gaussian(x_fit, a_fit, b_fit, c_fit)

SD = c_fit
print(f'gaussian fit SD :{SD} ')

weights = np.ones_like(df_all['subtraction_mode_AS_filler_ratio']) / len(df_all['subtraction_mode_AS_filler_ratio'])
ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)
ax.plot(x_fit, y_fit / max(y_fit), color='blue', label="Gaussian Fit (Left)")
ax.set_title('histogram gaussian fitting')
ax.set_ylabel("Frequency")
ax.set_xlabel("AS/filler ratio subtraction mode")

inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
#inset_ax.set_xlim(4000, 10000)
inset_ax.set_ylim(0, 0.01)
inset_ax.bar(bin_centers_all, hist_normalized_all, width=bin_edges_all[1] - bin_edges_all[0], alpha=0.7)
inset_ax.set_xlim(0.04, 0.27)

ax = fig.add_subplot(gs[2,0])
# 横軸をz_scoreのhistogramにするために変換;zscoreの定義よりmodeの値が0なのでそこまでのデータでまたgaussian fitを行う
z_scores = (df_all['subtraction_mode_AS_filler_ratio'] - mode_value_kde) / c_fit
filtered_zscore_data = z_scores[z_scores <= 0]
folded_zscore_data = np.abs(filtered_zscore_data)
symmetric_zscore_data = np.concatenate([filtered_zscore_data, folded_zscore_data])

bin_width_z0 = 0.25
bin_edges_z0 = np.arange(min(symmetric_zscore_data), max(symmetric_zscore_data) + bin_width_z0, bin_width_z0)
#gussian fit scale after zscore normalization
hist_z0, bin_edges_z0 = np.histogram(symmetric_zscore_data, bins=bin_edges_z0)
hist_z0_max = hist_z0.max()
hist_z0_normalized = hist_z0 / hist_z0_max
bin_z0_centers = 0.5 * (bin_edges_z0[1:] + bin_edges_z0[:-1])

popt, _ = curve_fit(gaussian, bin_z0_centers, hist_z0, p0=[max(hist_z0), np.median(symmetric_zscore_data), 2*np.std(symmetric_zscore_data)],
                    maxfev=10000)
a_fit_z0, b_fit_z0, c_fit_z0 = popt

gaussian_peak = gaussian(b_fit_z0, *popt)
scaling_factor = 1 / gaussian_peak
a_fit_z0_scaled = a_fit_z0 * scaling_factor

x_fit_z0 = np.linspace(min(z_scores) - 1, max(z_scores), 1000)
y_fit_z0 = gaussian(x_fit_z0, a_fit_z0_scaled, b_fit_z0, c_fit_z0)

bin_edges_zscores = np.arange(min(z_scores), max(z_scores) + bin_width_z0, bin_width_z0)
hist_zscores, bin_zscores = np.histogram(z_scores, bins=bin_edges_zscores)
hist_zscores_max = hist_zscores.max()
hist_zscores_normalized = hist_zscores / hist_zscores_max
bin_zscores_centers = 0.5 * (bin_zscores[1:] + bin_zscores[:-1])

weights = np.ones_like(z_scores) / len(z_scores)
ax.bar(bin_zscores_centers, hist_zscores_normalized, width=bin_width_z0, alpha=0.7)
ax.plot(x_fit_z0, y_fit_z0 / max(y_fit_z0), color='blue', label="Gaussian Fit (Left)")
ax.set_title(f"zscore_histogram and gaussian fit")
ax.set_xlabel("zscore")
ax.set_ylabel("frequency")
#ax.set_xlim(-10, 330)

inset_ax = inset_axes(ax, width='40%', height='40%', loc="upper right")
#inset_ax.set_xlim(4000, 10000)
inset_ax.set_ylim(0, 0.01)
inset_ax.bar(bin_zscores_centers, hist_zscores_normalized, width=bin_width_z0, alpha=0.7)
inset_ax.set_xlim(20, 110)

ax = fig.add_subplot(gs[2,1])
threshold = b_fit_z0 + c_fit_z0 * 3
print(f"b_fit + 3*c_fit_threshold: {threshold}")
zscores_above_threshold = z_scores[z_scores >= threshold]
bin_width_zscore_above = bin_width_z0 * 10
bin_edges_zscore_above = np.arange(threshold, max(z_scores) + bin_width_zscore_above, bin_width_zscore_above)
hist_zscores_above, bin_zscores_above = np.histogram(zscores_above_threshold, bins=bin_edges_zscore_above)

max_frequency = max(hist_zscores.max(), hist_zscores_above.max()/10)
# print(f"max_hist_zscores:{hist_zscores.max()}")
# print(f"max_hist_zscore_above:{hist_zscores_above.max()}")
# print(max_frequency)
hist_zscore_normalized = hist_zscores  / max_frequency
hist_zscores_above_normalized = (hist_zscores_above / (bin_width_zscore_above/bin_width_z0)) / max_frequency
bin_zscores_above_centers = 0.5 * (bin_zscores_above[1:] + bin_zscores_above[:-1])
# print(f"hist_zscore_normalized:{hist_zscore_normalized}")
# print(f"hist_zscores_above_normalized:{hist_zscores_above_normalized}")
# print(f"bin_width_z0: {bin_width_z0}")
# print(f"bin_width_zscore_above:{bin_width_zscore_above}")
# for i in range (len(hist_zscores)):
#     print(f"hist_zscores_Bin {i+1} ({bin_zscores[i]:.2f} to {bin_zscores[i+1]:.2f}): {hist_zscores[i]} elements")
# for j in range(len(hist_zscores_above)):
#     print(f"hist_zscores_above_Bin {j+1} ({bin_zscores_above[j]:.2f} to {bin_zscores_above[j+1]:.2f}): {hist_zscores_above[j]} elements")

ax.bar(bin_zscores_centers, hist_zscore_normalized, width=bin_width_z0, alpha=0.7)
ax.bar(bin_zscores_above_centers, hist_zscores_above_normalized, width=bin_width_zscore_above, alpha=0.7)
ax.plot(x_fit_z0, y_fit_z0 / max(y_fit_z0), color='blue', label="Gaussian Fit (Left)")
ax.set_title(f"hist and gaussian fit ")
ax.set_xlabel("zscore")
ax.set_ylabel("frequency")

# 縦軸を対数表記に設定
ax.set_yscale('log')

# 必要に応じて軸の範囲を設定（例）
ax.set_ylim(1e-4, 1e0)

df_all['z_scores'] = z_scores

# データフレームを保存
df_all.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_all2_spine_normalized.csv", index=False)

#AS(+)の基準を決める
filler_norm_threshold = df_all['AS_filler_ratio_filler_normalized'].mean() + 2.5 * df_all['AS_filler_ratio_filler_normalized'].std()
subtraction_mode_threshold = b_fit_z0 + c_fit_z0 * 3
z_score_3SD_threshold = df_all['z_scores'].mean() + 2.5 * df_all['z_scores'].std()

print(F"filler_norm_threshold: {filler_norm_threshold}")
print(f"subtration_mode_threshold: {subtraction_mode_threshold}")
print(f"z_score_3SD_threshold: {z_score_3SD_threshold}")
soma_iAS_threshold = df_neurons['AS_soma'].median() + 2*df_neurons['AS_soma'].std()
print(f'soma_iAS_threshold :{soma_iAS_threshold}')





date_list = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]
        df_neurons = analyse_AS_positive(date, cell, filler_norm_threshold, df_cell, df_soma, df_neurons)


# 結果を CSV として保存
df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_cell_base_subtraction.csv")

ax = fig.add_subplot(gs[(2,2)])
ax.scatter(df_neurons['subtraction_zscore_above_gaussian_fit_ratio_AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('above_gaussian fit spine (%)')
ax.set_ylabel('Somatic AS intensity')
ax.set_title('AS/filler ratio subtraction')

ax = fig.add_subplot(gs[(2, 3)])
ax.scatter(df_neurons['filler_norm_AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine (%)')
ax.set_ylabel('Somatic AS intensity')
ax.set_title('AS/filler ratio filler norm')

ax = fig.add_subplot(gs[(3, 0)])
ax.scatter(df_neurons['subtraction_zscore_above_3SD_ratio_AS(+)_percentage'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine (%)')
ax.set_ylabel('Somatic AS intensity')
ax.set_title('AS/filler ratio subtraction')

ax = fig.add_subplot(gs[(3, 1)])
ax.scatter(df_neurons['subtraction_zscore_above_gaussian_fit_ratio_AS(+)_sum'].values, df_neurons['AS_soma'].values, s=3, alpha=0.8, c=colors, cmap='viridis')
ax.set_xlabel('Positive spine sum (a.u.)')
ax.set_ylabel('Somatic AS intensity')
ax.set_title('AS/filler ratio subtraction')

# ax = fig.add_subplot(gs[3,2])
# bin_width = 0.09
# bins= np.arange(min(df_neurons['AS_soma']), 0.8, bin_width)
# hist, bin_edges = np.histogram(df_neurons['AS_soma'], bins=8, density=True)
# hist_max = hist.max()
# hist_normalized = hist / hist_max
# bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
#
#
# ax.bar(bin_centers, hist_normalized, width=bin_edges[1] - bin_edges[0], alpha=0.7)
# ax.plot(bin_centers, hist_normalized, color='red')
# ax.set_title(f'histogram_AS_soma')
# ax.set_xlabel("AS_soma (a.u.)")

# ax = fig.add_subplot(gs[3,1])
data = df_neurons['AS_soma']
# data1 = df_neurons['AS_soma'].values.reshape(-1,1)
# kmeans = KMeans(n_clusters=2, random_state=42)
# kmeans.fit(data1)
# # クラスタリングの結果を取得
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# # クラスタラベルを「negative」と「positive」にマッピング
# mapped_labels = ['negative' if label == 0 else 'positive' for label in labels]
#
# ax.scatter(data1, np.zeros_like(data1), c=labels, cmap='viridis', alpha=0.5)


# ax = fig.add_subplot(gs[3,])
# kde = gaussian_kde(data)
# x_values = np.linspace(min(data), max(data), 1000)
# y_values = kde(x_values)


# ax.plot(x_values, y_values, label="Density (KDE)", color='red')
# ax = fig.add_subplot(gs[3,3])
# sns.violinplot(data=data, inner="point", color='skyblue', ax=ax)
# ax.set_title("Violin Plot for Bimodal Data")
# ax.set_xlabel("Data Distribution")


df_neurons['c-fos_category'] = np.where(df_neurons['c-fos_nucleus_norm_mean'] < c_fos_threshold, 'c-fos(-)', 'c-fos(+)')
df_neurons['iAS_category'] = np.where(df_neurons['AS_soma'] < iAS_threshold, 'iAS(-)', 'iAS(+)')
category_counts = df_neurons.groupby(['c-fos_category', 'iAS_category']).size().unstack(fill_value=0)
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100


# ax = fig.add_subplot(gs[(2, 0)])
# data_points = np.arange(len(category_percentages.index))
# bottom_data = pd.Series(np.zeros(len(category_percentages.index)), index=category_percentages.index.tolist())
# colors = {'iAS(-)': 'lightgray', 'iAS(+)': 'blue'}
#
# for i, column in enumerate(category_percentages.columns):
#     bar_list = ax.bar(data_points, category_percentages[column], bottom=bottom_data, label=column)
#
#     for bar in bar_list:
#         bar.set_color(colors[column])
#
#     bottom_data += category_percentages[column]
#
# ax.set_xticks(data_points)
# ax.set_xticklabels(category_percentages.index)
#
# ax.set_ylabel('Percentage (%)')
# ax.legend(title='iAS Category')

fig3, ax = plt.subplots(figsize=(8,6))
print(df_neurons[['c-fos_category', 'iAS_category']].head())
print(df_neurons['c-fos_category'].unique())
print(df_neurons['iAS_category'].unique())
# Combined category列の生成
df_neurons['combined_category'] = df_neurons['c-fos_category'] + ', ' + df_neurons['iAS_category']

# カテゴリの順序を指定
category_order = ['c-fos(-), iAS(-)', 'c-fos(+), iAS(-)', 'c-fos(+), iAS(+)', 'c-fos(-), iAS(+)']

# groupbyでcountを計算
category_counts = df_neurons.groupby('combined_category').size().reset_index(name='count')

# カテゴリが欠けている場合にゼロ埋め
category_counts = category_counts.set_index('combined_category').reindex(category_order, fill_value=0).reset_index()

# 全体に対する割合を計算
total_count = category_counts['count'].sum()
category_counts['percentage'] = category_counts['count'] / total_count * 100

# デバッグ情報の出力
print("Combined category counts:")
print(category_counts)

# 色の設定
colors = {
    'c-fos(-), iAS(-)': 'lightgray',
    'c-fos(+), iAS(-)': 'blue',
    'c-fos(-), iAS(+)': 'orange',
    'c-fos(+), iAS(+)': 'green'
}

left = 0  # 積み上げ棒グラフの開始位置
for index, row in category_counts.iterrows():
    ax.barh(
        ['Combined Categories'],  # 1つの棒グラフを描画
        row['percentage'],
        color=colors[row['combined_category']],
        edgecolor='black',
        label=row['combined_category'],
        left=left  # 現在の開始位置
    )
    left += row['percentage']  # 次のカテゴリの開始位置を更新

# ラベルと凡例を追加
#ax.set_ylabel('Percentage (%)')
ax.set_title('Distribution of Combined Categories')
ax.legend(title='Category', loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
fig3.tight_layout()
fig3.savefig(rf"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_stacked_bar_chart_100percent_iAS_{iAS_threshold:.2f}_c-fos_{c_fos_threshold:.2f}.pdf", dpi=300, transparent=True)



# c-fos(+) / c-fos(-) のカテゴリーを作成
df_neurons['c-fos_category'] = df_neurons['c-fos_nucleus_norm_mean'].apply(lambda x: 'c-fos(+)' if x >= 0.3 else 'c-fos(-)')
df_neurons['iAS_category'] = df_neurons['AS_soma'].apply(lambda x: 'iAS(+)' if x >= 0.2 else 'iAS(-)')

# 4条件ごとのAS(+)スパイン割合を計算
grouped_df = df_neurons.groupby(['c-fos_category', 'iAS_category'])['filler_norm_AS(+)_percentage'].mean().reset_index()

# 4条件の順番を指定
category_order = ['c-fos(+), iAS(+)', 'c-fos(-), iAS(+)', 'c-fos(+), iAS(-)', 'c-fos(-), iAS(-)']
grouped_df['combined_category'] = grouped_df['c-fos_category'] + ', ' + grouped_df['iAS_category']
grouped_df = grouped_df.set_index('combined_category').reindex(category_order, fill_value=0).reset_index()

# 色の設定
colors = {
    'c-fos(+), iAS(+)': 'green',
    'c-fos(-), iAS(+)': 'orange',
    'c-fos(+), iAS(-)': 'blue',
    'c-fos(-), iAS(-)': 'lightgray'
}

# 棒グラフの作成
ax = fig.add_subplot(gs[3,2])
ax.bar(grouped_df['combined_category'], grouped_df['filler_norm_AS(+)_percentage'],
       color=[colors[cat] for cat in grouped_df['combined_category']], edgecolor='black')

# グラフの装飾
ax.set_ylabel('AS(+) spine percentage (%)')
ax.set_title('AS(+) spine percentage')
mean_values = grouped_df['filler_norm_AS(+)_percentage'].tolist()
# std_values = group_df['filler_norm_AS(+)_percentage'].std()
ax.set_ylim(0, max(mean_values) * 1.2)  # Y軸の範囲を適切に設定
for i, v in enumerate(mean_values):
    ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)

# # c_fos(+) と c_fos(-) に分類
# cfos_positive_df = df_neurons[df_neurons['c-fos_category'] == 'c-fos(+)']
# cfos_negative_df = df_neurons[df_neurons['c-fos_category'] == 'c-fos(-)']
#
# cfos_positive_mean_iAS = cfos_positive_df['filler_norm_AS(+)_percentage'].mean()
# cfos_negative_mean_iAS = cfos_negative_df['filler_norm_AS(+)_percentage'].mean()

# # グループごとのデータをリストにまとめる
# categories = ['c-fos(+)', 'c-fos(-)']
# mean_values = [cfos_positive_mean_iAS, cfos_negative_mean_iAS]
#
# # 棒グラフの作成
# ax = fig.add_subplot(gs[3,3])
# ax.bar(categories, mean_values, color=['blue', 'lightgray'], edgecolor='black')
# # グラフの装飾
# ax.set_ylabel('AS(+) spine percentage (%)')
# ax.set_title('AS(+) spine percentage vs c-fos')
# ax.set_ylim(0, max(mean_values) * 1.2)  # Y軸の範囲を適切に設定
# for i, v in enumerate(mean_values):
#     ax.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontsize=10)




contingency_table = pd.crosstab(df_neurons['c-fos_category'], df_neurons['iAS_category'])
print("Contingency Table:")
print(contingency_table)

chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
print(f"\nカイ二乗検定の結果:")
print(f"カイ二乗統計量: {chi2_stat:.4f}")
print(f"p値: {p_val:.4e}")
print(f"自由度: {dof}")
print(f"期待値:\n{expected}")

# フィッシャーの正確確率検定
if contingency_table.shape == (3, 2):
    _, fisher_p_val = fisher_exact(contingency_table)
    print(f"\nフィッシャーの正確確率検定の結果:")
    print(f"p値: {fisher_p_val:.4e}")
else:
    print("\nフィッシャーの正確確率検定は 2x2 のテーブルでのみ使用可能です。")
# グラフ表示
plt.tight_layout()



# 結果をファイルに保存


df_neurons.to_csv(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\combined_c-fos_iAS_data.csv", index=False)


fig.tight_layout()
fig.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_subtraction_summary.pdf", dpi=300,
            transparent=True)


#全細胞　AS/fillerのplot
fig2 = plt.figure(figsize=(130, 50))
gs2 = gridspec.GridSpec(10, 25)
plt.rc('font', size=15)
date_list = df_all['date'].unique()
for d, date in enumerate(date_list):
    df_date = df_all[df_all['date'] == date]
    cell_list = df_date['cell'].unique()
    for c, cell in enumerate(cell_list):
        df_cell = df_date[df_date['cell'] == cell]
        try:
            df_neurons = analyse_AS_positive(date, cell, filler_norm_threshold, df_cell, df_soma, df_neurons)
            plot_AS_filler_ratio(df_cell, gs=gs2, ax=(d, c), threshold=filler_norm_threshold)
        except Exception as e:
            print(f"Error in processing date: {date}, cell: {cell}. Error: {e}")
            continue


fig2.tight_layout()
fig2.savefig(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\_unmixing_c-fos_iAS_correlation\_AS_filler_ratio_normalized.pdf", dpi=300, transparent=True)


