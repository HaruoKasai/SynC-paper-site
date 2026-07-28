import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import glob
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_ransac_statistics(spine_csv, back_csv):
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    back_df = pd.read_csv(back_csv)

    # Drop NaN values
    spine_df = spine_df.dropna()
    back_df = back_df.dropna()

    # 'area' 列を優先し、存在しない場合は 'area_in_pixel' を使用
    area_column = 'area' if 'area' in spine_df.columns else 'area_in_pixel'

    # 選択された列を使用
    area = spine_df[area_column].values

    label = spine_df['label']
    sp_minus_back_x_area = (spine_df['mean_intensity-0'].values - back_df['mean_intensity-0'].values) * area
    sp_minus_back_y_area = (spine_df['mean_intensity-1'].values - back_df['mean_intensity-1'].values) * area

    # 最終行の値を保存（dendrite）
    last_x_value = sp_minus_back_x_area[-1]
    last_y_value = sp_minus_back_y_area[-1]

    # 最終行を除外
    sp_minus_back_x_area = sp_minus_back_x_area[:-1]
    sp_minus_back_y_area = sp_minus_back_y_area[:-1]

    # Use the first time point column
    X_normalized = sp_minus_back_x_area.reshape(-1, 1) / np.mean(sp_minus_back_x_area)
    y_normalized = sp_minus_back_y_area.reshape(-1, 1) / np.mean(sp_minus_back_y_area)
    X = sp_minus_back_x_area.reshape(-1, 1)
    y = sp_minus_back_y_area.reshape(-1, 1)

    # Drop NaN values in X and y
    non_nan_indices = np.logical_and(~np.isnan(X_normalized), ~np.isnan(y_normalized))
    X_normalized = X_normalized[non_nan_indices].reshape(-1, 1)
    y_normalized = y_normalized[non_nan_indices].reshape(-1, 1)
    X = X[non_nan_indices].reshape(-1, 1)
    y = y[non_nan_indices].reshape(-1, 1)

    ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False),
                                          min_samples=0.1,
                                          random_state=42,
                                          stop_probability=0.99,
                                          max_trials=1000,
                                          loss='squared_error')

    ransac.fit(X_normalized, y_normalized)

    # Calculate the ratio of y_normalized to the RANSAC regression curve
    ratio_y_ransac = y_normalized / ransac.predict(X_normalized)

    # Use all data, not just the top 5% as threshold
    selected_values = ratio_y_ransac.flatten()

    # Save RANSAC scores and y values to CSV
    ransac_scores_df = pd.DataFrame({'label': label[:-1],
                                     'RANSAC Score': selected_values,
                                     'filler_sum': X.flatten(),
                                     'mVenus_sum': y.flatten()})
    ransac_csv_path = os.path.join(dir, f'ransac_scores_{fname[:-9]}.csv')
    ransac_scores_df.to_csv(ransac_csv_path, index=False)

    return X_normalized, ratio_y_ransac, selected_values



root = tk.Tk()
root.withdraw()
dir = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate")
spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
print("Number of files: %s" % len(spine_csv_files))

# All RANSAC and AS scores
control_ransac_scores = []
apv_ransac_scores = []
dendrite_ransac_scores = []
apv_dendrite_ransac_scores = []


# データセットごとに処理を行う
control_X_normalized = []
control_ratio_y_ransac = []

apv_X_normalized = []
apv_ratio_y_ransac = []


for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
    if "control" in spine_csv.lower():
        X_norm, ratio_y_ransac, ransac_scores = calculate_ransac_statistics(spine_csv, back_csv)
        # 最後の行をdendriteに分ける
        dendrite_ransac_scores.append(ransac_scores[-1])
        control_ransac_scores.extend(ransac_scores[:-1])
        control_X_normalized.extend(X_norm)
        control_ratio_y_ransac.extend(ratio_y_ransac)


    elif "apv" in spine_csv.lower():
        X_norm, ratio_y_ransac, ransac_scores = calculate_ransac_statistics(spine_csv, back_csv)
        # 最後の行をapv dendriteに分ける
        apv_dendrite_ransac_scores.append(ransac_scores[-1])
        apv_ransac_scores.extend(ransac_scores[:-1])
        apv_X_normalized.extend(X_norm)
        apv_ratio_y_ransac.extend(ratio_y_ransac)



# Prepare DataFrame for plotting and CSV writing
control_ransac_scores_df = pd.DataFrame(control_ransac_scores, columns=['RANSAC Score'])
control_ransac_scores_df['Group'] = 'Control'
apv_ransac_scores_df = pd.DataFrame(apv_ransac_scores, columns=['RANSAC Score'])
apv_ransac_scores_df['Group'] = 'APV'
dendrite_ransac_scores_df = pd.DataFrame(dendrite_ransac_scores, columns=['RANSAC Score'])
dendrite_ransac_scores_df['Group'] = 'Control Dendrite'
apv_dendrite_ransac_scores_df = pd.DataFrame(apv_dendrite_ransac_scores, columns=['RANSAC Score'])
apv_dendrite_ransac_scores_df['Group'] = 'APV Dendrite'

# Write to CSV
control_csv_path = os.path.join(dir, 'control_ransac_scores.csv')
apv_csv_path = os.path.join(dir, 'apv_ransac_scores.csv')
dendrite_csv_path = os.path.join(dir, 'dendrite_ransac_scores.csv')
apv_dendrite_csv_path = os.path.join(dir,'apv_dendrite_ransac_scores.csv')

control_ransac_scores_df.to_csv(control_csv_path, index=False)
apv_ransac_scores_df.to_csv(apv_csv_path, index=False)
dendrite_ransac_scores_df.to_csv(dendrite_csv_path, index=False)
apv_dendrite_ransac_scores_df.to_csv(apv_dendrite_csv_path, index=False)

# Combine DataFrames for plotting
combined_ransac_df = pd.concat([dendrite_ransac_scores_df, apv_dendrite_ransac_scores_df, apv_ransac_scores_df, control_ransac_scores_df])


# Create the output directory if it doesn't exist
output_dir = os.path.join(dir, 'graph')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Plot RANSAC box plot and strip plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Group', y='RANSAC Score', data=combined_ransac_df, whis=np.inf, linewidth=1.5)
sns.stripplot(x='Group', y='RANSAC Score', data=combined_ransac_df, jitter=True, color='black', alpha=0.5)
plt.ylabel("RANSAC Score")
plt.title('Dendrite, APV Dendrite, APV, and Control RANSAC Scores Box Plot with Individual Data Points')
plt.savefig(os.path.join(output_dir, 'ransac_boxplot_dendrite_apv_control.pdf'), dpi=300, transparent=True)
plt.show()


# Plot scatter plots for control and APV (RANSAC)
plt.figure(figsize=(4, 6))
plt.scatter(control_X_normalized, control_ratio_y_ransac, alpha=0.5, label='Control')
plt.xlabel('v (a.u)')
plt.ylabel('RANSAC Ratio')
plt.ylim(0, 35)
plt.title('Control RANSAC Score Scatter Plot')
plt.legend()
plt.savefig(os.path.join(output_dir, 'control_ransac_scatter_plot.pdf'), dpi=300, transparent=True)
plt.show()

plt.figure(figsize=(4, 6))
plt.scatter(apv_X_normalized, apv_ratio_y_ransac, alpha=0.5, label='APV')
plt.xlabel('v (a.u)')
plt.ylabel('RANSAC Ratio')
plt.ylim(0, 35)
plt.title('APV RANSAC Score Scatter Plot')
plt.legend()
plt.savefig(os.path.join(output_dir, 'apv_ransac_scatter_plot.pdf'), dpi=300, transparent=True)
plt.show()


print("Finished")

