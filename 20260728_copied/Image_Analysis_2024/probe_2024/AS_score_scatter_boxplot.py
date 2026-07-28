import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import glob
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_as_score(spine_csv, back_csv):
    print(spine_csv)
    dir = os.path.dirname(os.path.dirname(spine_csv))
    fname = os.path.basename(spine_csv)
    spine_df = pd.read_csv(spine_csv)
    print(len(spine_df))
    spine_df = spine_df[spine_df['dendrite'].notna()].reset_index(drop=True)
    print(len(spine_df))
    print(spine_df)
    back_df = pd.read_csv(back_csv)[:len(spine_df)]

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

    # Normalize to make the last row (or mean of all data) equal to 1
    norm_factor = last_y_value / last_x_value
    as_score = (sp_minus_back_y_area / sp_minus_back_x_area) / norm_factor
    X_normalized = sp_minus_back_x_area / np.mean(sp_minus_back_x_area)

    # Save AS scores and y values to CSV
    as_scores_df = pd.DataFrame({'label': label[:-1],
                                 'AS Score': as_score,
                                 'filler_sum': sp_minus_back_x_area,
                                 'mVenus_sum': sp_minus_back_y_area})
    as_csv_path = os.path.join(dir, f'as_scores_{fname[:-9]}.csv')
    as_scores_df.to_csv(as_csv_path, index=False)

    return X_normalized, as_score  # X_normalized and as_score (excluding the last row for dendrite groups)

root = tk.Tk()
root.withdraw()
probe_list = glob.glob(os.path.join(r"\\DESKTOP-WS2\data\Probe_paper_2023\dissociate\UTR_screening_data", "[!_]*"))
for probe in probe_list:
    dir = os.path.join(probe, "violin_plot_box_plot")
    spine_csv_files = glob.glob(os.path.join(dir, "*spine.csv"))
    back_csv_files = [spine_csv[:-9] + "back.csv" for spine_csv in spine_csv_files]
    print("Number of files: %s" % len(spine_csv_files))

    # All AS scores
    control_as_scores = []
    apv_as_scores = []

    # データセットごとに処理を行う
    control_as_X = []
    control_as_Y = []
    apv_as_X = []
    apv_as_Y = []

    for spine_csv, back_csv in zip(spine_csv_files, back_csv_files):
        if "control" in spine_csv.lower():
            as_X, as_scores = calculate_as_score(spine_csv, back_csv)
            control_as_X.extend(as_X)
            control_as_Y.extend(as_scores)
            control_as_scores.extend(as_scores)
        elif "apv" in spine_csv.lower():
            as_X, as_scores = calculate_as_score(spine_csv, back_csv)
            apv_as_X.extend(as_X)
            apv_as_Y.extend(as_scores)
            apv_as_scores.extend(as_scores)

    # Prepare DataFrame for plotting and CSV writing
    control_as_scores_df = pd.DataFrame(control_as_scores, columns=['AS Score'])
    control_as_scores_df['Group'] = 'Control'
    apv_as_scores_df = pd.DataFrame(apv_as_scores, columns=['AS Score'])
    apv_as_scores_df['Group'] = 'APV'

    # Write to CSV
    control_as_csv_path = os.path.join(dir, 'control_as_scores.csv')
    apv_as_csv_path = os.path.join(dir, 'apv_as_scores.csv')

    control_as_scores_df.to_csv(control_as_csv_path, index=False)
    apv_as_scores_df.to_csv(apv_as_csv_path, index=False)

    # Combine DataFrames for plotting
    combined_as_df = pd.concat([apv_as_scores_df, control_as_scores_df])  # dendrite groups are not included




    # Create the output directory if it doesn't exist
    output_dir = os.path.join(dir, 'graph')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    # Plot AS box plot and strip plot (excluding dendrite groups)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Group', y='AS Score', data=combined_as_df, whis=np.inf, linewidth=1.5)
    sns.stripplot(x='Group', y='AS Score', data=combined_as_df, jitter=True, color='black', alpha=0.5)
    plt.ylabel("AS Score")
    plt.title('APV and Control AS Scores Box Plot with Individual Data Points')
    plt.savefig(os.path.join(output_dir, 'as_boxplot_apv_control.pdf'), dpi=300, transparent=True)

    #Z-score defined by APV distribution
    APVmean = apv_as_scores_df['AS Score'].mean()
    APVstd = apv_as_scores_df['AS Score'].std()
    control_as_scores_df["z-score"] = (control_as_scores_df['AS Score'] - APVmean) / APVstd
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Group', y='z-score', data=control_as_scores_df, whis=np.inf, linewidth=1.5)
    sns.stripplot(x='Group', y='z-score', data=control_as_scores_df, jitter=True, color='black', alpha=0.5)
    plt.ylabel("Z Score (defined by APV group distribution)")
    plt.ylim(-10, 50)
    plt.savefig(os.path.join(output_dir, 'asscore_zscore_box.pdf'), dpi=300, transparent=True)



    # Plot scatter plots for control and APV (AS)
    plt.figure(figsize=(4, 6))
    plt.scatter(control_as_X, control_as_Y, alpha=0.5, label='Control')
    plt.xlabel('filler_sum (a.u)')
    plt.ylabel('AS Score')
    plt.ylim(0, 35)  # Adjust the y-axis limit as needed
    plt.title('Control AS Score Scatter Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'control_as_scatter_plot.pdf'), dpi=300, transparent=True)

    plt.figure(figsize=(4, 6))
    plt.scatter(apv_as_X, apv_as_Y, alpha=0.5, label='APV')
    plt.xlabel('filler_sum (a.u)')
    plt.ylabel('AS Score')
    plt.ylim(0, 35)  # Adjust the y-axis limit as needed
    plt.title('APV AS Score Scatter Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'apv_as_scatter_plot.pdf'), dpi=300, transparent=True)

    print("Finished")