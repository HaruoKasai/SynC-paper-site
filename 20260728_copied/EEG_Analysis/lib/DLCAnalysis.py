import h5py
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob

def get_roi_coordinate(target_name, param_ind):
    with open(param_ind, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if target_name == "arena_box":
        coordinate = data["video_param"]["arena_box"]
    else:
        coordinate =  data["video_param"]["roi"][target_name]
    return coordinate

def extract_frame_around_roi (df, coordinate, body_part="snout", distance_to_boundary_px=150):
    x, y, w, h = coordinate[0][0], coordinate[0][1], coordinate[1][0], coordinate[1][1]
    d =distance_to_boundary_px
    condition = (df[(body_part, 'x')] >= x-d) & (df[(body_part, 'x')] <= x+w+d) & \
                (df[(body_part, 'y')] >= y-d) & (df[(body_part, 'y')] <= y+h+d)
    frame_around_roi = df.index[condition]
    return frame_around_roi.to_numpy()

def frame_to_sec(frame_list, real_frame_time, event_df, event_name, start_time, tolerable_frame_drop = 0, min_duration=1): #, exp_duration=600)
    if frame_list.size>0:
        start, end = frame_list[0], frame_list[0]
        for i in range (1, len(frame_list)):
            if frame_list[i] <=frame_list[i-1]+1+tolerable_frame_drop:
                end = frame_list[i]
            else:
                if (end+1-start)*real_frame_time>min_duration and start*real_frame_time>=0: #and (end+1)*real_frame_time<exp_duration:
                    event_df.loc[len(event_df)] = [start*real_frame_time + start_time, (end+1)*real_frame_time+start_time,event_name]
                start, end = frame_list[i], frame_list[i]
        if (end+1-start)*real_frame_time>min_duration and start*real_frame_time>0: #and (end+1)*real_frame_time<exp_duration:
            event_df.loc[len(event_df)] = [start*real_frame_time+start_time, (end+1)*real_frame_time+start_time,event_name] #append the last range
    return event_df

def frame_to_sec_v2(frame_list, real_frame_time, event_df, event_name, before_sec=0, after_sec=0, exp_duration=600): #各eventが1 frameで検出されている場合
    for i in range(len(frame_list)):
        if frame_list[i]*real_frame_time + before_sec>0 and frame_list[i]*real_frame_time+after_sec<exp_duration:
            event_df.loc[len(event_df)] = [frame_list[i]*real_frame_time+before_sec, frame_list[i]*real_frame_time+after_sec,event_name]
    return event_df

def plot_time_series (df, tname, yname, title, exp_duration,fig, gs, ax, ylim, twinx=False):
    ax = fig.add_subplot(gs[ax])
    if twinx:
        ax.yaxis.set_visible(False)
        ax =ax.twinx()
        color ="#ff7f0e"
        title_loc = "right"
    else:
        color="#1f77b4"
        title_loc = "center"
    ax.plot(df[tname], df[yname], color =color)
    ax.set_title(title, color = color, loc = title_loc)
    ax.margins(x=0, y=0)
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(0, exp_duration)

def time_series_distance_to_object (df, object_coordinate, real_frame_time, arena_mm_per_pix, body_part = "snout", distance_to_boundary_mm = 100, plot=True):
    obj_x = object_coordinate[0][0] + 0.5*object_coordinate[1][0]
    obj_y = object_coordinate[0][1] + 0.5 * object_coordinate[1][1]
    df = df[[col for col in df.columns if col[0] == body_part]]
    df.columns = [col[1] for col in df.columns]
    df = df.astype('float64')
    df['Distance_to_object'] = np.sqrt((df['x'] - obj_x)**2 + (df['y'] - obj_y)**2)*arena_mm_per_pix
    # df['time'] = df.index * real_frame_time

    d = distance_to_boundary_mm
    roll=200 #20 fps * 10 sec
    condition_1 = (df['Distance_to_object'] < d) & (df['Distance_to_object'].rolling(window=roll, min_periods=roll).min().shift(1) >= d) #現在距離d未満、かつ、直前10秒に距離d以上
    condition_2 = (df['Distance_to_object'] < d) & (df['Distance_to_object'].rolling(window=roll, min_periods=roll).min().shift(-roll) >= d)
    frame_approaching = df.index[condition_1]
    frame_leaving = df.index[condition_2]
    return df['Distance_to_object'].to_numpy(), frame_approaching.to_numpy(), frame_leaving.to_numpy()

def time_series_distance_to_object_v2 (df,object_coordinate,real_frame_time, arena_mm_per_pix, d1, d2, interval, body_part = "centroid"): #d1, d2 (mm)
    obj_x = object_coordinate[0][0] + 0.5*object_coordinate[1][0]
    obj_y = object_coordinate[0][1] + 0.5 * object_coordinate[1][1]
    df = df[[col for col in df.columns if col[0] == body_part]]
    df.columns = [col[1] for col in df.columns]
    df = df.astype('float64')
    df['Distance_to_object'] = np.sqrt((df['x'] - obj_x)**2 + (df['y'] - obj_y)**2)*arena_mm_per_pix
    # df['time'] = df.index * real_frame_time
    distance_to_object = df['Distance_to_object'].to_numpy()


    #Hysterisis thresholding
    max_frames = int(interval/real_frame_time)  # interval秒をフレーム数に変換
    d1_indices = np.where(distance_to_object <= d1)[0]
    d2_indices = np.where(distance_to_object <= d2)[0]
    approach_indices = []
    d2_idx_ptr = 0  # d2 インデックスのポインタ

    for d1_idx in d1_indices:
        # d2 のインデックスを d1_idx 以降で探す
        while d2_idx_ptr < len(d2_indices) and d2_indices[d2_idx_ptr] < d1_idx:
            d2_idx_ptr += 1  # d1 より前の d2 をスキップ

        # 条件を満たす d2 インデックスがあるかチェック
        if d2_idx_ptr < len(d2_indices) and d2_indices[d2_idx_ptr] <= d1_idx + max_frames:
            approach_indices.append(d2_indices[d2_idx_ptr])
            d2_idx_ptr += 1  # 一度マッチしたら次へ進む（d1 に対して 1 回のみ）
    print(approach_indices)
    return distance_to_object, approach_indices

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(dot_product / (magnitude_v1 * magnitude_v2))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def time_series_angle_to_object_direction (df, object_coordinate, real_frame_time, exp_duration,fig, ax, gs):
    obj_x = object_coordinate[0][0] + 0.5 * object_coordinate[1][0]
    obj_y = object_coordinate[0][1] + 0.5 * object_coordinate[1][1]

    df = df[[col for col in df.columns if col[0] == "head_center" or col[0] == "body_center"]]
    df = df.astype('float64')

    angles = []
    for index, row in df.iterrows():
        vector_bc_to_hc = np.array([
            row[('head_center', 'x')] - row[('body_center', 'x')],
            row[('head_center', 'y')] - row[('body_center', 'y')]
        ])
        vector_bc_to_fixed = np.array([
            obj_x - row[('body_center', 'x')],
            obj_y - row[('body_center', 'y')]
        ])

        angle = calculate_angle(vector_bc_to_hc, vector_bc_to_fixed)
        angles.append(angle)

    # Add the angles as a new column to the DataFrame
    df[('Analysis', 'Angle_to_object_direction')] = angles
    df[('Analysis', 'time')] = df.index * real_frame_time
    df = df[[col for col in df.columns if col[0] == "Analysis"]]
    df.columns = [col[1] for col in df.columns]
    df['Angle_to_object_direction'] = df['Angle_to_object_direction'].rolling(window=21, min_periods=1, center=True).mean() #およそ1秒くらいのVelocityの移動平均

    plot_time_series(df, "time", "Angle_to_object_direction", "Angle_to_object_direction", exp_duration,fig, gs, ax, (0,180))

def time_series_velocity (df,real_frame_time, arena_mm_per_pix, body_part = "body_center", velocity_boundary=[5,25,200]):
    df = df[[col for col in df.columns if col[0] == body_part]]
    df.columns = [col[1] for col in df.columns]
    df = df.astype('float64')

    roll = 4 #ひとまず。あまり数字にロジックはない
    df['x_avg_prev'] = df['x'].shift(-roll).rolling(roll, min_periods=1).mean()
    df['x_avg_next'] = df['x'].shift(1).rolling(roll,min_periods=1).mean()
    df['y_avg_prev'] = df['y'].shift(-roll).rolling(roll,min_periods=1).mean()
    df['y_avg_next'] = df['y'].shift(1).rolling(roll,min_periods=1).mean()

    df['Velocity'] = (np.sqrt((df['x_avg_prev'] - df['x_avg_next']) ** 2 +
                              (df['y_avg_prev'] - df['y_avg_next']) ** 2)
                      * arena_mm_per_pix / real_frame_time / (roll + 1))

    df['Velocity'] = df['Velocity'].fillna(method='bfill').fillna(method='ffill')

    df['Cumulative_distance'] = (df['Velocity'] * real_frame_time / 1000).cumsum()  # unit: meter
    df['time'] = (df.index + 0.5) * real_frame_time

    # velocityごとにframeを分類
    window_size = 1
    rolling_mean = df['Velocity'].rolling(window=window_size, center=True).mean()
    min = 0
    frames_list = []
    for i in range(len(velocity_boundary)):
        frames = df.index[(rolling_mean >= min) & (rolling_mean < velocity_boundary[i])]
        frames_list.append(frames.to_numpy())
        min = velocity_boundary[i]
    return df["Velocity"].to_numpy(), df["Cumulative_distance"], frames_list, df["likelihood"].to_numpy()

########################################################################################################################################
