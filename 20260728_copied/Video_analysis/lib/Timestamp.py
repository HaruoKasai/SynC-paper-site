from tkinter import W
import pandas as pd
import numpy as np
import lib.Timestamp
import os
import glob



def correct_video_timestamp(time_n):
    # 記録していない部分をCut１　０をカット
    invalid_vals = np.where(time_n == 0)
    if len(invalid_vals[0]) > 0:
        print("Timestamp (total len: %d) corrected for length %d" %
              (len(time_n), len(invalid_vals)))
        time_n = time_n[:np.min(invalid_vals)]

    # 記録していない部分をCut２　値が同一になっているのをカット
    tdd = np.where(np.diff(time_n) == 0)[0]
    if len(tdd) > 0:
        idx_diff = np.where(np.diff(tdd) == 1)[0]
        if len(idx_diff) > 0:
            idx = tdd[idx_diff.min()]  # 連続して出現する最小のindex
            print("Timestamp tail cut: %d" % idx)
            time_n = time_n[:idx]

    # 途中でPCの時刻設定がリセットになるとTimestampが飛ぶのを修正
    skip = np.where(np.diff(time_n) < 0)[0]
    for s in skip:
        val = time_n[s] - time_n[s + 1] + np.diff(time_n).mean()
        time_n[s + 1:] = time_n[s + 1:] + val
        print("Correcting timeslip %.2f (index: %d)" % (val, s))
    return time_n

def generate_time_series(dir_name: str, start_frame: int):
    """s
    videoにtimestampとtrialを設定
    timestamp: *timestamp.npyより

    start_frame>0の場合にはstart_frameから開始する。
    時間のサイズはvideosize。videosizeはstart_frameが削られていることを前提
    """

    df = pd.DataFrame()
    timestamp_files = glob.glob(os.path.join(dir_name, "*timestamp.npy"))

    if len(timestamp_files) == 0:
        raise ValueError("NO TIME FILE: " + dir_name)

    time_n = np.load(timestamp_files[0]).astype(
        np.float64)  # ここで64bitにしておかないと後で情報削れる
    time_n = correct_video_timestamp(time_n)
    time_n = time_n[start_frame:]  # startを適用


    # ここでtimezoneを指定するとリサンプリングでエラーになるのでリサンプル後にtimezoneを指定する
    df["time"] = pd.to_datetime(time_n, unit="s").tz_localize('Asia/Tokyo')

    df.to_csv(os.path.join(dir_name,"raw_video", "_timestamp.csv"))
    return df