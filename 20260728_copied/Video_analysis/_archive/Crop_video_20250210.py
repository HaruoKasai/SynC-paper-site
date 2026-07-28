import subprocess
from tkinter import filedialog
import glob
import os
import cv2
import shutil


#TODO はじめに全部ROI選択して、動画保存はあとでまとめて行うようにする

mouse_dir = filedialog.askdirectory(initialdir=r"X:\Behavior\EEG\Turntable")
exp_list = glob.glob(os.path.join(mouse_dir, "[!_]*"))
for exp in exp_list:
    # raw_video_dir = os.path.join(exp, "raw_video")
    input_path = glob.glob(os.path.join(exp, "*.avi"))[0]
    file_name = os.path.basename(input_path)
    output_path = os.path.join(exp, file_name[:-4]+"_crop.avi")

    cap = cv2.VideoCapture(input_path)
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read video.")
        cap.release()
        exit()

    # 事前に指定するクロップのサイズ
    w, h = 300, 200  # 任意の幅・高さを指定

    # 初期位置を中央に設定
    x, y = (frame.shape[1] - w) // 2, (frame.shape[0] - h) // 2

    # マウスドラッグで矩形を移動するための変数
    dragging = False
    x_offset, y_offset = 0, 0


    def mouse_callback(event, mx, my, flags, param):
        """ マウス操作でクロップ矩形を動かす """
        global x, y, dragging, x_offset, y_offset

        if event == cv2.EVENT_LBUTTONDOWN:  # マウス押下でドラッグ開始
            if x <= mx <= x + w and y <= my <= y + h:  # 矩形内なら移動開始
                dragging = True
                x_offset, y_offset = mx - x, my - y

        elif event == cv2.EVENT_MOUSEMOVE:  # マウス移動時
            if dragging:
                x, y = mx - x_offset, my - y_offset
                # 範囲外に出ないように制限
                x = max(0, min(x, frame.shape[1] - w))
                y = max(0, min(y, frame.shape[0] - h))

        elif event == cv2.EVENT_LBUTTONUP:  # マウスを離したらドラッグ終了
            dragging = False


    # GUIウィンドウを開く
    cv2.namedWindow("Move Crop Region")
    cv2.setMouseCallback("Move Crop Region", mouse_callback)

    print("マウスで矩形を動かし、Enterキーを押して確定")

    while True:
        clone = frame.copy()
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Move Crop Region", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enterキーで確定
            break

    cv2.destroyAllWindows()

    print(f"選択された範囲: x={x}, y={y}, w={w}, h={h}")

    # 動画のフレームレートを取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # コーデックと出力ファイル設定
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # H.264 が動かない場合は MJPG
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 動画をクロップして保存
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y:y + h, x:x + w]
        out.write(cropped_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # copy raw file to "_uncropped"
    uncropped_dir = os.path.join(exp, "_uncropped")
    os.makedirs(uncropped_dir, exist_ok=True)
    shutil.move(input_path, os.path.join(uncropped_dir, file_name))

    print(f"クロップした動画を {output_path} に保存しました。")