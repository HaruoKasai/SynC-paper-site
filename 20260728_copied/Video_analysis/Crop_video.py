import os
import glob
import cv2
import shutil
import multiprocessing
from tkinter import filedialog
import lib.Timestamp as Timestamp


# 事前にROIを決めておき、並列で処理する
def select_roi(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Error: Failed to read {video_path}")
        return None

    h, w = 200, 300  # 任意のクロップサイズ
    x, y = (frame.shape[1] - w) // 2, (frame.shape[0] - h) // 2
    dragging = False
    x_offset, y_offset = 0, 0

    def mouse_callback(event, mx, my, flags, param):
        nonlocal x, y, dragging, x_offset, y_offset
        if event == cv2.EVENT_LBUTTONDOWN:
            if x <= mx <= x + w and y <= my <= y + h:
                dragging = True
                x_offset, y_offset = mx - x, my - y
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            x, y = mx - x_offset, my - y_offset
            x = max(0, min(x, frame.shape[1] - w))
            y = max(0, min(y, frame.shape[0] - h))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)

    while True:
        clone = frame.copy()
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Select ROI", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enterキーで確定
            break

    cv2.destroyAllWindows()
    return (x, y, w, h)


def process_video(args):
    input_path, roi, output_path, uncropped_dir = args
    x, y, w, h = roi

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 軽量なXVIDコーデックを使用
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[y:y + h, x:x + w]
        out.write(cropped_frame)

    cap.release()
    out.release()

    shutil.move(input_path, os.path.join(uncropped_dir, os.path.basename(input_path)))

    print(f"Processed: {input_path} -> {output_path}")


def main():
    mouse_dir = filedialog.askdirectory(initialdir=r"X:\Behavior\EEG\Turntable")
    exp_list = glob.glob(os.path.join(mouse_dir, "[!_]*"))

    roi_dict = {}
    video_tasks = []

    for exp in exp_list:

        if os.path.exists(os.path.join(exp, "raw_video", "_uncropped")):
            print(exp + ": Already cropped")

        else:
            Timestamp.generate_time_series(exp, start_frame=0)
            input_paths = glob.glob(os.path.join(exp, "raw_video", "*.avi"))
            if not input_paths:
                continue

            input_path = input_paths[0]
            file_name = os.path.basename(input_path)
            output_path = os.path.join(exp, "raw_video", file_name[:-4] + "_crop.avi")

            if exp not in roi_dict:
                roi = select_roi(input_path)
                if roi is None:
                    continue
                roi_dict[exp] = roi  # 各フォルダごとのROIを保存

            uncropped_dir = os.path.join(exp, "raw_video" , "_uncropped")
            os.makedirs(uncropped_dir, exist_ok=True)

            video_tasks.append((input_path, roi_dict[exp], output_path, uncropped_dir))

    # 並列処理
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 60)) as pool:
        pool.map(process_video, video_tasks)

    print("すべての動画処理が完了しました。")


if __name__ == "__main__":
    main()
