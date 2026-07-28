import os
import glob
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from moviepy.editor import VideoFileClip


def select_video_file():
    """使用文件对话框选择视频文件"""
    file_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv;*.flv")]
    )
    return file_path


def get_clip_times():
    """弹出对话框，获取用户输入的剪辑时间"""
    start_time = simpledialog.askfloat("输入", "Start-time(sec):", minvalue=0)
    end_time = simpledialog.askfloat("输入", "End-time(sec):", minvalue=0)
    return start_time, end_time


def cut_video(file_path, start_time, end_time):
    """剪辑视频并保存"""
    if not os.path.exists(file_path):
        messagebox.showerror("错误", "指定的视频文件不存在")
        return

    video = VideoFileClip(file_path)
    fps = video.fps  # 获取原视频 FPS

    if start_time < 0 or end_time > video.duration or start_time >= end_time:
        messagebox.showerror("错误", "无效的剪辑时间段")
        return

    cut_clip = video.subclip(start_time, end_time)
    base_name, ext = os.path.splitext(os.path.basename(file_path))

    # 处理 AVI 视频的编码格式
    if ext.lower() == ".avi":
        output_filename = f"_cut_{int(start_time)}_{int(end_time)}.avi"
        codec = "mpeg4"
        bitrate = "2000k"  # 提高码率，防止模糊
    else:
        output_filename = f"_cut_{int(start_time)}_{int(end_time)}.mp4"
        codec = "libx264"

    output_path = os.path.join(os.path.dirname(file_path), output_filename)

    # 保持原 FPS，确保音频编码
    cut_clip.write_videofile(output_path, codec=codec, fps=fps, bitrate=bitrate, preset="veryslow")

    messagebox.showinfo("成功", f"视频已保存至: {output_path}")

def main():
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口

    # 选择视频文件
    file_path = select_video_file()
    if not file_path:
        messagebox.showerror("错误", "未选择任何视频文件")
        return

    # 获取剪辑时间
    start_time, end_time = get_clip_times()
    if start_time is None or end_time is None:
        messagebox.showerror("错误", "无效的时间输入")
        return

    cut_video(file_path, start_time, end_time)


if __name__ == "__main__":
    main()
