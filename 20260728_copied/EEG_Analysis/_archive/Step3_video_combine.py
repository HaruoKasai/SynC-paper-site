import os
from tkinter import Tk, filedialog
from moviepy.editor import VideoFileClip, CompositeVideoClip

# 创建一个Tkinter root窗口（隐藏）
root = Tk()
root.withdraw()

# 打开文件对话框选择文件夹
dir_path = filedialog.askdirectory(initialdir=r"\\DESKTOP-WS2\data")

# 寻找目录中的AVI文件
avi_files = [f for f in os.listdir(dir_path) if f.endswith(".avi")]

# 确保至少有一个AVI文件
if not avi_files:
    raise FileNotFoundError("No .avi files found in the selected directory.")

# 指定路径中的第一个AVI文件作为video_file_path
video_file_path = os.path.join(dir_path, avi_files[0])

# 指定_spectrum_video.mp4.avi文件路径
combined_analysis_video_file_path = os.path.join(dir_path, "combined_analysis_video.mp4")



# 加载视频剪辑
video_clip = VideoFileClip(video_file_path)
combined_analysis_video_clip = VideoFileClip(combined_analysis_video_file_path)


# 计算最终视频的尺寸
final_width = video_clip.size[0] + combined_analysis_video_clip.size[0]
final_height = combined_analysis_video_clip.size[1]

# 合成最终视频
final_clip = CompositeVideoClip([
    video_clip.set_position((0, "center")),
    combined_analysis_video_clip.set_position((video_clip.size[0], 0))], size=(final_width, final_height))

# 输出路径为选择的文件夹下的_final_video.mp4
final_output_path = os.path.join(dir_path, "_final_video.mp4")

# 写入最终视频
final_clip.write_videofile(final_output_path, codec="libx264", fps=24)