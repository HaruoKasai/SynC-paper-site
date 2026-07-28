from moviepy.editor import VideoFileClip, vfx
import os
import cv2
import numpy as np

# ns2文件路径
video_file_path = r"\\DESKTOP-WS2\data\Zhou\Behavior\EEG\20240710_z161_ROI-ctrl_002\_video_Contrust_FPS\z161_beforeR_004.avi"
dir = os.path.dirname(video_file_path)
output_path = os.path.join(os.path.dirname(video_file_path), "output_video.mp4")

# 加载视频
video_clip = VideoFileClip(video_file_path)

'''
# 仅使用moviepy的失败部分
# 降低对比度，这里使用colorx函数，参数可以调整以适应你的具体需求
# 例如，降低亮度和增加饱和度可以降低对比度
# adjusted_clip = video_clip.fx(video_clip, lum=-20, contrast=-0.5)
'''
def adjust_contrast(image, contrast_factor):
    """
    Adjust the contrast of an image using a linear transformation.
    :param image: Input image
    :param contrast_factor: Factor by which to adjust contrast (1.0 means no change)
    :return: Image with adjusted contrast
    """
    # Clip contrast factor to ensure it’s within a reasonable range
    contrast_factor = max(0, min(contrast_factor, 3.0))
    return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

def process_frame(frame, contrast_factor):
    # Convert frame from RGB to BGR (OpenCV format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Adjust the contrast
    adjusted_bgr = adjust_contrast(frame_bgr, contrast_factor)
    # Convert frame back from BGR to RGB
    return cv2.cvtColor(adjusted_bgr, cv2.COLOR_BGR2RGB)

# Apply contrast adjustment to each frame
contrast_factor = 0.5  # Change this factor to adjust contrast
adjusted_clip = video_clip.fl_image(lambda frame: process_frame(frame, contrast_factor))

# 调整帧率
# adjusted_clip_fps = adjusted_clip.set_fps(19.88)
# 调整帧率
new_fps = 19.88
adjusted_clip_fps = adjusted_clip.set_fps(new_fps)

# 计算新的视频时长
new_duration = adjusted_clip_fps.fps / 20 * 3600  # 原始时长按照新帧率的播放时间

# 导出视频
adjusted_clip_fps.write_videofile(output_path, codec="libx264")