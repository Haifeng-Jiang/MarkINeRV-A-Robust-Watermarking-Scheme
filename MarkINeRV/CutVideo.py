import cv2
import os

# 指定视频文件路径
video_path = "D:/HNeRV-main/data/5.mp4"#改这里切片视频

# 获取视频文件名以创建输出目录（与视频同名）
output_folder = os.path.splitext(video_path)[0]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取视频
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 循环遍历每一帧
current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()

    # 如果成功读取到一帧
    if ret:
        # 构建输出图片路径及名称
        output_image_path = os.path.join(output_folder, f"frame_{current_frame:06d}.png")  # 格式化为六位数，例如：frame_000001.jpg

        # 保存当前帧为图像
        cv2.imwrite(output_image_path, frame)

        current_frame += 1
    else:
        break  # 视频读取结束时退出循环

# 关闭视频文件
cap.release()

print(f"成功提取并保存了{current_frame}帧图片到{output_folder}目录下。")