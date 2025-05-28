# 读取视频文件, 加载出关键帧
# 关键帧提取
# 关键帧保存到本地
import os

import cv2
import numpy as np


class VideoKeyframeExtractor:
    def __init__(self, video_path, interval=1):
        """
        初始化视频关键帧提取器

        Args:
            video_path (str): 视频文件路径
        """
        self.video_path = video_path
        self.cap = None
        self.keyframes = []
        self.interval = interval

    def load_video(self):
        """
        读取视频文件

        Returns:
            bool: 是否成功加载视频
        """
        if not os.path.exists(self.video_path):
            print(f"错误：视频文件 {self.video_path} 不存在")
            return False

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"错误：无法打开视频文件 {self.video_path}")
            return False

        print(f"成功加载视频：{self.video_path}")
        return True

    def extract_keyframes(self, method="content_change", threshold=30.0):
        """
        提取关键帧

        Args:
            method (str): 提取方法，可选值: "content_change", "uniform", "brightness"
            threshold (float): 阈值，用于判断帧变化是否足够大

        Returns:
            list: 关键帧列表
        """
        if self.cap is None and not self.load_video():
            return []

        self.keyframes = []
        prev_frame = None
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"总帧数: {total_frames}, FPS: {fps}")

        if method == "uniform":
            # 均匀采样：每隔一定数量的帧选择一个关键帧
            interval = int(fps * self.interval)  # 每5秒选择一帧

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    self.keyframes.append((frame_count, frame))
                    print(f"提取关键帧: {frame_count}/{total_frames}")

                frame_count += 1

        elif method == "brightness":
            # 基于亮度变化选择关键帧
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if prev_frame is None:
                    self.keyframes.append((frame_count, frame))
                    prev_frame = frame
                else:
                    # 计算亮度差异
                    curr_brightness = np.mean(frame)
                    prev_brightness = np.mean(prev_frame)
                    diff = abs(curr_brightness - prev_brightness)

                    if diff > threshold:
                        self.keyframes.append((frame_count, frame))
                        print(f"提取关键帧: {frame_count}/{total_frames}, 亮度变化: {diff:.2f}")
                        prev_frame = frame

                frame_count += 1

        else:  # default: content_change
            # 基于内容变化选择关键帧
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                if prev_frame is None:
                    self.keyframes.append((frame_count, frame))
                    prev_frame = frame
                else:
                    # 计算帧之间的差异(转为灰度图像进行比较)
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # 计算两帧之间的差异
                    diff = cv2.absdiff(prev_gray, curr_gray)
                    diff_score = np.mean(diff)

                    if diff_score > threshold:
                        self.keyframes.append((frame_count, frame))
                        print(f"提取关键帧: {frame_count}/{total_frames}, 差异程度: {diff_score:.2f}")
                        prev_frame = frame

                frame_count += 1

        self.cap.release()
        print(f"共提取 {len(self.keyframes)} 个关键帧")
        return self.keyframes

    def save_keyframes(self, output_dir="keyframes"):
        """
        保存关键帧到本地

        Args:
            output_dir (str): 输出目录

        Returns:
            list: 保存的文件路径列表
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not self.keyframes:
            print("没有关键帧可保存")
            return []

        saved_paths = []
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]

        for idx, (frame_idx, frame) in enumerate(self.keyframes):
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)
            print(f"保存关键帧 {idx+1}/{len(self.keyframes)}: {filepath}")

        print(f"所有关键帧已保存到目录: {output_dir}")
        return saved_paths

    def process_video(self, method="content_change", threshold=30.0, output_dir="keyframes"):
        """
        处理视频：加载、提取关键帧并保存

        Args:
            method (str): 提取方法
            threshold (float): 阈值
            output_dir (str): 输出目录

        Returns:
            list: 保存的文件路径列表
        """
        if self.load_video():
            self.extract_keyframes(method, threshold)
            return self.save_keyframes(output_dir)
        return []


# 使用示例
if __name__ == "__main__":
    extractor = VideoKeyframeExtractor(r"E:\桌面\demo\dataset\进球视频录制\VID_20250408_193434.mp4", interval=0.5)

    # 方法1：一次性处理
    res = extractor.process_video(
        method="uniform",  # 可选: "content_change", "uniform", "brightness"
        threshold=10.0,  # 差异阈值，数值越大筛选越严格
        output_dir="output_keyframes",
    )
    print(f"提取到的关键帧文件: {res}")
    # 方法2：单独调用每个步骤
    # extractor.load_video()
    # extractor.extract_keyframes(method="content_change", threshold=30.0)
    # extractor.save_keyframes(output_dir="output_keyframes")
