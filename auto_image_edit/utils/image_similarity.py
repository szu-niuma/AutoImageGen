import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from skimage.metrics import structural_similarity as ssim

import torch
import lpips

from .image_sift import ImageSift


class ImageSimilarity:
    LPIPS_MODEL = lpips.LPIPS(net="vgg", spatial=True)

    @staticmethod
    def load_image_array(img: Union[Image.Image, str, Path, np.ndarray], color_space: str = "Lab") -> np.ndarray:
        # 支持 numpy array
        if isinstance(img, np.ndarray):
            arr = img.copy()
            # 假设输入的numpy数组是BGR格式（如cv2读取的）
            input_format = "BGR"
        # 支持 PIL Image
        elif isinstance(img, Image.Image):
            # PIL图像转换为RGB格式的numpy数组
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img)
            input_format = "RGB"
        # 支持 网络 URL / Base64 / 本地路径
        elif isinstance(img, str) or isinstance(img, Path):
            img = str(img)  # 确保是字符串类型
            parsed = urlparse(img)
            # 网络 URL
            if parsed.scheme in ("http", "https"):
                resp = requests.get(img)
                if resp.status_code != 200:
                    raise ValueError(f"无法加载网络图像: {img}")
                img_pil = Image.open(BytesIO(resp.content)).convert("RGB")
                arr = np.array(img_pil)
                input_format = "RGB"
            # Base64 Data URI
            elif img.strip().startswith("data:"):
                header, b64 = img.split(",", 1)
                img_pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                arr = np.array(img_pil)
                input_format = "RGB"
            # 本地文件路径
            else:
                arr = cv2.imread(img, cv2.IMREAD_COLOR)
                if arr is None:
                    raise ValueError(f"无法加载图像: {img}")
                input_format = "BGR"
        else:
            raise ValueError("未知的图像类型。支持 PIL.Image.Image、str(URL/Base64/Path)、Path、np.ndarray。")

        # 统一颜色空间转换
        color_space = color_space.upper()

        if color_space == "HSV":
            if input_format == "RGB":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            else:  # BGR
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
        elif color_space == "RGB":
            if input_format == "BGR":
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            # 如果已经是RGB，不需要转换
        elif color_space == "YCBCR":
            if input_format == "RGB":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
            else:  # BGR
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
        elif color_space == "GRAY":
            if input_format == "RGB":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            else:  # BGR
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif color_space == "BGR":
            if input_format == "RGB":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            # 如果已经是BGR，不需要转换
        elif color_space == "LAB" or color_space == "CIELAB":
            if input_format == "RGB":
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            else:  # BGR
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)

        return arr

    @staticmethod
    def compare_images_mse(
        src: Union[Image.Image, str, Path, np.ndarray],
        tgt: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        color_space: str = "RGB",
        norm: str = "zscore",
        heatmap: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """基于逐像素 MSE 的差异图计算。"""
        sa = ImageSimilarity.load_image_array(src, color_space).astype(np.float32)
        ta = ImageSimilarity.load_image_array(tgt, color_space).astype(np.float32)
        if sa.shape != ta.shape:
            raise ValueError("源图像和目标图像尺寸不一致。")

        # 计算逐像素 MSE，并归一化到 [0,1]
        mse_map = np.mean((sa - ta) ** 2, axis=-1) / (255.0**2)
        diff_gray = ImageSimilarity.norm_method(mse_map, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        return diff_gray

    @staticmethod
    def compare_images_pixelwise(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        color_space: str = "HSV",
        norm: str = "zscore",
        heatmap: bool = False,
        gray: bool = False,
        align: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        src_array = ImageSimilarity.load_image_array(src_img, color_space)
        target_array = ImageSimilarity.load_image_array(target_img, color_space)

        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不匹配。")

        if align:
            src_array, target_array, _ = ImageSift.align_images(src_array, target_array)

        color_space_upper = color_space.upper()

        if color_space_upper == "HSV":
            # HSV 颜色空间处理
            sa = src_array.astype(np.int16)
            ta = target_array.astype(np.int16)

            # 确定H通道的最大值
            h_max = 179.0 if src_array.dtype == np.uint8 else 360.0

            # Hue 通道差值，考虑环绕性
            h_diff = np.abs(sa[:, :, 0] - ta[:, :, 0])
            h_diff_wrapped = np.minimum(h_diff, h_max - h_diff)
            h_norm = h_diff_wrapped * (255.0 / (h_max / 2.0))  # 缩放到0-255范围

            # S、V 通道差值并归一化
            s_diff = np.abs(sa[:, :, 1] - ta[:, :, 1])
            v_diff = np.abs(sa[:, :, 2] - ta[:, :, 2])

            # 综合H、S、V三通道差异，可以考虑加权
            diff_gray = h_norm + s_diff + v_diff
            # diff_gray = (h_norm * 0.5 + s_diff * 0.3 + v_diff * 0.2)

        elif color_space_upper in ["LAB", "CIELAB"]:
            src_lab = src_array.astype(np.float32)
            target_lab = target_array.astype(np.float32)

            # 统一的LAB数据归一化策略
            # 检查并转换L通道到0-100范围
            if src_lab[:, :, 0].max() > 100:
                src_lab[:, :, 0] = src_lab[:, :, 0] * 100.0 / 255.0
            if target_lab[:, :, 0].max() > 100:
                target_lab[:, :, 0] = target_lab[:, :, 0] * 100.0 / 255.0

            # 检查并转换a,b通道到-128到+127范围
            # 如果最小值>=0，说明可能是0-255范围
            if src_lab[:, :, 1:].min() >= 0:
                src_lab[:, :, 1:] = src_lab[:, :, 1:] - 128.0
            if target_lab[:, :, 1:].min() >= 0:
                target_lab[:, :, 1:] = target_lab[:, :, 1:] - 128.0

            # 计算Delta E
            delta_l = src_lab[:, :, 0] - target_lab[:, :, 0]
            delta_a = src_lab[:, :, 1] - target_lab[:, :, 1]
            delta_b = src_lab[:, :, 2] - target_lab[:, :, 2]
            delta_e = np.sqrt(delta_l**2 + delta_a**2 + delta_b**2)

            max_delta_e = 10
            diff_gray = np.clip(delta_e / max_delta_e, 0, 1)

        elif color_space_upper == "RGB":
            # RGB 颜色空间处理
            diff = cv2.absdiff(src_array, target_array).astype(np.float32)
            diff_gray = diff.mean(axis=2)

        elif color_space_upper == "YCBCR":
            # YCbCr 颜色空间处理
            diff = cv2.absdiff(src_array, target_array).astype(np.float32)
            diff_gray = diff.mean(axis=2)

        elif color_space_upper == "GRAY":
            # 灰度图像处理
            if src_array.ndim == 3:
                # 如果是3通道，转换为灰度（这里假设是RGB格式）
                src_array = cv2.cvtColor(src_array, cv2.COLOR_RGB2GRAY)
            if target_array.ndim == 3:
                target_array = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
            diff_gray = cv2.absdiff(src_array, target_array).astype(np.float32)

        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

        # 归一化处理
        diff_gray = ImageSimilarity.norm_method(diff_gray, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)

        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        elif gray:
            return ImageSimilarity.to_gray(diff_gray)

        return diff_gray

    @staticmethod
    def compare_images_ssim(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        norm: str = "zscore",
        heatmap: bool = False,
        gray: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        src_array = ImageSimilarity.load_image_array(src_img)
        target_array = ImageSimilarity.load_image_array(target_img)
        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不一致。")

        # 转换到灰度
        src_gray = cv2.cvtColor(src_array, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_array, cv2.COLOR_RGB2GRAY)
        _, diff_map = ssim(src_gray, target_gray, full=True)
        # 将相似度映射转为差异（0—1）
        diff_map = 1.0 - diff_map
        # 归一化
        diff_gray = ImageSimilarity.norm_method(diff_map, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        elif gray:
            return ImageSimilarity.to_gray(diff_gray)
        return diff_gray

    @staticmethod
    def check_and_resize(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
    ) -> Tuple[Image.Image, Image.Image]:
        """
        对 src_img 和 target_img 同时先放大到两者最大尺寸，再缩小到两者最小尺寸，
        最终输出相同大小的两张 PIL.Image。
        """
        src_array = ImageSimilarity.load_image_array(src_img)
        tgt_array = ImageSimilarity.load_image_array(target_img)

        h1, w1 = src_array.shape[:2]
        h2, w2 = tgt_array.shape[:2]
        # 先放大到最大尺寸
        up_h, up_w = 512, 512  # 假设最大尺寸为 512x512
        src_up = cv2.resize(src_array, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        tgt_up = cv2.resize(tgt_array, (up_w, up_h), interpolation=cv2.INTER_CUBIC)

        # 再缩小到最小尺寸
        down_h, down_w = min(h1, h2), min(w1, w2)
        src_down = cv2.resize(src_up, (down_w, down_h), interpolation=cv2.INTER_AREA)
        tgt_down = cv2.resize(tgt_up, (down_w, down_h), interpolation=cv2.INTER_AREA)

        src_out = Image.fromarray(src_down.astype(np.uint8), mode="RGB")
        tgt_out = Image.fromarray(tgt_down.astype(np.uint8), mode="RGB")
        return src_out, tgt_out

    @staticmethod
    def compare_images_lpips(
        img_real: Union[Image.Image, str, Path, np.ndarray],
        img_fake: Union[Image.Image, str, Path, np.ndarray],
        ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]] = None,
        norm="zscore",
        heatmap: bool = False,
        gray: bool = False,
        align: bool = False,
    ) -> Union[Image.Image, np.ndarray]:
        """
        使用 LPIPS (Learned Perceptual Image Patch Similarity) 库比较两张图像的相似度。
        LPIPS 通过​​深度特征空间的距离计算​​模拟人类视觉系统，使评估结果更贴近主观感知
        CODE: https://github.com/richzhang/PerceptualSimilarity
        """
        src_array = ImageSimilarity.load_image_array(img_real)
        target_array = ImageSimilarity.load_image_array(img_fake)
        if src_array.shape != target_array.shape:
            raise ValueError("源图像和目标图像的尺寸不一致。")

        if align:
            src_array, target_array, _ = ImageSift.align_images(src_array, target_array)

        src_tensor = lpips.im2tensor(src_array)
        target_tensor = lpips.im2tensor(target_array)

        with torch.no_grad():
            score_map_tensor = ImageSimilarity.LPIPS_MODEL.forward(src_tensor, target_tensor)

        # 归一化
        diff_gray = ImageSimilarity.norm_method(score_map_tensor, norm)
        diff_gray = ImageSimilarity.apply_ref_mask(diff_gray, ref_mask)
        if heatmap:
            return ImageSimilarity.to_heatmap(diff_gray)
        elif gray:
            return ImageSimilarity.to_gray(diff_gray)
        return diff_gray

    @staticmethod
    def diff_weight(
        src_np: Union[Image.Image, str, Path, np.ndarray],
        target_np: Union[Image.Image, str, Path, np.ndarray],
        heatmap: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """
        计算两张矩阵之间的差异，添加权重, src的值为0-1之间的小数, 越接近1越权重weight越大
        """
        # 计算加权差异
        diff = np.abs(src_np - target_np)
        weighted = diff * src_np
        if heatmap:
            return ImageSimilarity.to_heatmap(weighted)
        # 转为 PIL 灰度图，模式 “L”
        gray_uint8 = (weighted * 255).astype(np.uint8)
        # 如果多通道，把最后一维 squeeze 掉
        gray_uint8 = gray_uint8.squeeze()
        return Image.fromarray(gray_uint8, mode="L")
        # return weighted

    @staticmethod
    def threshold_gray(src_np: np.ndarray, threshold_gray: float) -> Union[np.ndarray, Image.Image]:
        """
        将灰度图像应用阈值处理，返回二值化图像。
        :param src_np: 输入的灰度图像数组
        :param threshold_gray: 阈值，范围 [0, 1]
        :return: 二值化后的图像数组或 PIL 图像
        """
        # 根据阈值, 将src_np 转为 0, 1
        # src_np的值为0-1之间的浮点数
        target_np = np.where(src_np >= threshold_gray, 1.0, 0.0).astype(np.float32)
        # 返回灰度图
        gray_uint8 = (target_np * 255).astype(np.uint8)
        return Image.fromarray(gray_uint8, mode="L")

    @staticmethod
    def norm_method(score_map: Union[torch.Tensor, np.ndarray], norm_method: str = "sigmoid", eps=1e-8) -> np.ndarray:
        # 兼容 numpy array 输入
        if isinstance(score_map, np.ndarray):
            score_map = torch.from_numpy(score_map)
            score_map = score_map.to(torch.float32)
        if norm_method == "minmax":
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + eps)
        elif norm_method == "zscore":
            score_map = (score_map - score_map.mean()) / (score_map.std() + eps)
            score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + eps)
        elif norm_method == "percentile":
            lo, hi = torch.quantile(score_map, torch.tensor([0.05, 0.95], dtype=torch.float32))
            score_map = torch.clamp(score_map, min=lo, max=hi)
            score_map = (score_map - lo) / (hi - lo + eps)
        elif norm_method == "sigmoid":
            score_map = torch.sigmoid(score_map)
        elif norm_method is None:
            score_map = score_map
        else:
            raise ValueError(f"未知的归一化方法: {norm_method}")

        # 移除多余的维度
        score_map = score_map.squeeze()
        return score_map.cpu().numpy() if torch.is_tensor(score_map) else score_map

    @staticmethod
    def apply_ref_mask(diff: np.ndarray, ref_mask: Optional[Union[Image.Image, str, Path, np.ndarray]]) -> np.ndarray:
        if ref_mask is not None:
            m = ImageSimilarity.load_image_array(ref_mask, "GRAY")
            _, mb = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            diff[mb.astype(bool)] = diff.max()
        return diff

    @staticmethod
    def to_heatmap(diff: np.ndarray) -> Image.Image:
        # 断言 diff 是归一化到 [0, 1] 的数组
        if not (0 <= diff.min() <= 1 and 0 <= diff.max() <= 1):
            raise ValueError("差异图必须是归一化到 [0, 1] 的数组。")
        c_map = (diff * 255).astype(np.uint8)
        heat = cv2.applyColorMap(c_map, cv2.COLORMAP_JET)
        return Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB))

    @staticmethod
    def to_gray(diff: np.ndarray) -> Image.Image:
        # 断言 diff 是归一化到 [0, 1] 的数组
        if not (0 <= diff.min() <= 1 and 0 <= diff.max() <= 1):
            raise ValueError("差异图必须是归一化到 [0, 1] 的数组。")
        # 差异越大越接近白色, 差异越小越接近黑色
        # 直接把归一化后的值映射到 [0,255]，并生成 L 模式灰度图
        c_map = (diff * 255).round().astype(np.uint8)
        return Image.fromarray(c_map, mode="L")

    @staticmethod
    def get_sobel(
        img: Union[Image.Image, str, Path, np.ndarray],
        norm_method: str = "minmax",
        smoothing_factor=0.1,
    ) -> np.ndarray:
        # 加载图像为灰度
        img_array = ImageSimilarity.load_image_array(img, "GRAY")

        # 确保是单通道灰度图
        if img_array.ndim == 3:
            img_gray = img_array[:, :, 0]
        else:
            img_gray = img_array

        # 计算 x 方向的 Sobel 算子
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)

        # 计算 y 方向的 Sobel 算子
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

        # 计算梯度幅值
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # 归一化到 [0, 1]
        sobel_magnitude = ImageSimilarity.norm_method(sobel_magnitude, norm_method)

        # 平滑处理
        return sobel_magnitude / (sobel_magnitude + smoothing_factor)

    @staticmethod
    def get_canny(
        img: Union[Image.Image, str, Path, np.ndarray],
        norm_method: str = "zscore",
        smoothing_factor=0.1,
    ) -> np.ndarray:
        # 加载图像为灰度
        img_array = ImageSimilarity.load_image_array(img, "GRAY")

        # 确保是单通道灰度图
        if img_array.ndim == 3:
            img_gray = img_array[:, :, 0]
        else:
            img_gray = img_array

        # 使用 Canny 边缘检测
        edges = cv2.Canny(img_gray, 100, 200)

        # 归一化到 [0, 1]
        edges_magnitude = ImageSimilarity.norm_method(edges, norm_method)

        # 平滑处理
        return edges_magnitude / (edges_magnitude + smoothing_factor)
