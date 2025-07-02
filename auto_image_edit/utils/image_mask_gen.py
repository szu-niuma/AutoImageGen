import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union
from urllib.parse import urlparse
from functools import lru_cache, partial
from skimage import filters

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import disk
from skimage.metrics import structural_similarity as ssim

import torch
import lpips

from .image_sift import ImageSift


class ImageMaskGen:
    @staticmethod
    def lpips_replace_pixel(pixel_diff, lpips_diff):
        return np.where(pixel_diff > 0, lpips_diff, pixel_diff)

    @staticmethod
    def pixel_multiplication(pixel_diff, lpips_diff):
        return lpips_diff * pixel_diff

    @staticmethod
    def rule_multiplication(pixel_diff: np.ndarray, lpips_diff: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        使用幂函数融合像素差异和LPIPS差异: f(P(x), L(x)) = (L(x) * P(x))^alpha

        Args:
            pixel_diff: 像素差异矩阵，范围[0,1]
            lpips_diff: LPIPS差异矩阵，范围[0,1]
            alpha: 指数参数，用于控制结果的强度，默认为1.0

        Returns:
            融合后的差异矩阵，范围[0,1]
        """
        # 创建掩码，标识 pixel_diff 为 0 的区域
        zero_mask = pixel_diff == 0

        # 计算乘积的幂
        result = np.power(lpips_diff * pixel_diff, alpha)

        # 将 pixel_diff 为 0 的区域设为 0
        result[zero_mask] = 0

        return result

    @staticmethod
    def entropy_fusion(pixel_diff, lpips_diff, window_size=3):
        """
        基于局部熵的图像融合方法

        Args:
            diff_map: 差异图像 (H, W)，已归一化到[0,1]
            lpips_map: LPIPS图像 (H, W)，已归一化到[0,1]
            window_size: 局部窗口大小，默认为5

        Returns:
            fused_map: 融合后的图像 (H, W)，范围[0,1]
        """

        # 快速输入验证
        if not (
            isinstance(pixel_diff, np.ndarray) and isinstance(lpips_diff, np.ndarray) and pixel_diff.ndim == 2 and lpips_diff.ndim == 2
        ):
            raise ValueError("所有输入必须是2D numpy数组")

        if pixel_diff.shape != lpips_diff.shape:
            raise ValueError("输入图像必须具有相同的尺寸")

        # 确保window_size是奇数
        window_size = max(3, window_size + (1 - window_size % 2))

        @lru_cache(maxsize=16)
        def _get_structure_element(radius):
            """缓存结构元素以避免重复创建"""
            return disk(radius)

        def fast_local_entropy(image):
            """优化的局部熵计算"""
            # 直接转换为uint8
            image_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)

            # 使用缓存的结构元素
            selem = _get_structure_element(window_size // 2)

            # 使用skimage的熵计算函数
            entropy_map = filters.rank.entropy(image_uint8, selem)

            # 理论最大熵值归一化
            max_entropy = np.log2(256)
            return entropy_map.astype(np.float32) / max_entropy

        # 计算两个差异图的熵
        entropy_pixel = fast_local_entropy(pixel_diff)
        entropy_lpips = fast_local_entropy(lpips_diff)

        # 计算总熵和权重
        total_entropy = entropy_pixel + entropy_lpips + 1e-10  # 避免除零
        weight = entropy_pixel / total_entropy

        fused_map = weight * pixel_diff + (1 - weight) * lpips_diff
        # 断言 fused_map 的值在 [0, 1] 范围内
        if not np.all((0 <= fused_map) & (fused_map <= 1)):
            raise ValueError("融合后的图像值必须在 [0, 1] 范围内")
        return fused_map

    @staticmethod
    def adapter(pixel_diff: np.ndarray, lpips_diff: np.ndarray, adapter) -> np.ndarray:
        # 创建掩码，标识 pixel_diff 为 0 的区域
        zero_mask = pixel_diff == 0
        # 在 pixel_diff 不为 0 的区域应用权重
        result = adapter * pixel_diff + (1 - adapter) * lpips_diff
        # 将 pixel_diff 为 0 的区域设为 0
        result[zero_mask] = 0
        return result

    @staticmethod
    def l2_weight(pixel_diff, lpips_diff, weight):
        """
        使用加权 L2 范数结合两个已归一化的差异 Mask。
        Args:
            mask_pixel_diff_norm (np.ndarray): 归一化后的像素差 Mask (值在 [0, 1] 之间)。
            mask_lpips_norm (np.ndarray): 归一化后的人眼感知 LPIPS Mask (值在 [0, 1] 之间)。
            w_pixel_diff (float): 像素差 Mask 的权重。
            w_lpips (float): LPIPS Mask 的权重。
        Returns:
            np.ndarray: 结合后的差异 Mask，值也在 [0, 1] 之间。
        Raises:
            ValueError: 如果输入 Mask 的形状不一致。
            ValueError: 如果权重之和不接近 1.0 (可选检查)。
        """
        # 确保输入 Mask 形状一致
        if pixel_diff.shape != lpips_diff.shape:
            raise ValueError("输入 Mask 的形状必须一致。")

        # 创建掩码，标识 pixel_diff 为 0 的区域
        zero_mask = pixel_diff == 0

        # 计算加权平方和
        weighted_sum_of_squares = weight * (pixel_diff**2) + (1 - weight) * (lpips_diff**2)
        # 开平方根，得到最终的组合 Mask
        combined_mask = np.sqrt(weighted_sum_of_squares)

        # 将 pixel_diff 为 0 的区域设为 0
        combined_mask[zero_mask] = 0

        # 确保结果在 [0, 1] 范围内
        assert np.all((0 <= combined_mask) & (combined_mask <= 1)), "组合后的 Mask 值必须在 [0, 1] 范围内"
        return combined_mask

    @staticmethod
    def linear_weight(pixel_diff: np.ndarray, lpips_diff: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        线性加权融合
        """
        if not (0 <= weight <= 1):
            raise ValueError("权重必须在0到1之间")

        # 创建掩码，标识 pixel_diff 为 0 的区域
        zero_mask = pixel_diff == 0

        # 进行线性加权融合
        result = weight * pixel_diff + (1 - weight) * lpips_diff

        # 将 pixel_diff 为 0 的区域设为 0
        result[zero_mask] = 0

        return result

    @staticmethod
    def sigmoid_weight(pixel_diff: np.ndarray, lpips_diff: np.ndarray, k: float = 10, d0: float = 0.5) -> np.ndarray:
        """
        使用sigmoid函数动态调整像素差异和LPIPS差异的权重

        Args:
            pixel_diff: 像素差异矩阵，范围[0,1]
            lpips_diff: LPIPS差异矩阵，范围[0,1]
            k: sigmoid曲线的陡峭程度，值越大曲线越陡
            d0: 阈值点，当pixel_diff=d0时，权重alpha=0.5

        Returns:
            融合后的差异矩阵
        """
        # 创建掩码，标识 pixel_diff 为 0 的区域
        zero_mask = pixel_diff == 0

        # 计算动态权重和融合结果
        alpha = 1 / (1 + np.exp(-k * (pixel_diff - d0)))
        result = alpha * pixel_diff + (1 - alpha) * lpips_diff

        # 将 pixel_diff 为 0 的区域设为 0
        result[zero_mask] = 0

        return result
