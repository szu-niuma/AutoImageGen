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
    def rule_multiplication(pixel_diff, lpips_diff):
        return np.where((pixel_diff > 0) & (lpips_diff > 0), lpips_diff * pixel_diff * 255 * 255, lpips_diff + pixel_diff)

    @staticmethod
    def adapter(pixel_diff: np.ndarray, lpips_diff: np.ndarray, adapter) -> np.ndarray:
        return adapter * lpips_diff + (1 - adapter) * pixel_diff

    @staticmethod
    def entropy_fusion(pixel_diff, lpips_diff, window_size=5):
        """
        基于局部熵的图像融合方法

        Args:
            diff_map: 差异图像 (H, W)，已归一化到[0,1]
            ssim_map: SSIM图像 (H, W)，已归一化到[0,1]
            lpips_map: LPIPS图像 (H, W)，已归一化到[0,1]
            window_size: 局部窗口大小，默认为5

        Returns:
            fused_map: 融合后的图像 (H, W)，范围[0,1]
        """

        @lru_cache(maxsize=8)
        def _get_structure_element(radius):
            """缓存结构元素以避免重复创建"""
            return disk(radius)

        def local_entropy(image, window_size=5):
            """
            计算局部熵，针对已归一化输入进行优化
            """
            # 确保window_size是奇数且合理
            window_size = max(3, window_size if window_size % 2 == 1 else window_size + 1)
            radius = window_size // 2

            # 由于输入已归一化到[0,1]，直接转换为uint8
            image_uint8 = (image * 255).astype(np.uint8)

            # 使用缓存的结构元素
            selem = _get_structure_element(radius)

            # 计算局部熵
            entropy_map = filters.rank.entropy(image_uint8, selem)

            # 优化的归一化：直接除以理论最大值
            max_entropy = np.log2(256)  # 8位图像的最大熵
            return (entropy_map.astype(np.float32) / max_entropy).clip(0, 1)

        # 快速输入验证
        maps = [pixel_diff, lpips_diff]
        if not all(isinstance(img, np.ndarray) and img.ndim == 2 for img in maps):
            raise ValueError("所有输入必须是2D numpy数组")

        shape = pixel_diff.shape
        if not all(img.shape == shape for img in maps[1:]):
            raise ValueError("所有输入图像必须具有相同的尺寸")

        # 批量计算局部熵
        try:
            entropies = [local_entropy(img, window_size) for img in maps]
        except Exception as e:
            raise RuntimeError(f"计算局部熵时出错: {e}")

        # 向量化权重计算
        entropy_stack = np.stack(entropies, axis=0)  # (3, H, W)
        total_entropy = np.sum(entropy_stack, axis=0) + 1e-10  # 避免除零

        # 计算权重
        weights = entropy_stack / total_entropy[None, ...]  # 广播除法

        # 向量化融合
        map_stack = np.stack(maps, axis=0)  # (3, H, W)
        fused_map = np.sum(weights * map_stack, axis=0)

        # 由于输入已归一化，输出应该也在[0,1]范围内，但仍进行裁剪确保安全
        return np.clip(fused_map, 0, 1).astype(np.float32)

    @staticmethod
    def linear_weight(pixel_diff: np.ndarray, lpips_diff: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        线性加权融合
        """
        if not (0 <= weight <= 1):
            raise ValueError("权重必须在0到1之间")
        return weight * pixel_diff + (1 - weight) * lpips_diff
