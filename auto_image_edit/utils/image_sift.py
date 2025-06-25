from typing import Union, Optional, Tuple, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import cv2
import numpy as np
from PIL import Image


@dataclass
class SIFTConfig:
    """SIFT配置参数"""

    max_features: int = 500
    good_match_percent: float = 0.15
    homography_confidence: float = 0.99
    ransac_threshold: float = 5.0

    # 多尺度参数
    multi_scale: bool = True
    scale_factors: Optional[List[float]] = None
    octave_layers: int = 3
    contrast_threshold: float = 0.04
    edge_threshold: float = 10
    sigma: float = 1.6

    # 匹配参数
    match_ratio_threshold: float = 0.7
    min_match_count: int = 10
    use_adaptive_threshold: bool = True
    use_parallel: bool = True

    def __post_init__(self):
        if self.scale_factors is None:
            self.scale_factors = [1.0, 0.8, 1.2, 0.6, 1.5]


class AlignmentStatus(Enum):
    SUCCESS = "success"
    INSUFFICIENT_FEATURES = "insufficient_features"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    HOMOGRAPHY_FAILED = "homography_failed"
    INVALID_HOMOGRAPHY = "invalid_homography"


class AlignmentError(Exception):
    """图像对齐异常"""

    pass


class ImageSift:
    @staticmethod
    def load_image_array(img_input: Union[Image.Image, str, Path, np.ndarray], color_space: str = "RGB") -> np.ndarray:
        """
        加载图像并转换为指定颜色空间的numpy数组

        Args:
            img_input: 图像输入（PIL Image、文件路径或numpy数组）
            color_space: 目标颜色空间 ("RGB", "LAB", "BGR", "GRAY")

        Returns:
            np.ndarray: 转换后的图像数组
        """
        # 加载图像为numpy数组
        if isinstance(img_input, (str, Path)):
            img = cv2.imread(str(img_input))
            if img is None:
                raise ValueError(f"无法加载图像: {img_input}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input.convert("RGB"))
        elif isinstance(img_input, np.ndarray):
            img = img_input.copy()
        else:
            raise TypeError(f"不支持的图像类型: {type(img_input)}")

        # 确保图像是RGB格式
        if len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

        # 转换颜色空间
        color_space = color_space.upper()
        if color_space == "RGB":
            return img
        elif color_space == "LAB":
            return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        elif color_space == "BGR":
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif color_space == "GRAY":
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"不支持的颜色空间: {color_space}")

    @staticmethod
    def align_images_sift_lab(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        channel: str = "L",
        config: Optional[SIFTConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], AlignmentStatus]:
        """
        使用多尺度SIFT在Lab颜色空间进行特征点匹配和对齐

        Args:
            src_img: 源图像（参考图像）
            target_img: 目标图像（待对齐图像）
            channel: 使用的Lab通道 ("L", "A", "B", "ALL")
            config: SIFT配置参数

        Returns:
            tuple: (对齐后的源图像Lab, 对齐后的目标图像Lab, 单应矩阵, 对齐状态)
        """
        if config is None:
            config = SIFTConfig()

        try:
            # 加载图像为Lab颜色空间
            src_lab = ImageSift.load_image_array(src_img, "LAB")
            target_lab = ImageSift.load_image_array(target_img, "LAB")

            # 提取特征检测用的通道
            src_feature, target_feature = ImageSift._extract_lab_channel(src_lab, target_lab, channel)

            # 多尺度SIFT特征检测
            if config.multi_scale:
                all_kp1, all_des1 = ImageSift._multi_scale_sift_detection(src_feature, config, use_parallel=config.use_parallel)
                all_kp2, all_des2 = ImageSift._multi_scale_sift_detection(target_feature, config, use_parallel=config.use_parallel)
            else:
                # 单尺度SIFT检测
                kp1, des1 = ImageSift._single_scale_sift_detection(src_feature, config)
                kp2, des2 = ImageSift._single_scale_sift_detection(target_feature, config)

                if des1 is not None and des2 is not None:
                    all_kp1, all_des1 = [kp1], [des1]
                    all_kp2, all_des2 = [kp2], [des2]
                else:
                    all_kp1, all_des1 = [], []
                    all_kp2, all_des2 = [], []

            if not all_kp1 or not all_kp2:
                return src_lab, target_lab, None, AlignmentStatus.INSUFFICIENT_FEATURES

            # 高效组合特征点和描述符
            combined_kp1, combined_des1 = ImageSift._combine_features_efficiently(all_kp1, all_des1)
            combined_kp2, combined_des2 = ImageSift._combine_features_efficiently(all_kp2, all_des2)

            if combined_des1.size == 0 or combined_des2.size == 0:
                return src_lab, target_lab, None, AlignmentStatus.INSUFFICIENT_FEATURES

            # 特征匹配
            good_matches = ImageSift._feature_matching(combined_kp1, combined_des1, combined_kp2, combined_des2, config)

            # 检查匹配点数量
            if len(good_matches) < config.min_match_count:
                return src_lab, target_lab, None, AlignmentStatus.INSUFFICIENT_MATCHES

            # 筛选最佳匹配点
            good_matches = ImageSift._filter_best_matches(good_matches, config)

            # 提取匹配点坐标
            points1, points2 = ImageSift._extract_match_points(good_matches, combined_kp1, combined_kp2)

            # 使用RANSAC估计单应矩阵
            homography = ImageSift._robust_homography_estimation(points2, points1, config)

            if homography is None:
                return src_lab, target_lab, None, AlignmentStatus.HOMOGRAPHY_FAILED

            # 验证单应矩阵
            if not ImageSift._validate_homography(homography, src_lab.shape):
                return src_lab, target_lab, None, AlignmentStatus.INVALID_HOMOGRAPHY

            # 应用透视变换
            height, width = src_lab.shape[:2]
            aligned_target_lab = cv2.warpPerspective(target_lab, homography, (width, height))

            return src_lab, aligned_target_lab, homography, AlignmentStatus.SUCCESS

        except Exception as e:
            return (
                src_lab if "src_lab" in locals() else None,
                target_lab if "target_lab" in locals() else None,
                None,
                AlignmentStatus.HOMOGRAPHY_FAILED,
            )

    @staticmethod
    def _extract_lab_channel(src_lab: np.ndarray, target_lab: np.ndarray, channel: str) -> Tuple[np.ndarray, np.ndarray]:
        """提取Lab颜色空间的指定通道"""
        channel_upper = channel.upper()

        if channel_upper == "L":
            src_feature = src_lab[:, :, 0]
            target_feature = target_lab[:, :, 0]
        elif channel_upper == "A":
            src_feature = src_lab[:, :, 1]
            target_feature = target_lab[:, :, 1]
        elif channel_upper == "B":
            src_feature = src_lab[:, :, 2]
            target_feature = target_lab[:, :, 2]
        elif channel_upper == "ALL":
            # 加权组合所有通道
            src_feature = (src_lab[:, :, 0] * 0.6 + src_lab[:, :, 1] * 0.2 + src_lab[:, :, 2] * 0.2).astype(np.uint8)
            target_feature = (target_lab[:, :, 0] * 0.6 + target_lab[:, :, 1] * 0.2 + target_lab[:, :, 2] * 0.2).astype(np.uint8)
        else:
            raise ValueError(f"不支持的Lab通道: {channel}，请使用 'L', 'A', 'B', 或 'ALL'")

        return src_feature, target_feature

    @staticmethod
    def _single_scale_sift_detection(image: np.ndarray, config: SIFTConfig) -> Tuple[List, np.ndarray]:
        """单尺度SIFT特征检测"""
        sift = cv2.SIFT_create(
            nfeatures=config.max_features,
            nOctaveLayers=config.octave_layers,
            contrastThreshold=config.contrast_threshold,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma,
        )
        return sift.detectAndCompute(image, None)

    @staticmethod
    def _detect_at_scale(args) -> Tuple[List, np.ndarray, float]:
        """单个尺度的特征检测（用于并行处理）"""
        scaled_image, config, scale = args

        sift = cv2.SIFT_create(
            nfeatures=config.max_features // len(config.scale_factors),
            nOctaveLayers=config.octave_layers,
            contrastThreshold=config.contrast_threshold * scale,
            edgeThreshold=config.edge_threshold,
            sigma=config.sigma,
        )

        keypoints, descriptors = sift.detectAndCompute(scaled_image, None)

        if keypoints is not None and descriptors is not None:
            # 将关键点坐标转换回原始尺度
            if scale != 1.0:
                for kp in keypoints:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                    kp.size = kp.size / scale

        return keypoints, descriptors, scale

    @staticmethod
    def _multi_scale_sift_detection(
        image: np.ndarray,
        config: SIFTConfig,
        use_parallel: bool = True,
    ) -> Tuple[List[List], List[np.ndarray]]:
        """多尺度SIFT特征检测"""
        original_height, original_width = image.shape[:2]

        # 准备所有尺度的图像
        scale_data = []
        for scale in config.scale_factors:
            if scale != 1.0:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                # 确保尺寸不会太小
                if new_width < 50 or new_height < 50:
                    continue

                scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_image = image.copy()

            scale_data.append((scaled_image, config, scale))

        # 并行或串行处理
        if use_parallel and len(scale_data) > 1:
            max_workers = min(len(scale_data), multiprocessing.cpu_count())
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(ImageSift._detect_at_scale, scale_data))
        else:
            results = [ImageSift._detect_at_scale(data) for data in scale_data]

        # 提取结果
        all_keypoints = []
        all_descriptors = []

        for keypoints, descriptors, scale in results:
            if keypoints is not None and descriptors is not None:
                all_keypoints.append(keypoints)
                all_descriptors.append(descriptors)

        return all_keypoints, all_descriptors

    @staticmethod
    def _combine_features_efficiently(all_keypoints: List[List], all_descriptors: List[np.ndarray]) -> Tuple[List, np.ndarray]:
        """高效组合特征点和描述符"""
        if not all_keypoints or not all_descriptors:
            return [], np.array([])

        # 组合关键点
        combined_keypoints = []
        for kp_list in all_keypoints:
            combined_keypoints.extend(kp_list)

        # 高效组合描述符
        if len(all_descriptors) == 1:
            combined_descriptors = all_descriptors[0]
        else:
            # 预先计算总长度，避免重复分配内存
            total_length = sum(desc.shape[0] for desc in all_descriptors)
            feature_dim = all_descriptors[0].shape[1]

            combined_descriptors = np.empty((total_length, feature_dim), dtype=all_descriptors[0].dtype)
            current_idx = 0

            for desc in all_descriptors:
                end_idx = current_idx + desc.shape[0]
                combined_descriptors[current_idx:end_idx] = desc
                current_idx = end_idx

        return combined_keypoints, combined_descriptors

    @staticmethod
    def _feature_matching(kp1: List, des1: np.ndarray, kp2: List, des2: np.ndarray, config: SIFTConfig) -> List:
        """特征匹配"""
        if des1.size == 0 or des2.size == 0:
            return []

        # 使用FLANN匹配器
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return []

        # 自适应阈值计算
        adaptive_threshold = config.match_ratio_threshold
        if config.use_adaptive_threshold and matches:
            distances = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    distances.append(match_pair[0].distance / match_pair[1].distance)

            if distances:
                distances = np.array(distances)
                median_ratio = np.median(distances)
                std_ratio = np.std(distances)
                adaptive_threshold = min(config.match_ratio_threshold, median_ratio + 0.1 * std_ratio)

        # Lowe's比率测试
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < adaptive_threshold * n.distance:
                    good_matches.append(m)

        # 几何一致性检查
        if len(good_matches) > 4:
            good_matches = ImageSift._geometric_consistency_check(good_matches, kp1, kp2)

        return good_matches

    @staticmethod
    def _geometric_consistency_check(matches: List, kp1: List, kp2: List, threshold: float = 2.0) -> List:
        """几何一致性检查"""
        if len(matches) < 4:
            return matches

        consistent_matches = []
        for i, match in enumerate(matches):
            pt1 = np.array(kp1[match.queryIdx].pt)
            pt2 = np.array(kp2[match.trainIdx].pt)

            # 检查与其他匹配点的几何关系
            consistent_count = 0
            total_checks = 0

            for j, other_match in enumerate(matches):
                if i != j:
                    other_pt1 = np.array(kp1[other_match.queryIdx].pt)
                    other_pt2 = np.array(kp2[other_match.trainIdx].pt)

                    # 计算距离比率
                    dist1 = np.linalg.norm(pt1 - other_pt1)
                    dist2 = np.linalg.norm(pt2 - other_pt2)

                    if dist1 > 5 and dist2 > 5:  # 避免过近的点
                        ratio = dist2 / dist1 if dist1 > 0 else float("inf")
                        if 1.0 / threshold <= ratio <= threshold:
                            consistent_count += 1
                        total_checks += 1

            # 如果大部分检查都通过，则认为该匹配点一致
            if total_checks == 0 or consistent_count / total_checks >= 0.5:
                consistent_matches.append(match)

        return consistent_matches

    @staticmethod
    def _filter_best_matches(good_matches: List, config: SIFTConfig) -> List:
        """筛选最佳匹配点"""
        if not good_matches:
            return []

        # 按距离排序
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        # 计算保留的匹配点数量
        num_good_matches = int(len(good_matches) * config.good_match_percent)
        num_good_matches = max(num_good_matches, config.min_match_count)
        num_good_matches = min(num_good_matches, len(good_matches))

        return good_matches[:num_good_matches]

    @staticmethod
    def _extract_match_points(good_matches: List, kp1: List, kp2: List) -> Tuple[np.ndarray, np.ndarray]:
        """提取匹配点坐标"""
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        return points1, points2

    @staticmethod
    def _robust_homography_estimation(
        points2: np.ndarray, points1: np.ndarray, config: SIFTConfig, max_attempts: int = 3
    ) -> Optional[np.ndarray]:
        """鲁棒的单应矩阵估计，支持多次尝试"""
        for attempt in range(max_attempts):
            try:
                # 调整RANSAC参数
                current_threshold = config.ransac_threshold * (1.0 + 0.5 * attempt)
                current_confidence = max(config.homography_confidence - 0.05 * attempt, 0.9)

                homography, mask = cv2.findHomography(
                    points2, points1, cv2.RANSAC, current_threshold, confidence=current_confidence, maxIters=2000
                )

                if homography is not None:
                    return homography

            except cv2.error:
                continue

        return None

    @staticmethod
    def _validate_homography(homography: np.ndarray, image_shape: tuple) -> bool:
        """验证单应矩阵的合理性"""
        if homography is None:
            return False

        try:
            # 检查矩阵条件数
            cond_number = np.linalg.cond(homography)
            if cond_number > 1e6:  # 条件数过大，矩阵接近奇异
                return False

            # 检查变换是否过于极端
            h, w = image_shape[:2]
            corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

            transformed_corners = homography @ corners
            transformed_corners = transformed_corners[:2, :] / transformed_corners[2, :]

            # 检查变换后的角点是否合理
            if np.any(np.abs(transformed_corners) > 10 * max(image_shape)):
                return False

            return True

        except:
            return False

    @staticmethod
    def evaluate_alignment_quality(src_img: np.ndarray, aligned_img: np.ndarray, homography: Optional[np.ndarray]) -> dict:
        """评估对齐质量"""
        try:
            from skimage.metrics import structural_similarity as ssim

            # 转换为灰度图
            if len(src_img.shape) == 3:
                gray_src = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
                gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
            else:
                gray_src = src_img
                gray_aligned = aligned_img

            # 计算SSIM
            ssim_score = ssim(gray_src, gray_aligned)

            # 计算均方误差
            mse = np.mean((gray_src - gray_aligned) ** 2)

            # 单应矩阵条件数
            condition_number = np.linalg.cond(homography) if homography is not None else float("inf")

            return {
                "ssim_score": float(ssim_score),
                "mse": float(mse),
                "homography_condition": float(condition_number),
                "is_good_alignment": ssim_score > 0.7 and condition_number < 1e5,
            }

        except ImportError:
            # 如果没有skimage，使用简单的MSE
            if len(src_img.shape) == 3:
                gray_src = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
                gray_aligned = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2GRAY)
            else:
                gray_src = src_img
                gray_aligned = aligned_img

            mse = np.mean((gray_src - gray_aligned) ** 2)
            condition_number = np.linalg.cond(homography) if homography is not None else float("inf")

            return {
                "ssim_score": None,
                "mse": float(mse),
                "homography_condition": float(condition_number),
                "is_good_alignment": mse < 1000 and condition_number < 1e5,
            }

    # 保持向后兼容的接口
    @staticmethod
    def align_images(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        channel: str = "L",
        max_features: int = 500,
        good_match_percent: float = 0.15,
        homography_confidence: float = 0.99,
        ransac_threshold: float = 5.0,
        multi_scale: bool = True,
        scale_factors: Optional[list] = None,
        octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
        match_ratio_threshold: float = 0.7,
        min_match_count: int = 10,
        use_adaptive_threshold: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """保持向后兼容的接口"""
        config = SIFTConfig(
            max_features=max_features,
            good_match_percent=good_match_percent,
            homography_confidence=homography_confidence,
            ransac_threshold=ransac_threshold,
            multi_scale=multi_scale,
            scale_factors=scale_factors,
            octave_layers=octave_layers,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            sigma=sigma,
            match_ratio_threshold=match_ratio_threshold,
            min_match_count=min_match_count,
            use_adaptive_threshold=use_adaptive_threshold,
        )

        src_aligned, target_aligned, homography, _ = ImageSift.align_images_sift_lab(src_img, target_img, channel, config)

        return src_aligned, target_aligned, homography
