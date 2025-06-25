from typing import Union, Optional, Tuple, List, Dict, Any
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from dataclasses import dataclass
import matplotlib.font_manager as fm

from .image_sift import ImageSift, SIFTConfig, AlignmentStatus


@dataclass
class VisualizationConfig:
    """可视化配置参数"""

    figure_size: Tuple[int, int] = (15, 10)
    match_line_width: float = 1.0
    match_line_alpha: float = 0.7
    keypoint_size: int = 3
    keypoint_color: str = "red"
    match_color: str = "green"
    good_match_color: str = "blue"
    bad_match_color: str = "red"
    show_match_distance: bool = True
    max_matches_display: int = 100
    font_size: int = 12
    save_dpi: int = 300


class ImageSiftVisualizer:
    """SIFT图像对齐可视化工具"""

    @staticmethod
    def _setup_chinese_font():
        """设置中文字体"""
        font_path = "/home/yuyangxin/data/AutoImageGen/resource/front/MSYH.TTC"
        # 如果指定了字体路径，优先使用
        if font_path and Path(font_path).exists():
            try:
                # 注册字体文件
                fm.fontManager.addfont(font_path)
                font_name = fm.FontProperties(fname=font_path).get_name()
                plt.rcParams["font.sans-serif"] = [font_name] + plt.rcParams["font.sans-serif"]
                plt.rcParams["axes.unicode_minus"] = False
                print(f"成功加载字体文件: {font_path}")
                return True
            except Exception as e:
                print(f"加载字体文件失败: {e}")
        else:
            return False

    @staticmethod
    def visualize_alignment_process(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        channel: str = "L",
        config: Optional[SIFTConfig] = None,
        vis_config: Optional[VisualizationConfig] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        完整的对齐过程可视化

        Args:
            src_img: 源图像
            target_img: 目标图像
            channel: Lab通道
            config: SIFT配置
            vis_config: 可视化配置
            save_path: 保存路径

        Returns:
            包含可视化结果和统计信息的字典
        """
        # 设置中文字体
        ImageSiftVisualizer._setup_chinese_font()

        if config is None:
            config = SIFTConfig()
        if vis_config is None:
            vis_config = VisualizationConfig()

        # 执行对齐
        src_lab, target_aligned_lab, homography, status = ImageSift.align_images_sift_lab(src_img, target_img, channel, config)

        # 转换为RGB用于显示
        src_rgb = cv2.cvtColor(src_lab, cv2.COLOR_LAB2RGB)
        target_rgb = cv2.cvtColor(ImageSift.load_image_array(target_img, "LAB"), cv2.COLOR_LAB2RGB)
        target_aligned_rgb = cv2.cvtColor(target_aligned_lab, cv2.COLOR_LAB2RGB)

        # 创建可视化
        fig = plt.figure(figsize=(20, 15))

        # 1. 原始图像对比
        ax1 = plt.subplot(3, 3, 1)
        plt.imshow(src_rgb)
        plt.title("源图像 (参考)", fontsize=vis_config.font_size)
        plt.axis("off")

        ax2 = plt.subplot(3, 3, 2)
        plt.imshow(target_rgb)
        plt.title("目标图像 (待对齐)", fontsize=vis_config.font_size)
        plt.axis("off")

        ax3 = plt.subplot(3, 3, 3)
        plt.imshow(target_aligned_rgb)
        plt.title("对齐后图像", fontsize=vis_config.font_size)
        plt.axis("off")

        # 2. 特征点检测可视化
        src_feature, target_feature = ImageSift._extract_lab_channel(src_lab, ImageSift.load_image_array(target_img, "LAB"), channel)

        # 检测特征点
        if config.multi_scale:
            all_kp1, all_des1 = ImageSift._multi_scale_sift_detection(src_feature, config)
            all_kp2, all_des2 = ImageSift._multi_scale_sift_detection(target_feature, config)
            combined_kp1, combined_des1 = ImageSift._combine_features_efficiently(all_kp1, all_des1)
            combined_kp2, combined_des2 = ImageSift._combine_features_efficiently(all_kp2, all_des2)
        else:
            combined_kp1, combined_des1 = ImageSift._single_scale_sift_detection(src_feature, config)
            combined_kp2, combined_des2 = ImageSift._single_scale_sift_detection(target_feature, config)

        # 绘制特征点
        ax4 = plt.subplot(3, 3, 4)
        src_with_kp = ImageSiftVisualizer._draw_keypoints(src_rgb, combined_kp1, vis_config)
        plt.imshow(src_with_kp)
        plt.title(f"源图像特征点 ({len(combined_kp1)}个)", fontsize=vis_config.font_size)
        plt.axis("off")

        ax5 = plt.subplot(3, 3, 5)
        target_with_kp = ImageSiftVisualizer._draw_keypoints(target_rgb, combined_kp2, vis_config)
        plt.imshow(target_with_kp)
        plt.title(f"目标图像特征点 ({len(combined_kp2)}个)", fontsize=vis_config.font_size)
        plt.axis("off")

        # 3. 特征匹配可视化
        if combined_des1.size > 0 and combined_des2.size > 0:
            good_matches = ImageSift._feature_matching(combined_kp1, combined_des1, combined_kp2, combined_des2, config)

            ax6 = plt.subplot(3, 3, 6)
            match_img = ImageSiftVisualizer._draw_matches(src_rgb, combined_kp1, target_rgb, combined_kp2, good_matches, vis_config)
            plt.imshow(match_img)
            plt.title(f"特征匹配 ({len(good_matches)}个)", fontsize=vis_config.font_size)
            plt.axis("off")

        # 4. 对齐前后对比
        ax7 = plt.subplot(3, 3, 7)
        before_blend = ImageSiftVisualizer._create_blend_image(src_rgb, target_rgb, 0.5)
        plt.imshow(before_blend)
        plt.title("对齐前叠加", fontsize=vis_config.font_size)
        plt.axis("off")

        ax8 = plt.subplot(3, 3, 8)
        after_blend = ImageSiftVisualizer._create_blend_image(src_rgb, target_aligned_rgb, 0.5)
        plt.imshow(after_blend)
        plt.title("对齐后叠加", fontsize=vis_config.font_size)
        plt.axis("off")
        plt.tight_layout()

        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=vis_config.save_dpi, bbox_inches="tight")

        # 计算统计信息
        stats = {
            "alignment_status": status.value,
            "num_src_features": len(combined_kp1) if "combined_kp1" in locals() else 0,
            "num_target_features": len(combined_kp2) if "combined_kp2" in locals() else 0,
            "num_matches": len(good_matches) if "good_matches" in locals() else 0,
            "homography_condition": np.linalg.cond(homography) if homography is not None else None,
        }

        # 评估对齐质量
        if status == AlignmentStatus.SUCCESS:
            quality = ImageSift.evaluate_alignment_quality(src_rgb, target_aligned_rgb, homography)
            stats.update(quality)

        return {"figure": fig, "statistics": stats, "src_aligned": src_rgb, "target_aligned": target_aligned_rgb, "homography": homography}

    @staticmethod
    def visualize_matches_detailed(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        matches: List,
        kp1: List,
        kp2: List,
        vis_config: Optional[VisualizationConfig] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        详细的匹配点可视化
        """
        if vis_config is None:
            vis_config = VisualizationConfig()

        src_rgb = ImageSift.load_image_array(src_img, "RGB")
        target_rgb = ImageSift.load_image_array(target_img, "RGB")

        fig, axes = plt.subplots(2, 2, figsize=vis_config.figure_size)

        # 1. 匹配点连线图
        ax1 = axes[0, 0]
        match_img = ImageSiftVisualizer._draw_matches(src_rgb, kp1, target_rgb, kp2, matches, vis_config)
        ax1.imshow(match_img)
        ax1.set_title(f"匹配点连线 ({len(matches)}个)", fontsize=vis_config.font_size)
        ax1.axis("off")

        # 2. 匹配距离分布
        ax2 = axes[0, 1]
        distances = [m.distance for m in matches]
        ax2.hist(distances, bins=30, alpha=0.7, color="skyblue")
        ax2.set_xlabel("匹配距离")
        ax2.set_ylabel("频次")
        ax2.set_title("匹配距离分布")
        ax2.grid(True, alpha=0.3)

        # 3. 匹配点空间分布
        ax3 = axes[1, 0]
        src_points = [kp1[m.queryIdx].pt for m in matches]
        target_points = [kp2[m.trainIdx].pt for m in matches]

        src_x, src_y = zip(*src_points)
        ax3.scatter(src_x, src_y, c="red", s=20, alpha=0.6, label="源图像")
        ax3.set_xlim(0, src_rgb.shape[1])
        ax3.set_ylim(src_rgb.shape[0], 0)
        ax3.set_title("源图像匹配点分布")
        ax3.set_aspect("equal")
        ax3.grid(True, alpha=0.3)

        # 4. 匹配向量场
        ax4 = axes[1, 1]
        # 降采样显示向量场
        step = max(1, len(matches) // 50)
        for i in range(0, len(matches), step):
            m = matches[i]
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            # 计算位移向量
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            ax4.arrow(pt1[0], pt1[1], dx * 0.1, dy * 0.1, head_width=5, head_length=5, fc="blue", ec="blue", alpha=0.6)

        ax4.set_xlim(0, max(src_rgb.shape[1], target_rgb.shape[1]))
        ax4.set_ylim(max(src_rgb.shape[0], target_rgb.shape[0]), 0)
        ax4.set_title("匹配向量场")
        ax4.set_aspect("equal")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=vis_config.save_dpi, bbox_inches="tight")

        return fig

    @staticmethod
    def visualize_multi_scale_features(
        image: Union[Image.Image, str, Path, np.ndarray],
        config: Optional[SIFTConfig] = None,
        vis_config: Optional[VisualizationConfig] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        多尺度特征检测可视化
        """
        if config is None:
            config = SIFTConfig()
        if vis_config is None:
            vis_config = VisualizationConfig()

        img_rgb = ImageSift.load_image_array(image, "RGB")
        img_lab = ImageSift.load_image_array(image, "LAB")
        img_feature = img_lab[:, :, 0]  # 使用L通道

        # 多尺度特征检测
        all_kp, all_des = ImageSift._multi_scale_sift_detection(img_feature, config)

        # 创建可视化
        n_scales = len(config.scale_factors)
        cols = min(3, n_scales)
        rows = (n_scales + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (scale, kp_list) in enumerate(zip(config.scale_factors, all_kp)):
            row, col = i // cols, i % cols
            ax = axes[row, col]

            # 绘制该尺度的特征点
            img_with_kp = ImageSiftVisualizer._draw_keypoints(img_rgb, kp_list, vis_config)
            ax.imshow(img_with_kp)
            ax.set_title(f"尺度 {scale:.1f} ({len(kp_list)}个特征点)", fontsize=vis_config.font_size)
            ax.axis("off")

        # 隐藏多余的子图
        total_plots = rows * cols
        for i in range(n_scales, total_plots):
            row, col = i // cols, i % cols
            axes[row, col].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=vis_config.save_dpi, bbox_inches="tight")

        return fig

    @staticmethod
    def visualize_homography_transformation(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        homography: np.ndarray,
        vis_config: Optional[VisualizationConfig] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        单应矩阵变换可视化
        """
        if vis_config is None:
            vis_config = VisualizationConfig()

        src_rgb = ImageSift.load_image_array(src_img, "RGB")
        h, w = src_rgb.shape[:2]

        # 定义原始角点和网格点
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # 创建网格点
        grid_x, grid_y = np.meshgrid(np.linspace(0, w, 10), np.linspace(0, h, 8))
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        # 应用单应矩阵变换
        corners_transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
        grid_transformed = cv2.perspectiveTransform(grid_points.reshape(-1, 1, 2), homography).reshape(-1, 2)

        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1. 原始图像及网格
        ax1 = axes[0]
        ax1.imshow(src_rgb)

        # 绘制原始网格
        for i in range(0, len(grid_points), 10):
            row_points = grid_points[i : i + 10]
            ax1.plot(row_points[:, 0], row_points[:, 1], "b-", alpha=0.5)

        for i in range(10):
            col_points = grid_points[i::10]
            ax1.plot(col_points[:, 0], col_points[:, 1], "b-", alpha=0.5)

        # 绘制原始边界
        corners_closed = np.vstack([corners, corners[0]])
        ax1.plot(corners_closed[:, 0], corners_closed[:, 1], "r-", linewidth=3, label="原始边界")
        ax1.set_title("原始图像", fontsize=vis_config.font_size)
        ax1.axis("off")
        ax1.legend()

        # 2. 变换后的网格
        ax2 = axes[1]
        ax2.imshow(src_rgb, alpha=0.3)  # 半透明背景

        # 绘制变换后的网格
        grid_transformed_reshaped = grid_transformed.reshape(8, 10, 2)
        for i in range(8):
            row_points = grid_transformed_reshaped[i]
            ax2.plot(row_points[:, 0], row_points[:, 1], "g-", alpha=0.7)

        for i in range(10):
            col_points = grid_transformed_reshaped[:, i]
            ax2.plot(col_points[:, 0], col_points[:, 1], "g-", alpha=0.7)

        # 绘制变换后的边界
        corners_transformed_closed = np.vstack([corners_transformed, corners_transformed[0]])
        ax2.plot(corners_transformed_closed[:, 0], corners_transformed_closed[:, 1], "r-", linewidth=3, label="变换后边界")
        ax2.set_title("单应矩阵变换", fontsize=vis_config.font_size)
        ax2.axis("off")
        ax2.legend()

        # 3. 应用变换的图像
        ax3 = axes[2]
        transformed_img = cv2.warpPerspective(src_rgb, homography, (w, h))
        ax3.imshow(transformed_img)
        ax3.set_title("变换后图像", fontsize=vis_config.font_size)
        ax3.axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=vis_config.save_dpi, bbox_inches="tight")

        return fig

    @staticmethod
    def create_alignment_report(
        src_img: Union[Image.Image, str, Path, np.ndarray],
        target_img: Union[Image.Image, str, Path, np.ndarray],
        result_dict: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> str:
        """
        创建对齐报告
        """
        stats = result_dict["statistics"]

        report = f"""
图像对齐报告
=================================

对齐状态: {stats['alignment_status']}

特征检测:
- 源图像特征点数: {stats['num_src_features']}
- 目标图像特征点数: {stats['num_target_features']}
- 匹配点数: {stats['num_matches']}

单应矩阵质量:
- 条件数: {stats.get('homography_condition', 'N/A')}

对齐质量评估:
"""

        if "ssim_score" in stats:
            report += f"- SSIM分数: {stats['ssim_score']:.4f}\n"
        if "mse" in stats:
            report += f"- 均方误差: {stats['mse']:.4f}\n"
        if "is_good_alignment" in stats:
            report += f"- 对齐质量: {'良好' if stats['is_good_alignment'] else '需要改进'}\n"

        report += "\n推荐:n"
        if stats["alignment_status"] != "success":
            report += "- 对齐失败，建议调整参数或检查图像质量\n"
        elif stats.get("homography_condition", float("inf")) > 1e5:
            report += "- 单应矩阵条件数较高，变换可能不稳定\n"
        elif stats.get("ssim_score", 0) < 0.7:
            report += "- SSIM分数较低，建议检查对齐结果\n"
        else:
            report += "- 对齐成功，质量良好\n"

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report

    # 辅助函数
    @staticmethod
    def _draw_keypoints(image: np.ndarray, keypoints: List, vis_config: VisualizationConfig) -> np.ndarray:
        """绘制特征点"""
        img_copy = image.copy()
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(img_copy, (x, y), vis_config.keypoint_size, (255, 0, 0), -1)
        return img_copy

    @staticmethod
    def _draw_matches(
        img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List, matches: List, vis_config: VisualizationConfig
    ) -> np.ndarray:
        """绘制匹配点"""
        # 限制显示的匹配点数量
        display_matches = matches[: vis_config.max_matches_display]

        # 创建并排图像
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2

        combined_img = np.zeros((h, w, 3), dtype=np.uint8)
        combined_img[:h1, :w1] = img1
        combined_img[:h2, w1 : w1 + w2] = img2

        # 绘制匹配线
        for match in display_matches:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, (kp2[match.trainIdx].pt[0] + w1, kp2[match.trainIdx].pt[1])))

            # 根据匹配距离选择颜色
            color = (0, 255, 0) if match.distance < 100 else (0, 0, 255)
            cv2.line(combined_img, pt1, pt2, color, 1)

            # 绘制关键点
            cv2.circle(combined_img, pt1, 3, (255, 0, 0), -1)
            cv2.circle(combined_img, pt2, 3, (255, 0, 0), -1)

        return combined_img

    @staticmethod
    def _create_blend_image(img1: np.ndarray, img2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """创建混合图像"""
        return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)


# 便捷函数
def quick_alignment_visualization(
    src_img: Union[Image.Image, str, Path, np.ndarray],
    target_img: Union[Image.Image, str, Path, np.ndarray],
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    快速对齐可视化

    Args:
        src_img: 源图像
        target_img: 目标图像
        save_dir: 保存目录

    Returns:
        包含所有结果的字典
    """
    # 创建保存路径
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        process_path = Path(save_dir) / "alignment_process.png"
        report_path = Path(save_dir) / "alignment_report.txt"
    else:
        process_path = None
        report_path = None

    # 执行完整的可视化
    result = ImageSiftVisualizer.visualize_alignment_process(src_img, target_img, save_path=process_path)

    # 生成报告
    report = ImageSiftVisualizer.create_alignment_report(src_img, target_img, result, save_path=report_path)

    result["report"] = report

    return result
