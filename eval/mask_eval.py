# 评估Mask的生成是否有效
# 1. 读取json文件
from collections import defaultdict
import json
import os
import cv2
import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.preprocessing import binarize
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
import threading
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pointbiserialr, spearmanr
from scipy.stats import entropy
from skimage import filters
from skimage.morphology import disk
from scipy import ndimage
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_similarity import ImageSimilarity
from auto_image_edit.utils.image_estimate import ImageEstimate


def metric(gt_mask, pred_mask, threshold=0):
    """
    计算IOU和F1分数，使用库函数优化

    Args:
        gt_mask: 真实mask，numpy数组
        pred_mask: 预测mask，numpy数组
        threshold: 二值化阈值

    Returns:
        dict: 包含多个评估指标的字典
    """
    # 使用sklearn的binarize函数进行二值化
    gt_binary = binarize(gt_mask.reshape(-1, 1), threshold=threshold).flatten().astype(int)
    pred_binary = binarize(pred_mask.reshape(-1, 1), threshold=threshold).flatten().astype(int)

    # 使用sklearn计算多个指标
    iou = jaccard_score(gt_binary, pred_binary, average="binary", zero_division=0).item()
    f1 = f1_score(gt_binary, pred_binary, average="binary", zero_division=0)
    precision = precision_score(gt_binary, pred_binary, average="binary", zero_division=0)
    recall = recall_score(gt_binary, pred_binary, average="binary", zero_division=0)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def metric_auc(gt_mask, pred_mask):

    # 确保输入是一维数组
    gt_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    # 检查是否为常数数组
    if len(np.unique(gt_mask_flat)) == 1 or len(np.unique(pred_mask_flat)) == 1:
        raise ValueError("输入数组不能是常数数组")

    auc_roc = roc_auc_score(gt_mask_flat, pred_mask_flat)
    auc_pr = average_precision_score(gt_mask_flat, pred_mask_flat)

    # 点二列相关系数
    correlation, p_value = pointbiserialr(gt_mask_flat, pred_mask_flat)
    if (-1 <= correlation.item() <= 1) is False:
        raise ValueError("点二列相关系数应在[-1, 1]范围内")
    # # 斯皮尔曼相关系数
    # spearmanr_corr, spearman_p_value = spearmanr(gt_mask_flat, pred_mask_flat)
    return {
        "pointbiserialr": correlation.item(),
        "auc_roc": auc_roc.item(),
        "auc_pr": auc_pr.item(),
    }


def entropy_fusion(diff_map, lpips_map, window_size=5):
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
    maps = [diff_map, lpips_map]
    if not all(isinstance(img, np.ndarray) and img.ndim == 2 for img in maps):
        raise ValueError("所有输入必须是2D numpy数组")

    shape = diff_map.shape
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


def post_process_mask(
    mask,
    morphology_kernel_size=2,
    morphology_iterations=1,
    min_area_threshold=2,  # 新增：最小连通区域面积阈值
    use_connected_components=True,  # 新增：是否使用连通组件分析
):
    """
    会降低性能 -- 不处理
    对掩码图像进行后处理: 使用形态学操作和连通组件分析去除孤立噪点。

    Args:
        mask: 输入掩码图像 (numpy array)
        morphology_kernel_size: 形态学操作核大小
        morphology_iterations: 形态学操作迭代次数
        min_area_threshold: 最小连通区域面积阈值，小于此值的区域将被移除
        use_connected_components: 是否使用连通组件分析
    Returns:
        processed_mask: 处理后的掩码图像
    """
    # 确保输入是numpy数组
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)

    # 转换为单通道灰度图
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # 二值化处理
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 方法1: 形态学操作 - 开运算去除小噪点
    if morphology_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel, iterations=morphology_iterations)

        # 可选：闭运算填补空洞
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 方法2: 连通组件分析去除小区域
    if use_connected_components and min_area_threshold > 0:
        # 查找连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)

        # 创建输出mask
        filtered_mask = np.zeros_like(mask_binary)

        # 保留面积大于阈值的连通组件（跳过背景标签0）
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area_threshold:
                filtered_mask[labels == i] = 255

        mask_binary = filtered_mask

    # 转换回原始数据类型
    return (mask_binary / 255.0).astype(np.float32)


def process_single_item(item, norm):
    """
    处理单个数据项的函数，用于并发执行

    Args:
        item: 单个数据项
        types: 要处理的类型列表
        norm: 归一化方法

    Returns:
        dict: 各类型的评估结果
    """
    try:
        gt_mask = ImageSimilarity.load_image_array(item["gt_mask"], color_space="L")
        # 转为二值化mask
        gt_mask = (gt_mask > 0).astype(np.float32)
        # 只要H, W,不要C
        gt_mask = gt_mask[:, :, 0] if gt_mask.ndim == 3 else gt_mask

        real_img = item["real_image"]
        fake_img = item["fake_image"]

        # 对每种type计算差异
        diffs = {}

        lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm=norm, gray=False)
        pixel_diff = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="lab")
        ssim_diff = ImageSimilarity.compare_images_ssim(real_img, fake_img, heatmap=False, norm=norm, gray=False)

        diffs["pixel"] = pixel_diff
        diffs["lpips"] = lpips_diff
        diffs["ssim"] = ssim_diff

        # # rule_replace type
        # diffs["lpips_replace_pixel"] = np.where(pixel_diff > 0, lpips_diff, pixel_diff)

        # # pixel_multiplication type
        # diffs["pixel_multiplication"] = lpips_diff * pixel_diff

        # # rule_multiplication type
        # rule_multi_diff = np.where((pixel_diff > 0) & (lpips_diff > 0), lpips_diff * pixel_diff * 255 * 255, lpips_diff + pixel_diff)
        # diffs["rule_multiplication"] = ImageSimilarity.norm_method(rule_multi_diff, norm_method=norm)

        # # adapter type
        # canny = ImageSimilarity.get_canny(real_img)
        # diffs["adapter"] = canny * lpips_diff + (1 - canny) * pixel_diff

        ## 信息熵线性组合
        # diffs["entropy_fusion"] = entropy_fusion(pixel_diff, lpips_diff, window_size=5)

        ## 线性组合
        diffs["linear_combination"] = (pixel_diff + lpips_diff) / 2.0

        post_diff = post_process_mask(
            diffs["linear_combination"],
            morphology_kernel_size=1,
            morphology_iterations=1,
            use_connected_components=False,
            min_area_threshold=2,
        )
        diffs["linear_combination_opt"] = (post_diff + lpips_diff) / 2.0

        # ## 自适应乘法
        # canny = ImageSimilarity.get_canny(real_img)
        # diffs["adapter_multiplication"] = canny * lpips_diff * (1 - canny) * pixel_diff

        # 对每种type计算指标
        results = {}
        for key, item in diffs.items():
            results[key] = metric_auc(gt_mask, item)
        return results

    except Exception as e:
        print(f"Error processing item: {e}")
        return None


def main(json_file, norm="zscore", max_workers=None):
    """
    主函数，支持多线程并发处理

    Args:
        json_file: JSON文件路径
        norm: 归一化方法
        max_workers: 最大工作线程数，默认为CPU核心数*2
    """
    # 检查文件是否存在
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON文件不存在: {json_file}")

    print(f"开始处理文件: {json_file}")

    # 读取数据
    try:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {e}")

    if not data:
        print("警告: JSON文件为空")
        return {}, {}

    print(f"数据总量: {len(data)}")

    # 存储所有结果，按方法类型分组
    all_results = defaultdict(lambda: defaultdict(list))
    processed_count = 0
    error_count = 0

    # 创建总进度条
    if max_workers is None:
        for item in data:
            result = process_single_item(item, norm)
            if result is not None:
                processed_count += 1
                for method_name, metrics in result.items():
                    for metric_name, value in metrics.items():
                        all_results[method_name][metric_name].append(value)
            else:
                error_count += 1
    else:
        max_workers = min(os.cpu_count() * 2, len(data), 32)  # 限制最大线程数
        with tqdm(total=len(data), desc="Processing items", unit="item") as pbar:
            # 使用ThreadPoolExecutor进行并发处理
            with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="MaskEval") as executor:
                # 提交所有单个任务
                future_to_index = {}
                for idx, item in enumerate(data):
                    future = executor.submit(process_single_item, item, norm)
                    future_to_index[future] = idx

                # 收集结果
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        if result is not None:
                            processed_count += 1
                            for method_name, metrics in result.items():
                                for metric_name, value in metrics.items():
                                    all_results[method_name][metric_name].append(value)
                        else:
                            error_count += 1

                    except Exception as e:
                        error_count += 1
                        print(f"错误: 处理索引 {idx} 时出现异常: {e}")

                    # 更新进度条
                    pbar.update(1)

    # 检查是否有有效结果
    if not all_results:
        print("错误: 没有获得任何有效结果")
        return {}, {}

    # 获取所有方法和指标名称
    method_names = list(all_results.keys())
    metric_names = set()
    for method_results in all_results.values():
        metric_names.update(method_results.keys())
    metric_names = sorted(list(metric_names))

    # 计算平均值
    table_data = {}
    for method_name in method_names:
        table_data[method_name] = {}
        for metric_name in metric_names:
            values = all_results[method_name].get(metric_name, [])
            if values:
                avg_value = np.mean(values)
                table_data[method_name][metric_name] = avg_value
            else:
                table_data[method_name][metric_name] = 0.0

    # # 输出评估结果表格 - 按方法分组显示
    # print(f"\n=== 评估结果平均值 (按方法分组) ===")
    # print(f"{'Method':<20} {'Metric':<15} {'Mean':<12}")
    # print("=" * 47)

    # for method_name in method_names:
    #     print(f"{method_name:<20} {'':<15} {'':<12}")  # 方法名行
    #     for metric_name in metric_names:
    #         avg_value = table_data[method_name][metric_name]
    #         print(f"{'':>20} {metric_name:<15} {avg_value:<12.4f}")
    #     print("-" * 47)

    # 输出评估结果表格 - 按指标分组显示（便于对比）
    print(f"\n=== 评估结果平均值 (按指标分组对比) ===")
    header = f"{'Metric':<30}"
    for method_name in method_names:
        header += f"{method_name:<30}"
    print(header)
    print("=" * (15 + 12 * len(method_names)))

    for metric_name in metric_names:
        row = f"{metric_name:<30}"
        for method_name in method_names:
            avg_value = table_data[method_name][metric_name]
            row += f"{avg_value:<30.4f}"
        print(row)

    # 输出处理统计信息
    print(f"\n=== 处理统计 ===")
    print(f"成功处理: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%)")

    if error_count > 0:
        print(f"处理失败: {error_count}")

    return all_results, table_data


if __name__ == "__main__":
    # 使用多线程处理，可以自定义参数
    main("/data0/yuyangxin/dataset/coverage/ins.json", norm="zscore", max_workers=32)
    main("/home/yuyangxin/data/dataset/CocoGlide/cocoGlide.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_90.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_75.json", norm="zscore", max_workers=32)
