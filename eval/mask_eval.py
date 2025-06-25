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
import threading
from scipy.stats import entropy

from scipy import ndimage
import sys

from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_similarity import ImageSimilarity
from auto_image_edit.utils.image_estimate import ImageEstimate
from auto_image_edit.utils.image_mask_gen import ImageMaskGen


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


def pre_process_mask(real_img, fake_img, target_size=(2048, 2048)):
    """
    target_size: 目标尺寸(64,64) - (1024,1024)
    """
    # real_img = Image.open(real_img).convert("RGB")
    # original_size = (real_img.width, real_img.height)  # 保存原始尺寸
    # real_img = real_img.resize(target_size, Image.LANCZOS)
    # real_img = real_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸

    # fake_img = Image.open(fake_img).convert("RGB")
    # original_size = (fake_img.width, fake_img.height)  # 保存原始尺寸
    # fake_img = fake_img.resize(target_size, Image.LANCZOS)
    # fake_img = fake_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸
    return real_img, fake_img


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

        real_img, fake_img = pre_process_mask(item["real_image"], item["fake_image"], target_size=(1024, 1024))

        # 对每种type计算差异
        diffs = {}
        lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm=norm, gray=False)
        pixel_diff = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="lab")

        ## 线性组合
        diffs["lpips_replace_pixel"] = ImageMaskGen.lpips_replace_pixel(pixel_diff, lpips_diff)
        diffs["pixel_multiplication"] = ImageMaskGen.pixel_multiplication(pixel_diff, lpips_diff)
        diffs["rule_multiplication"] = ImageMaskGen.rule_multiplication(pixel_diff, lpips_diff)

        canny_diff = ImageSimilarity.get_canny(fake_img)
        diffs["adapter"] = ImageMaskGen.adapter(pixel_diff, lpips_diff, canny_diff)

        diffs["entropy_fusion"] = ImageMaskGen.entropy_fusion(pixel_diff, lpips_diff)
        diffs["linear_combination"] = ImageMaskGen.linear_weight(pixel_diff, lpips_diff)
        # 对每种type计算指标
        results = {}
        for key, item in diffs.items():
            results[key] = ImageEstimate.metric(gt_mask, item)
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
    main("/data0/yuyangxin/dataset/coverage/ins.json", norm="zscore", max_workers=1)
    main("/home/yuyangxin/data/dataset/CocoGlide/cocoGlide.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_90.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_75.json", norm="zscore", max_workers=32)
