# 评估Mask的生成是否有效
# 1. 读取json文件
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
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


def pre_process_mask(real_img, fake_img, target_size=(1024, 1024)):
    """
    target_size: 目标尺寸(64,64) - (1024,1024)
    """
    real_img = Image.open(real_img).convert("RGB")
    original_size = (real_img.width, real_img.height)  # 保存原始尺寸
    real_img = real_img.resize(target_size, Image.LANCZOS)
    real_img = real_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸

    fake_img = Image.open(fake_img).convert("RGB")
    original_size = (fake_img.width, fake_img.height)  # 保存原始尺寸
    fake_img = fake_img.resize(target_size, Image.LANCZOS)
    fake_img = fake_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸
    return real_img, fake_img


def process_single_item(item, norm, output="./error_example"):
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

        real_img, fake_img = item["real_image"], item["fake_image"]

        # 计算不同颜色空间的像素差异并保存到字典中
        diffs = {}
        diffs["lpips"] = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm=norm, gray=False)
        diffs["rgb"] = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="rgb")
        diffs["hsv"] = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="hsv")
        diffs["ycbcr"] = ImageSimilarity.compare_images_pixelwise(
            real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="ycbcr"
        )
        diffs["lab"] = ImageSimilarity.compare_images_pixelwise(
            real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="lab", ciede_version="ciede76"
        )
        diffs["lab94"] = ImageSimilarity.compare_images_pixelwise(
            real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="lab", ciede_version="ciede94"
        )
        diffs["lab2000"] = ImageSimilarity.compare_images_pixelwise(
            real_img, fake_img, heatmap=False, norm=norm, gray=False, color_space="lab", ciede_version="ciede2000"
        )

        # 对每种type计算指标
        results = {}
        for key, diff_item in diffs.items():
            results[key], judge = ImageEstimate.metric_colorspace(gt_mask, diff_item)
            # if judge is False:
            #     try:
            #         # 评估不过关, 保存对应的图片
            #         file_path = Path(item["real_image"])
            #         file_type = file_path.parent.parent.name  # 获取文件所在目录名作为类型

            #         # 创建保存目录结构
            #         save_dir = os.path.join(output, file_type, key, file_path.stem)
            #         os.makedirs(save_dir, exist_ok=True)

            #         # 构建输出路径
            #         real_img_path = f"{save_dir}/real_image.png"
            #         fake_img_path = f"{save_dir}/fake_image.png"
            #         gt_mask_path = f"{save_dir}/gt_mask.png"
            #         pred_mask_path = f"{save_dir}/pred_mask.png"

            #         # 保存原图和伪造图
            #         shutil.copy(item["real_image"], real_img_path)
            #         shutil.copy(item["fake_image"], fake_img_path)

            #         # 保存掩码图像
            #         Image.fromarray((gt_mask * 255).astype(np.uint8)).save(gt_mask_path)
            #         Image.fromarray((diff_item * 255).astype(np.uint8)).save(pred_mask_path)
            #     except Exception as e:
            #         print(f"保存图像时出错: {e}")

        return results

    except Exception as e:
        print(f"Error processing item: {e}")
        return None


def main(json_file, norm="zscore", max_workers=None, csv_output=None):
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

    # 输出评估结果表格 - 按方法分组显示
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
    header = f"{'Metric':<20}"
    for method_name in method_names:
        header += f"{method_name:<20}"
    print(header)
    print("=" * (15 + 12 * len(method_names)))

    for metric_name in metric_names:
        row = f"{metric_name:<20}"
        for method_name in method_names:
            avg_value = table_data[method_name][metric_name]
            row += f"{avg_value:<20.4f}"
        print(row)

    # 输出处理统计信息
    print(f"\n=== 处理统计 ===")
    print(f"成功处理: {processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%)")

    if error_count > 0:
        print(f"处理失败: {error_count}")

    # 保存结果到csv文件
    if csv_output:
        try:
            with open(csv_output, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)

                # 写入表头
                header_row = ["Metric"] + method_names
                writer.writerow(header_row)

                # 写入数据行
                for metric_name in metric_names:
                    row = [metric_name]
                    for method_name in method_names:
                        row.append(table_data[method_name][metric_name])
                    writer.writerow(row)

                # 写入统计信息
                writer.writerow([])  # 空行
                writer.writerow(["处理统计"])
                writer.writerow(["成功处理", f"{processed_count}/{len(data)} ({processed_count/len(data)*100:.1f}%)"])
                if error_count > 0:
                    writer.writerow(["处理失败", error_count])

            print(f"\n结果已保存到CSV文件: {csv_output}")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")

    return all_results, table_data


if __name__ == "__main__":
    main(
        "/data0/yuyangxin/dataset/coverage/ins.json",
        norm="zscore",
        max_workers=32,
        csv_output="./coverage_ins_eval.csv",
    )
    main(
        "/home/yuyangxin/data/dataset/CocoGlide/cocoGlide.json",
        norm="zscore",
        max_workers=32,
        csv_output="./cocoGlide_eval.csv",
    )
    main(
        "/data0/yuyangxin/dataset/AutoSplice/auto_splice.json",
        norm="zscore",
        max_workers=32,
        csv_output="./auto_splice_eval.csv",
    )
    # main("/home/yuyangxin/data/dataset/CocoGlide/cocoGlide.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_90.json", norm="zscore", max_workers=32)
    # main("/data0/yuyangxin/dataset/AutoSplice/auto_splice_75.json", norm="zscore", max_workers=32)
