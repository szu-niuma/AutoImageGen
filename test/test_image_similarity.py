import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_similarity import ImageSimilarity
from auto_image_edit.utils.image_estimate import ImageEstimate


# 新增：通用函数
def generate_heatmap(src, dst, output_path, compare_func, **kwargs):
    """对比两张图并保存热力图"""
    if os.path.exists(output_path):
        os.remove(output_path)
    heatmap = compare_func(src, dst, heatmap=True, **kwargs)
    heatmap.save(output_path)


def generate_np(src, dst, compare_func, **kwargs):
    return compare_func(src, dst, **kwargs)


# 原有代码块改为调用函数
if __name__ == "__main__":
    src = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Au/real_069.jpg"
    dst = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Tp/fake_069.jpg"

    # generate_heatmap(src, dst, "./output/lpips_heatmap.png", ImageSimilarity.compare_images_lpips, norm="zscore")
    # generate_heatmap(src, dst, "./output/pixelwise_heatmap.png", ImageSimilarity.compare_images_pixelwise, norm="zscore")
    # generate_heatmap(src, dst, "./output/ssim_heatmap.png", ImageSimilarity.compare_images_ssim, norm="zscore")
    # generate_heatmap(src, dst, "./output/mse_heatmap.png", ImageSimilarity.compare_images_mse, norm="zscore")
    src_np = generate_np(src, dst, ImageSimilarity.compare_images_lpips, norm="zscore")
    dst_np = generate_np(src, dst, ImageSimilarity.compare_images_pixelwise, norm="zscore")

    # mse_value = ImageEstimate.mse_with_weight(src_np, dst_np)
    # mape_value = ImageEstimate.mape_with_weight(src_np, dst_np)
    # print(f"MSE: {mse_value}, MAPE: {mape_value}")

    # psnr = ImageEstimate.metric_psnr(src_np, dst_np)
    # ssim = ImageEstimate.metric_ssim(src_np, dst_np)
    # print(f"PSNR: {psnr}, SSIM: {ssim}")
    # thresholds = np.arange(0.3, 1.0, 0.1)
    # for threshold in thresholds:
    #     target = ImageSimilarity.threshold_gray(src_np, threshold_gray=threshold)
    #     target.save(f"./output/target_{threshold:.1f}.png")

    ap = ImageEstimate.metric_ap(src_np, dst_np, start_threshold=0.5, end_threshold=0.95, step=0.05)
    f1 = ImageEstimate.metric_f1(src_np, dst_np, start_threshold=0.5, end_threshold=0.95, step=0.05)
    iou = ImageEstimate.metric_iou(src_np, dst_np, start_threshold=0.5, end_threshold=0.95, step=0.05)
    print(f"AP: {ap}, F1: {f1}, IoU: {iou}")
