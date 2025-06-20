import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage
from sklearn.preprocessing import binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_similarity import ImageSimilarity
from auto_image_edit.utils.image_estimate import ImageEstimate
from eval.mask_eval import post_process_mask


# 新增：通用函数
def generate_heatmap(src, dst, output_path, compare_func, **kwargs):
    if os.path.exists(output_path):
        os.remove(output_path)
    heatmap = compare_func(src, dst, heatmap=False, **kwargs)

    # 有值的区域标为1, 无值的区域标为0
    heatmap = np.where(heatmap > 0, 1, 0)

    # 通过连通区域判断，仅保留面积(像素点)大于2的区域
    labels, num = ndimage.label(heatmap)
    sizes = ndimage.sum(heatmap, labels, range(1, num + 1))
    mask_sizes = sizes >= 2

    # 安全处理索引
    filtered = np.zeros_like(labels, dtype=bool)
    filtered[labels > 0] = mask_sizes[labels[labels > 0] - 1]
    heatmap_filtered = np.where(filtered, 1, 0)
    return heatmap_filtered


def generate_np(src, dst, compare_func, **kwargs):
    return compare_func(src, dst, **kwargs)


# 原有代码块改为调用函数
if __name__ == "__main__":
    # src = "/home/yuyangxin/data/dataset/CocoGlide/Au/clock_267351.png"
    # dst = "/home/yuyangxin/data/dataset/CocoGlide/Tp/glide_inpainting_val2017_267351_up.png"
    # dst = "/home/yuyangxin/data/dataset/CocoGlide/glide_inpainting_val2017_267351_up_quality_90.jpg"
    src = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Au/real_069.jpg"
    dst = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Tp/fake_069.jpg"

    dst = Path(dst)
    # generate_heatmap(src, dst, "./output/lpips_heatmap.png", ImageSimilarity.compare_images_lpips, norm="zscore")
    lpips_diff = ImageSimilarity.compare_images_lpips(src, dst, heatmap=False, norm="zscore", gray=False)
    pixel_diff = ImageSimilarity.compare_images_pixelwise(src, dst, heatmap=False, norm="zscore", gray=False, color_space="LAB")
    # ssim_diff = ImageSimilarity.compare_images_ssim(src, dst, heatmap=False, norm="zscore", gray=False)
    # rule_multi_diff = np.where((pixel_diff > 0) & (lpips_diff > 0), lpips_diff * pixel_diff * 255 * 255, lpips_diff + pixel_diff)
    # res = ImageSimilarity.norm_method(rule_multi_diff, norm_method="zscore")

    # res = entropy_fusion(pixel_diff, lpips_diff)

    # # 相加取均值
    # res = np.where(pixel_diff > 0, lpips_diff, pixel_diff)
    # res = np.where(pixel_diff > 0, lpips_diff, pixel_diff)
    res = (pixel_diff + lpips_diff) / 2
    res = post_process_mask(res, morphology_iterations=3, morphology_kernel_size=3)
    ImageSimilarity.to_heatmap(res).save("./output/lab_heatmap.png")
    ImageSimilarity.to_gray(res).save("./output/lab_pixel.png")

    # # # 相加条件：当任一差异指标为0时，采用相加操作保留非零指标的贡献
    # # # 意义：避免因某一指标失效（如颜色未变但结构篡改）导致整体差异被抑制
    # # # 相乘条件：当两个指标均非零时，采用相乘放大协同差异
    # # # 意义：强调同时被像素级和语义级方法检测到的区域（高置信差异）
    # # # 创建掩码：判断哪些位置的值为0（或接近0）
    # # threshold = 1e-6  # 防止浮点数精度问题
    # # lpips_zero_mask = np.abs(lpips) < threshold
    # # pixel_zero_mask = np.abs(pixel) < threshold

    # # # 当任一指标为0时使用相加，否则使用相乘
    # # res = np.where(lpips_zero_mask | pixel_zero_mask, lpips + pixel, lpips * pixel)  # 相加保留非零指标  # 相乘放大协同差异

    # # # 归一化到0-1
    # # res = ImageSimilarity.norm_method(res, norm_method="minmax")

    # # 获取平滑银子
    # sobel_factor = ImageSimilarity.get_canny(dst, norm_method="zscore")

    # # 获取res
    # res = (1 - sobel_factor) * lpips + sobel_factor * pixel
    # # res = sobel_factor * lpips + (1 - sobel_factor) * pixel

    # # 保存为热力图
    # res_img = ImageSimilarity.to_heatmap(res)
    # res_img.save("./output/lpips_pixelwise_canny_heat.png")

    # res_img = ImageSimilarity.to_grey(res)
    # res_img.save("./output/lpips_pixelwise_canny_grey.png")
