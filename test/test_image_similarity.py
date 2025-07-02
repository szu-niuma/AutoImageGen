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
from auto_image_edit.utils.image_sift import ImageSift
from auto_image_edit.utils.image_mask_gen import ImageMaskGen


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


def pre_process_mask(real_img, fake_img, target_size=(256, 256)):
    """
    target_size: 目标尺寸(64,64) - (1024,1024)
    """
    real_img = Image.open(real_img).convert("RGB")
    original_size = (real_img.width, real_img.height)  # 保存原始尺寸
    real_img = real_img.resize(target_size, Image.LANCZOS)
    real_img = real_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸

    fake_img = Image.open(fake_img).convert("RGB")
    fake_img = fake_img.resize(target_size, Image.LANCZOS)
    fake_img = fake_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸
    return real_img, fake_img


# 原有代码块改为调用函数
if __name__ == "__main__":
    real_img = "/data0/yuyangxin/dataset/FragFake/Image/original/original_broccoli_000000427521.jpg"
    fake_img = "/data0/yuyangxin/dataset/FragFake/Image/UltraEdit/hard/addition/broccoli_000000427521_addition.jpg"
    # dst = "/home/yuyangxin/data/dataset/CocoGlide/glide_inpainting_val2017_267351_up_quality_90.jpg"
    # src = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Au/real_069.jpg"
    # dst = "/home/yuyangxin/data/dataset/custom_dataset/llm_edit/Tp/fake_069.jpg"
    real_img, fake_img = pre_process_mask(real_img, fake_img, target_size=(256, 256))
    real_img.save("./real_img.png")
    fake_img.save("./fake_img.png")
    # 将real_img和fake_img转换为PIL Image对象
    lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm="zscore", gray=False, align=True)
    pixel_diff = ImageSimilarity.compare_images_pixelwise(
        real_img, fake_img, heatmap=False, norm="zscore", gray=False, color_space="LAB", align=True
    )

    # 自适应融合
    sigmoid_weight = ImageMaskGen.sigmoid_weight(pixel_diff, lpips_diff)
    sigmoid_weight_gray = ImageSimilarity.to_gray(sigmoid_weight)
    sigmoid_weight_gray.save("./sigmoid_weight_gray.png")
    sigmoid_weight_heatmap = ImageSimilarity.to_heatmap(sigmoid_weight)
    sigmoid_weight_heatmap.save("./sigmoid_weight_heatmap.png")

    # 算子组合
    canny_diff = ImageSimilarity.get_canny(real_img)
    canny_adapter = ImageMaskGen.adapter(pixel_diff, lpips_diff, canny_diff)
    canny_adapter_gray = ImageSimilarity.to_gray(canny_adapter)
    canny_adapter_gray.save("./canny_adapter_gray.png")
    canny_adapter_heatmap = ImageSimilarity.to_heatmap(canny_adapter)
    canny_adapter_heatmap.save("./canny_adapter_heatmap.png")

    # 线性组合
    linear_weight = ImageMaskGen.linear_weight(pixel_diff, lpips_diff, weight=0.5)  # 处理掩码
    linear_weight_gray = ImageSimilarity.to_gray(linear_weight)
    linear_weight_gray.save("./linear_weight_gray.png")
    linear_weight_heatmap = ImageSimilarity.to_heatmap(linear_weight)
    linear_weight_heatmap.save("./linear_weight_heatmap.png")

    # 乘积平方
    square_multiplication = ImageMaskGen.rule_multiplication(pixel_diff, lpips_diff, alpha=0.5)
    square_multiplication_gray = ImageSimilarity.to_gray(square_multiplication)
    square_multiplication_gray.save("./square_multiplication_gray.png")
    square_multiplication_heatmap = ImageSimilarity.to_heatmap(square_multiplication)
    square_multiplication_heatmap.save("./square_multiplication_heatmap.png")
