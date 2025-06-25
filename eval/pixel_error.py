# 识别像素误差
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image
from auto_image_edit.utils.image_similarity import ImageSimilarity
from auto_image_edit.utils.image_estimate import ImageEstimate


real_img = Path("/home/yuyangxin/data/AutoImageGen/resource/clock_real.png")
fake_img = Path("/home/yuyangxin/data/AutoImageGen/resource/clock_fake.png")
lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm="zscore", gray=False)
pixel_diff = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm="zscore", gray=False, color_space="LAB")
original_res = ImageSimilarity.to_gray((lpips_diff + pixel_diff) / 2)
original_res.save("pixel_diff.png")

# 先对fake_img进行下采样, 在对其上采样
fake_img = Image.open(fake_img).convert("RGB")
original_size = (fake_img.width, fake_img.height)  # 保存原始尺寸
fake_img = fake_img.resize((fake_img.width * 2, fake_img.height * 2), Image.LANCZOS)
fake_img = fake_img.resize(original_size, Image.LANCZOS)  # 恢复到原始尺寸

lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm="zscore", gray=False)
pixel_diff = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm=None, gray=False, color_space="LAB")
down_res = ImageSimilarity.to_gray((lpips_diff + pixel_diff) / 2)
down_res.save("pixel_diff_down.png")
