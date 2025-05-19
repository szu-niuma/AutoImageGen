import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


class ImageProcessor:
    def __init__(self, max_width=None, max_height=None, font_size=40):
        """
        初始化ImageProcessor类。
        """
        self.max_width = max_width
        self.max_height = max_height
        self.font_path = "./resource/front/MSYH.TTC"
        self.font_size = font_size  # 可根据需求调节大小

    def resize_image(self, image):
        # 获取原始图像的宽度和高度
        original_width, original_height = image.size

        if self.max_width is None and self.max_height is None:
            return image, 1

        # 如果原始图像本身已经符合最大宽高要求，无需缩放
        if original_width <= self.max_width and original_height <= self.max_height:
            return image, 1

        # 计算宽度和高度的缩放比例
        width_ratio = self.max_width / original_width
        height_ratio = self.max_height / original_height

        # 使用较小的比例进行缩放，保证宽度和高度都不超过最大值
        scale_ratio = min(width_ratio, height_ratio)

        # 计算新的宽度和高度
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # 使用LANCZOS进行高质量缩放
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return resized_image, scale_ratio

    @staticmethod
    def is_url(path: str):
        """判断路径是否为 URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def get_base64(self, image, format_info="WEBP"):
        if isinstance(image, str) or isinstance(image, Path):
            image = self.load_image(image)[1]
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with BytesIO() as buffered:
            image.save(buffered, format=format_info)
            trans_image_webp = base64.b64encode(buffered.getvalue()).decode()
        return trans_image_webp

    def base64_to_image(self, b64_str: str) -> Image.Image:
        """
        将 Base64 字符串解码为 PIL Image
        """
        b64_str = b64_str.split(",")[-1]  # 去掉前缀部分
        data = base64.b64decode(b64_str)
        return Image.open(BytesIO(data))

    def load_image(self, image_file: Path, image_type="RGB"):
        """
        加载并预处理图片
        """
        image_file = Path(image_file)
        if self.is_url(str(image_file)):
            # 将图片下载到本地, 后进行处理
            src_image = Image.open(requests.get(image_file, stream=True).raw).convert(image_type)
        else:
            src_image = Image.open(image_file)
        trans_image, scale_ratio = self.resize_image(src_image)
        return src_image, trans_image, scale_ratio

    def process_mask(self, src_mask, target_mask, scale_ratio):
        def read_mask(mask):
            if isinstance(mask, (str, Path)):
                return self.load_image(mask)[0]
            return mask

        # 读取mask图像
        src_mask_img = read_mask(src_mask)
        target_mask_img = read_mask(target_mask)

        # 确保图像大小相同
        if src_mask_img.shape != target_mask_img.shape:
            raise ValueError("Source and target masks must have the same dimensions.")

        # 合并target_mask和target_mask_img, 要求相同的像素保留, 不同的像素取大值
        combined_mask = np.maximum(src_mask_img, target_mask_img)

        # 如果需要缩放比例，可以在此处应用缩放
        if scale_ratio != 1.0:
            new_size = (int(combined_mask.shape[1] * scale_ratio), int(combined_mask.shape[0] * scale_ratio))
            combined_mask = cv2.resize(combined_mask, new_size, interpolation=cv2.INTER_NEAREST)

        # 转为PIL图像
        combined_mask = Image.fromarray(combined_mask)
        return combined_mask

    @staticmethod
    def combine_images(src_img, mask_img):
        # mask_img是灰度图像, src_img是RGB图像
        # mask_img中为白色区域的保留src_img内容,黑色区域的去除src_img内容
        src_img = src_img.convert("RGBA")
        mask_img = mask_img.convert("L")

        # 创建一个新的图像，白色区域的alpha值为255，黑色区域的alpha值为0
        mask_rgba = Image.new("L", mask_img.size)
        mask_rgba.putdata([255 if pixel == 255 else 0 for pixel in mask_img.getdata()])

        # 使用mask_rgba作为掩码合成src_img和透明背景
        combined_image = Image.composite(src_img, Image.new("RGBA", src_img.size, (0, 0, 0, 0)), mask_rgba)

        return combined_image

    @staticmethod
    def get_dataset(dataset_dir: Path):
        """
        获取文件夹下的所有图片文件及其对应的掩码文件

        Args:
            dataset_dir (Path): 数据集目录路径

        Returns:
            list: 包含[图像文件路径, 掩码文件路径]对的列表
        """
        dataset_dir = Path(dataset_dir)
        # 常见的图像文件扩展名
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"]

        # 收集所有图像文件（包括大写后缀）
        image_files = []
        for ext in image_extensions:
            image_files.extend(dataset_dir.glob(f"*{ext}"))
            image_files.extend(dataset_dir.glob(f"*{ext.upper()}"))
        return image_files

    @staticmethod
    def download_image(url: str, save_path: Path):
        """
        下载图片并保存到指定路径
        """
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            print(f"Image downloaded and saved to {save_path}")
        else:
            print(f"Failed to download image from {url}")

    # def draw_box(self, image, infos, show_image=True):
    #     draw = ImageDraw.Draw(image)
    #     font = ImageFont.truetype(self.font_path, self.font_size)
    #     # 增加颜色列表
    #     colors = ["red", "green", "blue", "yellow", "orange", "pink", "purple", "brown", "gray", "beige"]
    #     width, height = image.size

    #     for idx, info in enumerate(infos.objects):
    #         # 归一化坐标 -> 像素
    #         raw_box = info.box_2d

    #         # 选择颜色
    #         color = colors[idx % len(colors)]

    #         # Convert normalized coordinates to absolute coordinates
    #         abs_y1 = int(raw_box[0] / 1000 * height)
    #         abs_x1 = int(raw_box[1] / 1000 * width)
    #         abs_y2 = int(raw_box[2] / 1000 * height)
    #         abs_x2 = int(raw_box[3] / 1000 * width)

    #         if abs_x1 > abs_x2:
    #             abs_x1, abs_x2 = abs_x2, abs_x1
    #         if abs_y1 > abs_y2:
    #             abs_y1, abs_y2 = abs_y2, abs_y1

    #         # 绘制边框
    #         draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
    #         # 文本尺寸
    #         draw.text((abs_x1 + 8, abs_y1 + 6), info.label, fill=color, font=font)

    #     if show_image:
    #         image.show()
    #     return image
