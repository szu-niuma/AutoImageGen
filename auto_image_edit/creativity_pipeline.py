import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from .processor import AuthorityAssess, ImageAnalysis, ImageConsultant
from .utils.image_processor import ImageProcessor
from .utils.util import load_json, save_json


class CreativityPipeline:
    """
    判断图像真实性
    ↓
    明确可以判断为真
    ↓
    为每个编辑方式生成一条编辑想法
    给出一个自己独特的编辑想法
    ↓
    输出文件保存 (创意一部分 / 对话一部分)
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    DEFAULT_WORKERS = 16

    def __init__(self, config: dict, out_dir="output", is_debug: bool = False, max_num: int = None):
        self.image_processor = ImageProcessor()
        self.is_debug = is_debug
        # 图像真实性判别器
        self.authority_assessor = AuthorityAssess(config)

        # 图像内容表述器
        self.image_analyst = ImageAnalysis(config)

        # 创建图像创意生成器
        self.image_consultant = ImageConsultant(config)

        self.output_dir: Path = Path(out_dir) / "CreativityPipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_num = max_num if max_num is not None else float("inf")

    def pipeline(self, info: dict) -> dict:
        image_path = Path(info["image_path"])
        save_path = self.output_dir / f"{image_path.stem}.json"
        ret = load_json(save_path, raise_exception=False)
        if ret:
            logger.info(f"已存在结果文件: {save_path}")
            return ret
        src_img, _, _ = self.image_processor.load_image(image_path)
        image_base64 = self.image_processor.get_base64(src_img)

        annotations = info.get("annotations", None)
        if annotations:
            annotations = "\n".join([f"category: {ann['category_name']}" for ann in annotations])
        else:
            annotations = self.image_analyst.run(image_path, image_base64=image_base64)

        creativity_info = self.image_consultant.run(
            image_path,
            image_analysis=annotations,
            image_base64=image_base64,
        )
        authorities_info = self.authority_assessor.run(image_path, image_base64=image_base64)
        info = {
            "authorities": authorities_info,
            "analysis": info.get("annotations", None) or annotations,
            "creativity": creativity_info,
        }
        # 保存结果到json文件
        save_json(save_path, info)
        logger.info(f"处理完成: {image_path} -> {save_path}")
        return info

    def read_image(self, image_path: Path) -> Any:
        # 构造待处理路径列表 - 按需遍历，达到目标数量即停止
        valid_files = []
        if image_path.is_dir():
            for file_path in image_path.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTS:
                    valid_files.append({"image_path": file_path})
                    # 达到最大数量时停止遍历
                    if self.max_num and len(valid_files) >= self.max_num:
                        break
        else:
            if image_path.suffix.lower() in self.IMAGE_EXTS:
                valid_files = [{"image_path": image_path}]
        return valid_files

    def read_json(self, json_path: Path) -> list:
        data = load_json(json_path)
        return data

    def run(self, image_path: Path) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"图像路径不存在: {image_path}")
            raise FileNotFoundError(f"Image path not found: {image_path}")

        # 判断是否是json文件
        if image_path.suffix.lower() == ".json":
            valid_files = self.read_json(image_path)
        else:
            valid_files = self.read_image(image_path)

        if self.is_debug:
            # 单线程
            for info in valid_files:
                self.pipeline(info)
        else:
            # 多线程执行
            max_workers = min(os.cpu_count() or 16, len(valid_files))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.pipeline, f): f for f in valid_files}
                for future in as_completed(future_to_file):
                    info = future_to_file[future]
                    future.result()
