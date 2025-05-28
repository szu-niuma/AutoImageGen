import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

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
            # 单线程处理
            for info in tqdm(valid_files, desc="处理中", disable=len(valid_files) == 1):
                try:
                    self.pipeline(info)
                    logger.debug(f"成功处理文件: {info['image_path']}")
                except Exception as e:
                    logger.error(f"处理文件失败 {info['image_path']}: {str(e)}", exc_info=True)
                    # 可以选择是否继续处理其他文件
                    continue
        else:
            # 多线程处理
            max_workers = min(os.cpu_count() or self.DEFAULT_WORKERS, len(valid_files), self.DEFAULT_WORKERS)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.pipeline, file_info): file_info for file_info in valid_files}

                # 统计处理结果
                success_count = 0
                error_count = 0

                with tqdm(total=len(valid_files), desc="处理中", unit="文件") as pbar:
                    for future in as_completed(future_to_file):
                        file_info = future_to_file[future]
                        try:
                            future.result()
                            success_count += 1
                            logger.debug(f"成功处理文件: {file_info['image_path']}")
                        except Exception as e:
                            error_count += 1
                            logger.error(f"处理文件失败 {file_info['image_path']}: {str(e)}", exc_info=True)
                        finally:
                            pbar.update(1)
                            pbar.set_postfix({"成功": success_count, "失败": error_count})

                logger.info(f"处理完成 - 成功: {success_count}, 失败: {error_count}, 总计: {len(valid_files)}")
