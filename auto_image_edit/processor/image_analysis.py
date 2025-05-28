import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger

from ..parser import ImageAnalystPrompt
from .base_processor import BaseImageProcessor


class ImageAnalysis(BaseImageProcessor):
    # 支持的图片后缀
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
    USER_PROMPT = "请分析这张图片并给出回答"

    def __init__(self, config: dict, out_dir, is_debug: bool = False, max_num=None):
        super().__init__(config, out_dir, "image_analysis", is_debug, max_num)
        self.image_analyst = ImageAnalystPrompt(self.llm)

    def _process_file(self, image_path: Path) -> Optional[dict]:
        """单文件处理逻辑，供多线程调用"""
        try:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)

            image_info = self.load_image_data(image_path, self.USER_PROMPT, image_base64)

            target_file = self.output_dir / f"{image_path.stem}.json"
            if target_file.exists():
                ret = json.loads(target_file.read_text(encoding="utf-8"))
            else:
                img_analyst = self.image_analyst.run(image_info, image_path.name).model_dump()
                ret = self.save_info(target_file, image_path, img_analyst)
            ret["image_base64"] = image_base64  # 将base64添加到结果中
            return ret
        except Exception as e:
            logger.error(f"处理图片 {image_path} 时发生错误: {e}")
            return None

    def run(self, image_path: Path) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"图像路径不存在: {image_path}")
            raise FileNotFoundError(f"Image path not found: {image_path}")

        # 构造待处理路径列表
        paths = list(image_path.iterdir()) if image_path.is_dir() else [image_path]
        # 过滤有效图片文件
        valid_files = [f for f in paths if f.is_file() and f.suffix.lower() in self.IMAGE_EXTS]
        valid_files = valid_files[: self.max_num] if self.max_num else valid_files
        results = {}

        # 单线程
        if self.is_debug:
            valid_files = valid_files[:2]  # 仅处理前2张图片
            for file in valid_files:
                res = self._process_file(file)
                if res:
                    results[file.stem] = res
            return results
        else:
            # 多线程执行
            max_workers = min(os.cpu_count() or 16, len(valid_files))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self._process_file, f): f for f in valid_files}
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    res = future.result()
                    if res:
                        results[file.stem] = res
        return results
