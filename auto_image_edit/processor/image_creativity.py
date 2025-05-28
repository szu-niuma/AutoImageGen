import json
from pathlib import Path
from typing import Optional

from loguru import logger

from ..models import ResponseMethod
from ..parser import ImageStylistPrompt
from .base_processor import BaseImageProcessor


class ImageCreativity(BaseImageProcessor):
    def __init__(self, config: dict, out_dir, is_debug: bool = False, max_num=None):
        super().__init__(config, out_dir, "image_creativity", is_debug, max_num)
        self.image_stylist = ImageStylistPrompt(self.llm)

    def _process_file(self, img_path: Path, image_analysis: dict, image_base64: str) -> Optional[dict]:
        """单文件处理逻辑，供多线程调用"""
        try:
            img_path = Path(img_path)
            image_data = self.load_image_data(img_path, str(image_analysis["objects"]), image_base64)
            img_analyst: ResponseMethod = self.image_stylist.run(image_data, img_path.name).model_dump()
            res = {obj["label"]: [obj["caption"]] for obj in image_analysis["objects"]}

            # 根据label拼接起来
            ret = []
            for info in img_analyst["objects"]:
                label = info["edit_target"]
                if res.get(label) is None:
                    logger.warning(f"未找到标签 {label} 的分析结果")
                    continue
                info["target_caption"] = res[label][0]
                ret.append(info)
            return ret
        except Exception as e:
            logger.error(f"处理图片 {img_path} 时发生错误: {e}")
            return None

    def run(self, image_info: dict) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        res = {}
        for name, info in image_info.items():
            # 判断本地是否存在该文件, 如果存在则跳过
            target_file = self.output_dir / f"{name}.json"
            if target_file.exists():
                ret = json.loads(target_file.read_text(encoding="utf-8"))
            else:
                ret = self._process_file(**info)
                if ret is None:
                    continue
                self.save_info(target_file, info["img_path"], ret)
            ret.update(info)
            res[name] = ret
        return res
