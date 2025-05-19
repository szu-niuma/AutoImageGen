import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from .models import ResponseMethod
from .parser import ImageStylistPrompt
from .processor.image_processor import ImageProcessor


class ImageCreativity:
    def __init__(self, config: dict, out_dir, is_debug: bool = False, max_num=None):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**config)
        self.image_stylist = ImageStylistPrompt(self.llm)
        self.name = "image_creativity"
        self.output_dir: Path = Path(out_dir) / self.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug
        self.max_num = max_num

    def save_info(self, target_file, img_file: Path, content: dict) -> dict:
        """处理单个文件并保存结果"""
        logger.info(f"开始处理图片: {img_file}")
        result = {"img_path": str(img_file), self.name: content}

        target_file.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info(f"分析结果已保存到: {target_file}")
        return result

    def load_image_data(self, image_path: Path, image_info: str, image_base64=None) -> HumanMessage:
        if image_base64 is not None:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)
        # 构建图像信息消息
        image_info = HumanMessage(
            role="user",
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/webp;base64,{image_base64}",
                    },
                },
                {
                    "type": "text",
                    "text": str(image_info),
                },
            ],
        )
        return image_info

    def _process_file(self, img_path: Path, image_analysis, image_base64) -> dict:
        """单文件处理逻辑，供多线程调用"""
        try:
            img_path = Path(img_path)
            image_data = self.load_image_data(img_path, image_analysis["objects"], image_base64)
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
