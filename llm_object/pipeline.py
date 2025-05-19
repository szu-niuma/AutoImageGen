"""
生成图像信息; 掩码区域内物体信息; 编辑指导信息
"""

from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

import llm_object.utils as utils

from .parser import ImageAnalystPrompt, ImageEditorPrompt, ImageStylistPrompt
from .processor.image_processor import ImageProcessor


class Pipeline:
    def __init__(self, config: dict, img_dir: Path, out_dir="output"):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**config)
        self.store = {}

        self.image_analyst = ImageAnalystPrompt(self.llm, self.store)
        self.image_stylist = ImageStylistPrompt(self.llm, self.store)
        self.image_editor = ImageEditorPrompt(self.llm, self.store)

        self.img_dir: Path = Path(img_dir)
        self.output_dir: Path = Path(out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_image_data(self, image_path: Path) -> HumanMessage:
        """加载图像及其信息"""
        src_img, trans_img, scale_factor = self.image_processor.load_image(image_path)
        # 构建图像信息消息
        image_info = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_base64(src_img)}"},
                }
            ]
        )
        return image_info

    def run(self, image_path: Path) -> dict:
        image_path = self.img_dir / image_path
        if not image_path.exists():
            logger.error("图像文件未找到: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        logger.info(f"开始处理图片: {image_path}")

        image_data = self.load_image_data(image_path)
        # 图像分析
        image_analyst = self.image_analyst.run(image_data, image_path.name)
        # 提供创意
        image_stylist = self.image_stylist.run([], session_id=image_path.name)
        # 编辑结果
        history = self.image_stylist.get_session_history(image_path.name)
        image_stylist = history.messages[-1].content
