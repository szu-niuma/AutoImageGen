# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ..utils.image_processor import ImageProcessor


class BaseImageProcessor(ABC):
    """图像处理基类，提取通用功能"""

    def __init__(self, config: dict, processor_name: str):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**config)
        self.processor_name = processor_name

    def load_human_msg(
        self,
        image_path: Path,
        text_content: str = None,
        image_base64: Optional[str] = None,
    ) -> HumanMessage:
        """加载图像数据并构建消息"""
        if image_base64 is None:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)

        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{image_base64}",
                },
            }
        ]

        if text_content:
            content.append(
                {
                    "type": "text",
                    "text": text_content,
                }
            )

        return HumanMessage(role="user", content=content)

    @abstractmethod
    def process_file(self, *args, **kwargs) -> Optional[dict]:
        """处理单个文件的抽象方法，子类必须实现"""
        pass

    def pre_hook(self, image_path: Path, *args, **kwargs) -> None:
        """预处理钩子方法，可选实现"""
        # logger.info(f"执行任务 [{self.processor_name}]: [{image_path}]")
        pass

    def post_hook(self, image_path: Path, result: dict, *args, **kwargs) -> None:
        """后处理钩子方法，可选实现"""
        # logger.info(f"任务执行完毕 [{self.processor_name}]: [{image_path}]")
        pass

    def run(self, image_path: Path, image_base64: Optional[str] = None, *args, **kwargs) -> dict:
        """运行处理流程"""
        self.pre_hook(image_path, image_base64)
        result = self.process_file(image_path, image_base64=image_base64, *args, **kwargs)
        self.post_hook(image_path, result)
        return result
