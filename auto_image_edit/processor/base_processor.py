import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from ..utils.image_processor import ImageProcessor


class BaseImageProcessor(ABC):
    """图像处理基类，提取通用功能"""

    def __init__(
        self, config: dict, out_dir: str, processor_name: str, is_debug: bool = False, max_num: Optional[int] = None
    ):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**config)
        self.processor_name = processor_name
        self.output_dir: Path = Path(out_dir) / processor_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug
        self.max_num = max_num

    def save_info(self, target_file: Path, img_file: Path, content: Any) -> dict:
        """保存处理结果到文件"""
        logger.info(f"开始处理图片: {img_file}")
        result = {"img_path": str(img_file), self.processor_name: content}

        target_file.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info(f"分析结果已保存到: {target_file}")
        return result

    def load_image_data(self, image_path: Path, text_content: str, image_base64: Optional[str] = None) -> HumanMessage:
        """加载图像数据并构建消息"""
        if image_base64 is None:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)

        return HumanMessage(
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
                    "text": text_content,
                },
            ],
        )

    @abstractmethod
    def _process_file(self, *args, **kwargs) -> Optional[dict]:
        """处理单个文件的抽象方法，子类必须实现"""
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        """运行处理流程的抽象方法，子类必须实现"""
        pass
