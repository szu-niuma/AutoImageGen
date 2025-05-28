import json
from pathlib import Path
from typing import Optional

from loguru import logger

from ..models import ResponseMethod
from ..parser import ImageConsultantPrompt
from .base_processor import BaseImageProcessor


class ImageConsultant(BaseImageProcessor):
    USER_PROMPT = "请根据图片内容，给出编辑想法, 用中文回复, 内容如下:\n{OBJECT}"
    PROCESS_NAME = "图像编辑创意生成器"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESS_NAME)
        self.store = store if store is not None else {}
        self.image_consultant = ImageConsultantPrompt(self.llm, store=self.store)

    def process_file(self, img_path: Path, image_analysis: dict = None, image_base64: str = None) -> Optional[dict]:
        """单文件处理逻辑，供多线程调用"""
        try:
            img_path = Path(img_path)
            user_prompt = self.USER_PROMPT.format(OBJECT=image_analysis)
            human_msg = self.load_human_msg(img_path, user_prompt, image_base64)
            img_analyst: ResponseMethod = self.image_consultant.run(human_msg, img_path.name).model_dump()
            return img_analyst["editorial_inspiration"]
        except Exception as e:
            logger.error(f"处理图片 {img_path} 时发生错误: {e}")
            return None
