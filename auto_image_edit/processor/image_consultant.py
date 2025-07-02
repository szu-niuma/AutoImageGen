# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from ..models import ResponseMethod
from ..parser import ImageConsultantPrompt
from .base_processor import BaseImageProcessor


class ImageConsultant(BaseImageProcessor):
    USER_PROMPT = "Based on the image content, provide editing ideas. Please respond in English. Content as follows:\n{OBJECT}"
    PROCESS_NAME = "Image Editing Creative Generator"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESS_NAME)
        self.store = store if store is not None else {}
        self.image_consultant = ImageConsultantPrompt(self.llm, store=self.store)

    def process_file(self, img_path: Path, image_base64: str = None, *args, **kwargs) -> Optional[dict]:
        """单文件处理逻辑，供多线程调用"""
        img_path = Path(img_path)
        image_analysis = kwargs.get("image_analysis", "图像分析结果")
        user_prompt = self.USER_PROMPT.format(OBJECT=image_analysis)
        human_msg = self.load_human_msg(img_path, user_prompt, image_base64)
        img_analyst: ResponseMethod = self.image_consultant.run(human_msg, img_path.name).model_dump()
        return img_analyst["editorial_inspiration"]
