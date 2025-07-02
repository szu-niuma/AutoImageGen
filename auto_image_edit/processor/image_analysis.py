# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any

from ..parser import ImageAnalystPrompt, FaceAnalystPrompt
from .base_processor import BaseImageProcessor


class ImageAnalysis(BaseImageProcessor):
    # 支持的图片后缀
    USER_PROMPT = "Please analyze this image and provide the answer in English"
    PROCESS_NAME = "Image Content Analyzer"

    def __init__(self, config: dict, store=None, image_type="natural"):
        super().__init__(config, self.PROCESS_NAME)
        self.store = store if store is not None else {}
        if image_type == "face":
            self.image_analyst = FaceAnalystPrompt(self.llm, store=self.store)
        else:
            self.image_analyst = ImageAnalystPrompt(self.llm, store=self.store)

    def process_file(self, image_path: Path, image_base64: None, *args, **kwargs) -> Any:
        """执行图像分析"""
        human_prompt = self.load_human_msg(image_path, self.USER_PROMPT, image_base64=image_base64)
        analysis_result = self.image_analyst.run(human_prompt, image_path.name).model_dump()
        return analysis_result["objects"]
