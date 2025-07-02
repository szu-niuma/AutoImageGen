# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any

from ..parser import ForensicAnalysisDescription
from .base_processor import BaseImageProcessor


class AuthorityAssess(BaseImageProcessor):
    """图像真实性评估器"""

    USER_PROMPT = "Analyze the authenticity of the image below, determine whether it is a real image, and provide the analysis results in English"
    PROCESSOR_NAME = "Authenticity Assessment"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESSOR_NAME)
        self.store = store if store is not None else {}
        self.processor = ForensicAnalysisDescription(self.llm, store=self.store)

    def process_file(self, image_path: Path, image_base64: None, *args, **kwargs) -> Any:
        """执行图像分析"""
        human_prompt = self.load_human_msg(image_path, self.USER_PROMPT, image_base64=image_base64)
        analysis_result = self.processor.run(human_prompt, image_path.name).model_dump()
        return analysis_result
