import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

import numpy as np
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm import tqdm

from ..parser import ForensicAnalysisDescription
from ..utils.image_processor import ImageProcessor

# 假设这些导入是必要的，已保持项目结构
from .base_processor import BaseImageProcessor


class AuthorityAssess(BaseImageProcessor):
    """图像真实性评估器"""

    USER_PROMPT = "分析下述图片的真实性，判断是否为真实图片，并给出中文的分析结果"
    PROCESSOR_NAME = "真实性评估"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESSOR_NAME)
        self.store = store if store is not None else {}
        self.processor = ForensicAnalysisDescription(self.llm, store=self.store)

    def process_file(self, image_path: Path, image_base64: None) -> Any:
        """执行图像分析"""
        human_prompt = self.load_human_msg(image_path, self.USER_PROMPT, image_base64=image_base64)
        analysis_result = self.processor.run(human_prompt, image_path.name).model_dump()
        return analysis_result
