# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage
from typing import Optional

from ..parser import ForensicReasoningPrompt
from ..models.forensics_desc_model import ForensicsDescModel
from .base_processor import BaseImageProcessor


class ForensicReasoning(BaseImageProcessor):
    # 支持的图片后缀
    USER_PROMPT = "Please analyze the forgery traces in this image"
    PROCESS_NAME = "Forgery Content Analyzer"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESS_NAME)
        self.store = store if store is not None else {}
        self.forensic_reasoning = ForensicReasoningPrompt(self.llm, store=self.store)

    def load_human_msg(
        self,
        edited_image_base64,
        mask_image_base64,
        text_content: Optional[str] = None,
        *args,
        **kwargs,
    ) -> HumanMessage:
        """加载图像数据并构建消息"""
        content = [
            {
                "type": "text",
                "text": "The following image is a edited image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{edited_image_base64}",
                },
            },
            {
                "type": "text",
                "text": "The following image is a grey-scale image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{mask_image_base64}",
                },
            },
        ]
        if text_content:
            content.append(
                {
                    "type": "text",
                    "text": text_content,
                }
            )
        return HumanMessage(role="user", content=content)

    def process_file(self, image_path: Path, image_base64: None, *args, **kwargs) -> Any:
        """执行图像分析"""
        image_path = Path(image_path)
        human_prompt = self.load_human_msg(
            edited_image_base64=kwargs["edited_image_base64"],
            mask_image_base64=kwargs["mask_image_base64"],
            text_content=kwargs["text_content"],
        )
        analysis_result: ForensicsDescModel = self.forensic_reasoning.run(human_prompt, image_path.name)
        return analysis_result.description

    def post_hook(self, image_path, result, *args, **kwargs):
        return super().post_hook(image_path, result, *args, **kwargs)
