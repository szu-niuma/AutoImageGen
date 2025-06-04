# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from ..parser import ImageComparisonPrompt
from .base_processor import BaseImageProcessor


class ImageComparison(BaseImageProcessor):
    USER_PROMPT = "Please analyze the forensic differences in the following images."
    PROCESS_NAME = "图像对比处理器"

    def __init__(self, config: dict, store=None):
        super().__init__(config, self.PROCESS_NAME)
        self.store = store if store is not None else {}
        self.image_comparison = ImageComparisonPrompt(self.llm, store=self.store)

    def load_human_msg(
        self,
        src_image_base64,
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
                "text": "The following image is a real image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{src_image_base64}",
                },
            },
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
                "text": "The following image is a gt mask.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/webp;base64,{mask_image_base64}",
                },
            },
        ]
        if text_content:
            content.extend(
                {
                    "type": "text",
                    "text": text_content,
                }
            )
        return HumanMessage(role="user", content=content)

    def process_file(
        self,
        img_path,
        image_base64: str,
        edited_image_base64,
        gt_mask_base64,
        *args,
        **kwargs,
    ) -> Optional[dict]:
        """单文件处理逻辑，供多线程调用"""
        img_path = Path(img_path)
        human_msg = self.load_human_msg(image_base64, edited_image_base64, gt_mask_base64)
        return self.image_comparison.run(human_msg, img_path.name).model_dump()

    def post_hook(self, image_path: Path, result: dict, *args, **kwargs) -> None:
        """后处理钩子方法，可选实现"""
        # logger.info(f"任务执行完毕 [{self.processor_name}]: [{image_path}]")
        result["difficulty_level"] = result["difficulty_level"].value
        return result
