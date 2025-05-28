# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from ..parser import ImageEditorPrompt
from .base_processor import BaseImageProcessor


class ImageEditor(BaseImageProcessor):
    USER_PROMPT = "目标物体: {object}，编辑要求:{edit_process}"

    def __init__(self, config: dict, out_dir, store=None):
        super().__init__(config, out_dir, "image_edit")
        self.store = store if store is not None else {}
        self.image_edit = ImageEditorPrompt(self.llm, store=self.store)

    def load_human_msg(self, image_path: Path, image_info: dict, image_base64: Optional[str] = None) -> HumanMessage:
        """重写父类方法以适应特定的文本格式"""
        if image_base64 is None:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)

        text_content = self.USER_PROMPT.format(
            object=image_info["edit_target"],
            edit_process=image_info["edit_process"],
        )

        return super().load_human_msg(image_path, text_content, image_base64)

    def _extract_image_url(self, markdown: str) -> str:
        """从 Markdown 格式字符串中提取第一个图片链接"""
        m = re.search(r"!\[.*?\]\((.*?)\)", markdown)
        if not m:
            raise ValueError(f"无法解析图片链接：{markdown}")
        return m.group(1)

    def process_file(self, img_path: Path, image_base64: str, image_creativity: list, **kwargs) -> Optional[str]:
        """单文件处理逻辑，供多线程调用"""
        img_path = Path(img_path)
        for edit_info in image_creativity:
            image_data = self.load_human_msg(img_path, edit_info, image_base64)
            content = self.image_edit.run(image_data, img_path.name)
            # '![图片](https://tokensceshi.oss-ap-southeast-1.aliyuncs.com/sora/cc8052eb-427d-481f-a4d6-585f8080c16d.png)\n\n'
            # 将图片链接下载到本地
            image_url = self._extract_image_url(content)
            if image_url:  # 修复逻辑错误：应该是 if image_url 而不是 if not image_url
                save_path = self.output_dir / img_path.name
                self.image_processor.download_image(image_url, save_path)
                return str(save_path)
            else:
                return content

    def run(self, image_info: dict) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        for name, info in image_info.items():
            edit_img_path = self.process_file(**info)
            info["edit_img_path"] = str(edit_img_path)

        # 保存结果
        target_file = self.output_dir / f"{self.__class__.__name__}.json"
        # 保存为json文件
        with target_file.open("w", encoding="utf-8") as f:
            json.dump(image_info, f, ensure_ascii=False, indent=4)

        return image_info
