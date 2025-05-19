import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from .models import ResponseMethod
from .parser import ImageEditorPrompt
from .processor.image_processor import ImageProcessor


class ImageEditor:
    USER_PROMPT = "目标物体: {object}，编辑要求:{edit_process}"

    def __init__(self, config: dict, out_dir, is_debug: bool = False, max_num=None):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**config)
        self.image_edit = ImageEditorPrompt(self.llm)
        self.output_dir: Path = Path(out_dir) / "image_edit"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_debug = is_debug
        self.max_num = max_num

    def load_image_data(self, image_path: Path, image_info: str, image_base64=None) -> HumanMessage:
        if image_base64 is not None:
            src_img, _, _ = self.image_processor.load_image(image_path)
            image_base64 = self.image_processor.get_base64(src_img)
        # 构建图像信息消息
        image_info = HumanMessage(
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
                    "text": self.USER_PROMPT.format(
                        object=image_info["edit_target"],
                        edit_process=image_info["edit_process"],
                    ),
                },
            ],
        )
        return image_info

    def save_info(self, target_file: Path, img_file: Path, img_analyst: dict) -> dict:
        """处理单个文件并保存结果"""
        logger.info(f"开始处理图片: {img_file}")
        result = {"img_path": str(img_file), "image_analysis": img_analyst}
        target_file.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info(f"分析结果已保存到: {target_file}")
        return result

    def _extract_image_url(self, markdown: str) -> str:
        """从 Markdown 格式字符串中提取第一个图片链接"""
        m = re.search(r"!\[.*?\]\((.*?)\)", markdown)
        if not m:
            raise ValueError(f"无法解析图片链接：{markdown}")
        return m.group(1)

    def _process_file(self, img_path: Path, image_base64, image_creativity, **kwargs) -> dict:
        """单文件处理逻辑，供多线程调用"""
        img_path = Path(img_path)
        for edit_info in image_creativity:
            image_data = self.load_image_data(img_path, edit_info, image_base64)
            content = self.image_edit.run(image_data, img_path.name)
            # '![图片](https://tokensceshi.oss-ap-southeast-1.aliyuncs.com/sora/cc8052eb-427d-481f-a4d6-585f8080c16d.png)\n\n'
            # 将图片链接下载到本地
            image_url = self._extract_image_url(content)
            if not image_url:
                save_path = self.output_dir / img_path.name
                self.image_processor.download_image(image_url, save_path)
                return save_path
            else:
                return content

    def run(self, image_info: dict) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        for name, info in image_info.items():
            edit_img_path = self._process_file(**info)
            info["edit_img_path"] = str(edit_img_path)
        # 保存结果
        target_file = self.output_dir / f"{self.__class__.__name__}.json"
        # 保存为json文件
        with target_file.open("w", encoding="utf-8") as f:
            json.dump(image_info, f, ensure_ascii=False, indent=4)
