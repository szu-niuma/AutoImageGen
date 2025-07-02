from pathlib import Path
from typing import Any, Dict

from ..processor import AuthorityAssess, ImageAnalysis, ImageConsultant
from .base_pipeline import BasePipeline


class CreativityPipeline(BasePipeline):
    """
    重构后的CreativityPipeline，基于BasePipeline
    """

    def __init__(
        self,
        config: dict,
        out_dir="output",
        is_debug: bool = False,
        max_num: int = None,
        image_type: str = "natural",
    ):
        super().__init__(config=config, out_dir=out_dir, is_debug=is_debug, max_num=max_num, pipeline_name="CreativityPipeline")
        # 初始化各种处理器
        self.authority_assessor = AuthorityAssess(config)
        self.image_analyst = ImageAnalysis(config, image_type=image_type)
        self.image_consultant = ImageConsultant(config)

    def pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心处理逻辑：图像创意生成
        """
        if image_path := info.get("image_path") is None:
            # 如果没有image_path，尝试使用img_path兼容旧格式
            image_path = info.get("img_path")
        image_path = Path(image_path)

        # 加载和处理图像
        src_img, _, _ = self.image_processor.load_image(image_path)
        image_base64 = self.image_processor.get_base64(src_img)

        # 获取或生成图像分析
        annotations = info.get("annotations", None)
        if annotations:
            annotations = "\n".join([f"category: {ann['category_name']}" for ann in annotations])
        else:
            annotations = self.image_analyst.run(image_path, image_base64=image_base64)

        # 生成创意和权威性评估
        creativity_info = self.image_consultant.run(
            image_path,
            image_analysis=annotations,
            image_base64=image_base64,
        )
        authorities_info = self.authority_assessor.run(image_path, image_base64=image_base64)

        return {
            "authorities": authorities_info,
            "analysis": info.get("annotations", None) or annotations,
            "creativity": creativity_info,
        }
