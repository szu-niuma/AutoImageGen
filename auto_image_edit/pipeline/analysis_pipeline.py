from pathlib import Path
from typing import Any, Dict

from ..processor import ForensicReasoning
from ..utils.image_similarity import ImageSimilarity
from .base_pipeline import BasePipeline


class AnalysisPipeline(BasePipeline):
    """
    重构后的AnalysisPipeline，基于BasePipeline
    """

    def __init__(self, config: dict, out_dir="output", is_debug: bool = False, max_num: int = None):
        super().__init__(
            config=config,
            out_dir=out_dir,
            is_debug=is_debug,
            max_num=max_num,
            pipeline_name="AnalysisPipeline",
            image_size=(224, 224),
            maintain_aspect_ratio=False,
        )
        # 初始化法医推理处理器
        self.forensic_reasoning = ForensicReasoning(config)

    def pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心处理逻辑：法医分析
        """
        # 加载并预处理图像
        _, real_img, _ = self.image_processor.load_image(info["real_image"])
        _, fake_img, _ = self.image_processor.load_image(info["fake_image"])
        fake_b64 = self.image_processor.get_base64(fake_img)

        # 计算像素差异掩码
        mask = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, gray=True)
        mask_b64 = self.image_processor.get_base64(mask)

        # 拼接文本描述
        edited_msg = (
            f"Edited image description: {info.get('descriptive_modification')}, "
            f"Modification instruction: {info.get('brief_modification_instruction')}, "
            f"Modification goal: {info.get('modification_goal')}"
        )

        # 调用检测模型
        forensic_result = self.forensic_reasoning.run(
            image_path=info["fake_image"],
            edited_image_base64=fake_b64,
            mask_image_base64=mask_b64,
            text_content=edited_msg,
        )

        # 将结果添加到原info中
        info["forensic_analysis"] = forensic_result
        return info

    def _should_process_file(self, file_path: Path) -> bool:
        """
        重写文件过滤逻辑，支持JSON文件
        """
        return file_path.suffix.lower() in self.IMAGE_EXTS or file_path.suffix.lower() == ".json"
