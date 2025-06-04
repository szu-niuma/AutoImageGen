from typing import Any, Dict

from ..models.forensics_accessibility import RepForensicsAccessModel
from .base_prompt import BaseParser


# DifficultyAnalysisDescription类，整合错误处理
class ImageComparisonPrompt(BaseParser):
    """
    明确带有正确答案, 描述编辑区域内外的一些特征差异
    """

    def __init__(self, llm, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        super().__init__(llm, class_name, RepForensicsAccessModel, store=store)
