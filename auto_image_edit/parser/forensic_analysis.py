from typing import Any, Dict

from ..models import DetectionRes
from .base_prompt import BaseParser


class ForensicAnalysisDescription(BaseParser):
    def __init__(self, llm, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        super().__init__(llm, class_name, DetectionRes, store=store)
