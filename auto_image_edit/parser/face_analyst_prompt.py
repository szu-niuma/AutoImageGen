from typing import Any, Dict

from langchain.llms import BaseLLM
from ..models.object_model import ResponseObject
from .base_prompt import BaseParser


class FaceAnalystPrompt(BaseParser):

    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        super().__init__(llm, class_name, ResponseObject, store=store)
