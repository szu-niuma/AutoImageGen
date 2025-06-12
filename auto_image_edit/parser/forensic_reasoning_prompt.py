from typing import Any, Dict

from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from ..models.forensics_desc_model import ForensicsDescModel
from .base_prompt import BaseParser


class ForensicReasoningPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        super().__init__(llm, class_name, ForensicsDescModel, store=store)
