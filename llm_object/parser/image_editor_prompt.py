from typing import Any, Dict

from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

from .base_prompt import BaseParser


class ImageEditorPrompt(BaseParser):

    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        super().__init__(llm, class_name, store=store)

    def system_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {
            "format_instructions": output_parser.get_format_instructions() if output_parser is not None else None
        }
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template
