# guidance_description.py
import random
from typing import Any, Dict

from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ..models.editing_method_enum import EditingMethodEnum
from ..models.object_model import ObjectModels
from .base_prompt import BaseParser


class GuidanceDescription(BaseParser):
    def __init__(self, llm: BaseLLM):
        prompt_path = "guidance_desc.txt"
        super().__init__(llm, prompt_path, ObjectModels)

    def system_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "All available editing methods:{available_editing_methods}"),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {
            "format_instructions": output_parser.get_format_instructions() if output_parser is not None else None
        }
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_data: Dict[str, Any]) -> ObjectModels | Dict[str, Any]:
        # Prepare input variables
        available_editing_methods = []
        for value in EditingMethodEnum.values():
            available_editing_methods.append(value.get_editing_method_info())
        random.shuffle(available_editing_methods)

        input_variables = {"image_data": [image_data], "available_editing_methods": str(available_editing_methods)}
        try:
            # Run the chain and return the parsed result
            return self.chain.invoke(input_variables)
        except Exception as e:
            # Add logging or other exception handling mechanisms as needed
            return {"error": e}
