# guidance_description.py
from typing import Any, Dict, List, Type

from langchain.llms import BaseLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, create_model

from ..models import ResponseMethod
from ..models.editing_method_enum import EditingMethodEnum
from .base_prompt import BaseParser


class ImageConsultantPrompt(BaseParser):
    def __init__(self, llm: BaseLLM, store: Dict[str, Any] = None):
        class_name = self.__class__.__name__
        self.combine, self.edited_method_info = self.get_all_editing_methods()
        super().__init__(llm, class_name, ResponseMethod, store=store)

    @staticmethod
    def create_combined_model(model_name, models: List[Type[BaseModel]]) -> Type[BaseModel]:
        """
        动态创建一个组合多个 Pydantic BaseModel 的新模型。

        例如: 输入是: [A, B, C]
        输出期望是一个类, 类的结构如下
        class combine(BaseModel):
            A: A
            B: B
            C: C

        :param models: 要组合的 BaseModel 类的列表
        :return: 新的组合 BaseModel 类
        """
        if not models:
            raise ValueError("模型列表不能为空")

        fields = {}
        for model in models:
            field_name = model.__name__
            if field_name in fields:
                raise ValueError(f"字段名冲突: {field_name} 已存在于组合模型中。")
            fields[field_name] = (model, ...)  # 使用模型类作为字段类型，字段为必填

        # 创建新的组合模型类
        CombinedModel = create_model(model_name, **fields)

        return CombinedModel

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", f"Optional editing methods:{self.edited_method_info}"),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("user", "{image_info}"),
        ]
        partial_vars = {
            "format_instructions": output_parser.get_format_instructions() if output_parser is not None else None
        }
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def get_all_editing_methods(self):
        edited_method_info = ""
        combine = []
        for name, detail in EditingMethodEnum.items():
            edited_method_info += f"{detail.get_editing_method_info()} \n"
            combine.append(detail.model)
        combine_model = self.create_combined_model("CombinedModel", combine)
        return combine_model, edited_method_info
