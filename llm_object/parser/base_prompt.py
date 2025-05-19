import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import markdown
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import AIMessage, HumanMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel


class BaseParser(ABC):
    # 将重试提示模板设为类内变量
    RETRY_PROMPT_TEMPLATE = (
        "An error occurred in the json parsing: '{err_detail}'."
        "Please re-output according to the format instructions. Don't give explanations or wrong answers."
    )

    def __init__(self, llm: ChatOpenAI, prompt_path, pydantic_object=None, store=None):
        # 获取当前py文件的path
        prompt_path = Path(prompt_path)
        # 判断prompt_path有没有文件名结尾, 如果没有就加上
        if prompt_path.exists():
            self.prompt_path = prompt_path
        else:
            if not prompt_path.suffix:
                prompt_path = prompt_path.with_suffix(".txt")
            self.prompt_path = Path(__file__).parent.parent / "prompts" / prompt_path
        assert self.prompt_path.exists(), f"Prompt file [{self.prompt_path}] does not exist"

        self.chat_template = None
        self.llm = llm

        # 存储历史记录的地方
        if store is not None:
            self.store = store
        else:
            self.store = {}

        # Load the prompt template from the file
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            # 判断图像的后缀， 如果是以md结尾则读取文本内容为Markdown
            if prompt_path.suffix == ".md":
                system_msg = markdown.markdown(file_content)
            else:
                sections = file_content.split("###")
                system_msg = sections[-1]

        self.system_msg = system_msg
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object) if pydantic_object else None
        self.template = self.load_template(self.parser)
        self.chain = self.template | self.llm
        self.with_message_history = RunnableWithMessageHistory(self.chain, self.get_session_history)

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
        # return ChatPromptTemplate(placeholders)
        partial_vars = {
            "format_instructions": output_parser.get_format_instructions() if output_parser is not None else None
        }
        return ChatPromptTemplate(placeholders, partial_variables=partial_vars)

    def _add_retry_feedback(self, e: Exception, session_id):
        """提取错误详情并追加重试提示到 messages"""
        if hasattr(e, "errors"):
            err_detail = json.dumps(e.errors(), ensure_ascii=False, indent=2)
        else:
            err_detail = str(e)
        retry_prompt = self.RETRY_PROMPT_TEMPLATE.format(err_detail=err_detail)
        return HumanMessage(content=retry_prompt)

    def run(self, inputs: HumanMessage, session_id, max_retries: int = 5, *args, **kwargs) -> BaseModel:
        # 将用户输入注入到 human 占位符
        config = {"configurable": {"session_id": f"{session_id}"}, "request_timeout": 30}
        messages = inputs if isinstance(inputs, list) else [inputs]

        for attempt in range(1, max_retries + 1):
            resp: AIMessage = self.with_message_history.invoke({"messages": messages}, config=config)
            resp.content = resp.content.replace("“", '"').replace("”", '"')
            if self.parser is None:
                return resp.content
            # 如果需要解析, 则返回解析的内容
            try:
                # 尝试能否解析为json进行解析
                info = resp.content.replace("```json\n", "").replace("```", "").strip().replace("\n", "")
                json.loads(info)
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    messages = self._add_retry_feedback(e, session_id)
                else:
                    raise RuntimeError(f"经过 {max_retries} 次重试仍无法解析: {e}")
            else:
                # 尝试能否解析为内容
                try:
                    target = self.parser.parse(resp.content)
                    # 删除重试相关的提示信息
                    if attempt > 1:
                        self.delete_message(session_id, -(2 * (attempt - 1)) - 1, -2)
                    break
                except Exception as e:
                    if attempt < max_retries:
                        messages = self._add_retry_feedback(e, session_id)
                    else:
                        raise RuntimeError(f"经过 {max_retries} 次重试仍无法解析: {e}")
        return target

    def run_stream(self, inputs: HumanMessage, max_retries: int = 3):
        """
        支持可停止的流式传输：每次 yield 新的文本片段；所有尝试结束后 parse，
        若 parse 失败则重试（追加错误反馈 prompt 再次流式），否则返回解析结果并结束。
        """
        # 构造初始 prompt 和消息列表
        prompt = self.template.format_prompt(input=inputs.content)
        messages = prompt.to_messages()
        if not getattr(self.llm, "streaming", False):
            raise ValueError("请在初始化时设置 ChatOpenAI(streaming=True) 以支持流式传输")
        last_err = None

        for attempt in range(1, max_retries + 1):
            complete_content = ""
            for chunk in self.llm.stream(messages):
                delta = chunk.generations[0].message.content
                complete_content += delta
                yield delta

            messages.append(AIMessage(content=complete_content))
            try:
                parsed = self.parser.parse(complete_content)
                yield parsed
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    self._add_retry_feedback(messages, e)
                else:
                    raise RuntimeError(f"经过 {max_retries} 次流式重试仍无法解析") from last_err

    def delete_message(self, session_id: str, start_index: int, end_index: int = None):
        """
        删除指定历史消息。
        :param session_id: 会话 ID
        :param start_index: 要删除的消息在历史列表中的起始索引，可以为负数
        :param end_index: 要删除的消息在历史列表中的结束索引（包含），可以为负数。不传则只删除 start_index。
        """
        # 1. 获取历史记录
        history = self.get_session_history(session_id)
        msgs = history.messages  # List[BaseMessage]
        length = len(msgs)

        # 2. 归一化索引函数
        def norm(i: int) -> int:
            return i if i >= 0 else length + i

        # 3. 执行删除
        if end_index is None:
            idx = norm(start_index)
            if not (0 <= idx < length):
                raise IndexError(f"消息索引 {start_index} 超出范围")
            del msgs[idx]
        else:
            s = norm(start_index)
            e = norm(end_index)
            if not (0 <= s < length and 0 <= e < length):
                raise IndexError(f"删除范围 [{start_index}, {end_index}] 超出历史消息长度 {length}")
            # 正向或反向区间均可
            if s <= e:
                del msgs[s : e + 1]
            else:
                del msgs[e : s + 1]

    def modify_message(
        self,
        session_id: str,
        index: int,
        new_content: str,
    ):
        """
        修改指定历史消息并重新生成结果。
        :param session_id: 会话 ID
        :param index: 要修改的消息在历史列表中的索引
        :param new_content: 新的消息内容
        :return: 解析后的对象
        """
        # 1. 取历史
        history = self.get_session_history(session_id)
        msgs = history.messages  # List[BaseMessage]

        # 2. 修改指定条目
        if not (0 <= index < len(msgs)):
            raise IndexError(f"消息索引 {index} 超出范围")
        orig = msgs[index]
        msgs[index] = (
            HumanMessage(content=new_content) if isinstance(orig, HumanMessage) else AIMessage(content=new_content)
        )

        # 3. 重新调用 LLM（带历史）
        config = {"configurable": {"session_id": session_id}}
        ai_msg = self.with_message_history.invoke({"messages": msgs}, config=config)

        # 4. 解析并返回
        return self.parser.parse(ai_msg.content)
