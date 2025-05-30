import os
from pathlib import Path

from dotenv import load_dotenv

from .logger_setup import LoggerSetup


class Config:
    _instance = None

    def __new__(cls, env_path: str = ".env"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, env_path: str = ".env"):
        if self._initialized:
            return
        self._initialized = True
        LoggerSetup()
        self.load_config(env_path)

    def load_config(self, env_path: str):
        env_path: Path = Path(__file__).parent / env_path
        assert env_path.exists(), f"环境变量的配置文件不存在: {env_path}"
        if not load_dotenv(dotenv_path=env_path):
            raise EnvironmentError("Failed to load .env file")

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = os.getenv("OPENAI_MODEL_NAME")

        if not self.api_key or not self.base_url:
            raise EnvironmentError("API key or base URL not found in environment variables")

    def get_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": 300,
            "max_retries": 3,
            "temperature": 0,  # 控制输出的随机性和创造性, 0.0(完全确认)-2.0(高度随机和创造性)
            "top_p": 0.1,  # 控制词汇选择范围, 0.0(完全确定)-1.0(考虑所有可能的词汇)
        }
