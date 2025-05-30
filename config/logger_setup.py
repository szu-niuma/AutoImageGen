import sys

from loguru import logger


class LoggerSetup:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # 只在第一次初始化时执行配置
        if self._initialized:
            return

        # 移除默认输出
        logger.remove()

        # 添加文件日志
        logger.add(
            "logs/app.log",
            rotation="10 MB",
            retention="10 days",
            compression="zip",
            level="INFO",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )

        # 添加控制台日志
        logger.add(
            sys.stdout,
            level="INFO",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            colorize=True,
        )

        self._initialized = True
