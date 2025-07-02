import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import random
from loguru import logger
from tqdm import tqdm
from ..utils.image_processor import ImageProcessor
from ..utils.util import load_json, save_json


class BasePipeline(ABC):
    """
    Pipeline抽象基类，提供通用的多线程处理框架和文件管理功能
    """

    DEFAULT_WORKERS = 16
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    def __init__(
        self,
        config: Dict[str, Any],
        out_dir: str = "output",
        is_debug: bool = False,
        max_num: Optional[int] = None,
        pipeline_name: Optional[str] = None,
        image_size: Optional[tuple] = None,
        maintain_aspect_ratio: bool = True,
    ):
        """
        初始化基类

        Args:
            config: 配置字典
            out_dir: 输出根目录
            is_debug: 是否调试模式（单线程）
            max_num: 最大处理数量，None表示不限制
            pipeline_name: pipeline名称，用于输出目录命名
            image_size: 图像处理尺寸 (width, height)
            maintain_aspect_ratio: 是否保持宽高比
        """
        self.config = config
        self.is_debug = is_debug
        self.max_num = max_num if max_num is not None else float("inf")

        # 初始化图像处理器
        if image_size:
            self.image_processor = ImageProcessor(image_size[0], image_size[1], maintain_aspect_ratio=maintain_aspect_ratio)
        else:
            self.image_processor = ImageProcessor()

        # 设置输出目录
        pipeline_name = pipeline_name or self.__class__.__name__
        self.output_dir: Path = Path(out_dir) / pipeline_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        核心处理逻辑，子类必须实现

        Args:
            info: 包含处理信息的字典

        Returns:
            处理结果字典
        """
        pass

    def get_valid_files(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        获取待处理文件列表，支持目录、JSON文件和单个图像文件

        Args:
            input_path: 输入路径

        Returns:
            待处理文件信息列表
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")

        valid_files = []

        if input_path.suffix.lower() == ".json":
            # JSON文件：直接加载数据
            data = load_json(input_path)
            valid_files = data if isinstance(data, list) else [data]
        elif input_path.is_dir():
            # 目录：遍历所有文件
            for file_path in input_path.iterdir():
                if file_path.is_file():
                    # 根据子类需求决定是否过滤图像文件
                    if self._should_process_file(file_path):
                        valid_files.append({"image_path": str(file_path)})
                        # 达到最大数量时停止遍历
                        if self.max_num != float("inf") and len(valid_files) >= self.max_num:
                            break
        elif input_path.is_file():
            # 单个文件
            if self._should_process_file(input_path):
                valid_files = [{"image_path": str(input_path)}]
        else:
            raise ValueError(f"Unsupported path type: {input_path}")

        # 应用max_num限制
        if self.max_num != float("inf"):
            # 要求随机打乱
            random.shuffle(valid_files)
            valid_files = valid_files[: int(self.max_num)]

        return valid_files

    def _should_process_file(self, file_path: Path) -> bool:
        """
        判断是否应该处理该文件，子类可以重写此方法

        Args:
            file_path: 文件路径

        Returns:
            是否应该处理
        """
        # 默认只处理图像文件，子类可以重写
        return file_path.suffix.lower() in self.IMAGE_EXTS

    def _check_existing_result(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        检查是否已有处理结果，子类可以重写此方法

        Args:
            info: 输入信息

        Returns:
            已存在的结果，如果不存在则返回None
        """
        # 根据image_path生成保存路径
        image_path = info.get("image_path")
        if image_path is None:
            image_path = info.get("img_path")  # 兼容旧格式

        if image_path:
            image_path = Path(image_path)
            save_path = self.output_dir / f"{image_path.stem}.json"
            result = load_json(save_path, raise_exception=False)
            if result:
                logger.info(f"已存在结果文件: {save_path}")
                return result
        return None

    def _save_result(self, info: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        保存处理结果，子类可以重写此方法

        Args:
            info: 输入信息
            result: 处理结果
        """
        image_path = info.get("image_path")
        if image_path:
            image_path = Path(image_path)
            save_path = self.output_dir / f"{image_path.stem}.json"
            save_json(save_path, result)
            logger.info(f"处理完成: {image_path} -> {save_path}")

    def _execute_pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单个文件的完整处理流程（包括缓存检查和结果保存）

        Args:
            info: 输入信息

        Returns:
            处理结果
        """
        # 检查是否已有结果
        existing_result = self._check_existing_result(info)
        if existing_result:
            return existing_result

        # 执行核心处理逻辑
        result = self.pipeline(info)

        # 保存结果
        self._save_result(info, result)

        return result

    def _execute(self, valid_files: List[Dict[str, Any]]) -> None:
        """
        执行批量处理，支持单/多线程

        Args:
            valid_files: 待处理文件列表
        """
        if not valid_files:
            logger.warning("没有待处理文件")
            return

        workers = (
            1
            if self.is_debug
            else min(
                os.cpu_count() or self.DEFAULT_WORKERS,
                len(valid_files),
                self.DEFAULT_WORKERS,
            )
        )

        success_count = error_count = 0

        if workers == 1:
            # 单线程处理
            for info in tqdm(valid_files, desc="处理中", disable=len(valid_files) == 1):
                self._execute_pipeline(info)
                success_count += 1
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {executor.submit(self._execute_pipeline, file_info): file_info for file_info in valid_files}

                with tqdm(total=len(valid_files), desc="处理中", unit="文件") as pbar:
                    for future in as_completed(future_to_file):
                        file_info = future_to_file[future]
                        try:
                            future.result()
                            success_count += 1
                        except Exception as e:
                            error_count += 1
                            logger.error(f"处理文件失败 {file_info}: {str(e)}", exc_info=True)
                        finally:
                            pbar.update(1)
                            pbar.set_postfix({"成功": success_count, "失败": error_count})

        logger.info(f"处理完成 - 成功: {success_count}, 失败: {error_count}, 总计: {len(valid_files)}")

    def run(self, input_path: Union[str, Path]) -> Dict[str, Any]:
        """
        主入口方法：处理单张图片、目录或JSON文件

        Args:
            input_path: 输入路径（图片、目录或JSON文件）

        Returns:
            处理结果字典
        """
        input_path = Path(input_path)
        logger.info(f"开始处理: {input_path}")

        # 获取待处理文件列表
        valid_files = self.get_valid_files(input_path)
        logger.info(f"待处理文件数量: {len(valid_files)}")

        if not valid_files:
            logger.warning("没有找到待处理文件")
            return {}

        # 执行处理
        self._execute(valid_files)

        # 如果输入是JSON文件，保存更新后的结果
        if input_path.suffix.lower() == ".json":
            output_path = self.output_dir / f"{input_path.stem}_processed.json"
            save_json(output_path, valid_files)
            logger.info(f"批量处理结果已保存: {output_path}")

        # 返回结果映射
        return {str(item.get("image_path", item.get("fake_image", "unknown"))): item for item in valid_files}

    def set_max_workers(self, max_workers: int) -> None:
        """
        设置最大工作线程数

        Args:
            max_workers: 最大工作线程数
        """
        self.DEFAULT_WORKERS = max_workers

    def get_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息

        Returns:
            统计信息字典
        """
        return {
            "pipeline_name": self.__class__.__name__,
            "output_dir": str(self.output_dir),
            "is_debug": self.is_debug,
            "max_num": self.max_num,
            "default_workers": self.DEFAULT_WORKERS,
        }
