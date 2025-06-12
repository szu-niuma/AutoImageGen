import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from tqdm import tqdm

from .processor import ForensicReasoning
from .utils.image_processor import ImageProcessor
from .utils.image_similarity import ImageSimilarity
from .utils.util import load_json, save_json


class AnalysisPipeline:
    """
    分析流程：加载图片、计算差异掩码、调用 ForensicReasoning，支持单/多线程与 JSON 批量处理。
    """

    DEFAULT_WORKERS = 16

    def __init__(
        self,
        config: Dict[str, Any],
        out_dir: str = "output",
        is_debug: bool = False,
        max_num: int = None,
    ):
        """
        :param config: ForensicReasoning 配置
        :param out_dir: 输出根目录
        :param is_debug: 是否调试模式（单线程）
        :param max_num: 最多处理数量，None 表示不限制
        """
        self.image_processor = ImageProcessor(224, 224, maintain_aspect_ratio=False)
        self.forensic_reasoning = ForensicReasoning(config)
        self.is_debug = is_debug
        self.max_num = max_num if max_num is not None else None

        self.output_dir: Path = Path(out_dir) / "AnalysisPipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pipeline(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单张图片：加载原图与编辑图、计算差异掩码、调用 ForensicReasoning
        """
        # 加载并预处理
        _, real_img, _ = self.image_processor.load_image(info["real_image"])
        _, fake_img, _ = self.image_processor.load_image(info["fake_image"])
        fake_b64 = self.image_processor.get_base64(fake_img)

        # 计算像素差异掩码
        mask = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, gray=True)
        mask_b64 = self.image_processor.get_base64(mask)

        # 拼接文本描述
        edited_msg = (
            f"Edited image description: {info.get('descriptive_modification')}, "
            f"Modification instruction: {info.get('brief_modification_instruction')}, "
            f"Modification goal: {info.get('modification_goal')}"
        )

        # 调用检测模型
        return self.forensic_reasoning.run(
            image_path=info["fake_image"],
            edited_image_base64=fake_b64,
            mask_image_base64=mask_b64,
            text_content=edited_msg,
        )

    def get_valid_files(self, info_path: Path) -> List[Dict[str, Any]]:
        """
        支持目录或 JSON 文件：
          - 目录：遍历所有文件
          - JSON：加载待处理列表
        过滤已处理记录，并按 max_num 截断。
        """
        if not info_path.exists():
            raise FileNotFoundError(f"Path not found: {info_path}")

        if info_path.suffix.lower() == ".json":
            data = load_json(info_path)
        elif info_path.is_dir():
            data = [{"real_image": None, "fake_image": p} for p in info_path.iterdir() if p.is_file()]
        else:
            raise ValueError("只支持目录或 JSON 文件")

        pending = []
        for item in data:
            if item.get("forensic_analysis"):
                logger.info(f"跳过已处理文件: {item.get('fake_image', item.get('image_path'))}")
                continue
            pending.append(item)

        if self.max_num:
            pending = pending[: self.max_num]
        return pending

    def _execute(self, valid_files: List[Dict[str, Any]]) -> None:
        """
        核心执行逻辑，单/多线程通用
        """
        workers = (
            1
            if self.is_debug
            else min(
                os.cpu_count() or self.DEFAULT_WORKERS,
                len(valid_files),
                self.DEFAULT_WORKERS,
            )
        )
        success = err = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.pipeline, f): f for f in valid_files}
            with tqdm(total=len(valid_files), desc="处理中", unit="文件") as p_bar:
                for future in as_completed(futures):
                    info = futures[future]
                    try:
                        info["forensic_analysis"] = future.result()
                        success += 1
                    except Exception as e:
                        err += 1
                        logger.error(f"处理失败: {e}", exc_info=True)
                    p_bar.update(1)
                    p_bar.set_postfix({"成功": success, "失败": err})
        logger.info(f"完成: 成功 {success} 失败 {err} 总计 {len(valid_files)}")

    def run(self, info_path: Path) -> Dict[str, Dict[str, Any]]:
        """
        入口：支持单图、目录与 JSON 批量。
        返回 {文件路径: forensic_analysis} 映射，并在 JSON 情况下保存结果。
        """
        info_path = Path(info_path)
        valid_files = self.get_valid_files(info_path)
        if not valid_files:
            logger.info("未发现待处理文件")
            return {}
        else:
            logger.info(f"待处理文件数量: {len(valid_files)}")

        # 执行分析
        self._execute(valid_files)

        # 若输入为 JSON，保存覆盖
        if info_path.suffix.lower() == ".json":
            out_file = self.output_dir / f"{info_path.stem}.json"
            save_json(out_file, valid_files)

        # 返回映射
        return {str(item.get("fake_image", item.get("image_path"))): item["forensic_analysis"] for item in valid_files}
