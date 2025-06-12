import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from .processor import ImageComparison
from .utils.image_processor import ImageProcessor
from .utils.util import load_json, save_json


class DiffPipeline:
    DEFAULT_WORKERS = 16

    def __init__(self, config: dict, out_dir="output", is_debug: bool = False, max_num: int = None):
        self.image_processor = ImageProcessor(256, 256)
        self.is_debug = is_debug

        # 图像对比
        self.image_consultant = ImageComparison(config)

        self.output_dir: Path = Path(out_dir) / "DiffPipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_num = max_num if max_num is not None else float("inf")

    def pipeline(self, info: dict) -> dict:
        image_path = info.get("image_path", None)
        if image_path:
            image_path = Path(image_path)
            save_path = self.output_dir / f"{image_path.stem}.json"
            ret = load_json(save_path, raise_exception=False)
            if ret:
                logger.info(f"已存在结果文件: {save_path}")
                return ret
            else:
                ret = load_json(image_path, raise_exception=False)

            _, trans_real_img, _ = self.image_processor.load_image(ret["real_image"])
            real_image_base64 = self.image_processor.get_base64(trans_real_img)

            for info in ret["creativity"]:
                _, trans_edited_img, _ = self.image_processor.load_image(info["output_img"])
                edited_image_base64 = self.image_processor.get_base64(trans_edited_img)

                _, trans_gt_mask_img, _ = self.image_processor.load_image(info["gt_mask"])
                gt_mask_base64 = self.image_processor.get_base64(trans_gt_mask_img)

                res_info = self.image_consultant.run(
                    **{
                        "image_path": ret["real_image"],
                        "image_base64": real_image_base64,
                        "edited_image_base64": edited_image_base64,
                        "gt_mask_base64": gt_mask_base64,
                    }
                )
                info["diff"] = res_info
            # 保存处理结果
            save_json(save_path, ret)
            logger.info(f"处理完成: {image_path} -> {save_path}")

        else:
            _, trans_real_img, _ = self.image_processor.load_image(info["real_image"])
            real_image_base64 = self.image_processor.get_base64(trans_real_img)
            _, trans_edited_img, _ = self.image_processor.load_image(info["fake_image"])
            edited_image_base64 = self.image_processor.get_base64(trans_edited_img)
            _, trans_gt_mask_img, _ = self.image_processor.load_image(info["gt_mask"])
            gt_mask_base64 = self.image_processor.get_base64(trans_gt_mask_img)
            res_info = self.image_consultant.run(
                **{
                    "image_path": info["real_image"],
                    "image_base64": real_image_base64,
                    "edited_image_base64": edited_image_base64,
                    "gt_mask_base64": gt_mask_base64,
                }
            )
            info["diff"] = res_info
        return info

    def read_dir(self, image_path: Path) -> Any:
        # 构造待处理路径列表 - 按需遍历，达到目标数量即停止
        valid_files = []
        if image_path.is_dir():
            for file_path in image_path.iterdir():
                if file_path.is_file():
                    valid_files.append({"image_path": file_path})
                    # 达到最大数量时停止遍历
                    if self.max_num and len(valid_files) >= self.max_num:
                        break
        else:
            logger.error(f"提供的路径不是一个目录: {image_path}")
            raise ValueError(f"Provided path is not a directory: {image_path}")
        return valid_files

    def read_json(self, json_path: Path) -> list:
        data = load_json(json_path)
        # 判断 max_num 是否为无穷大
        if self.max_num == float("inf"):
            return data
        else:
            return data[: self.max_num]

    def run(self, image_path: Path) -> dict:
        """
        处理单张图片或目录下所有图片，返回 {文件名: 分析结果} 的字典 (多线程)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"图像路径不存在: {image_path}")
            raise FileNotFoundError(f"Image path not found: {image_path}")

        # 判断是否是json文件
        if image_path.suffix.lower() == ".json":
            valid_files = self.read_json(image_path)
        elif image_path.is_dir():
            # 如果是目录，读取目录下的所有图片
            valid_files = self.read_dir(image_path)
        else:
            raise ValueError(f"不支持的文件类型: {image_path.suffix}. 仅支持目录或json文件.")

        if self.is_debug:
            # 单线程处理
            for info in tqdm(valid_files, desc="处理中", disable=len(valid_files) == 1):
                try:
                    self.pipeline(info)
                except Exception as e:
                    logger.error(f"处理文件失败 {info}: {str(e)}", exc_info=True)
                    # 可以选择是否继续处理其他文件
                    continue
        else:
            # 多线程处理
            max_workers = min(os.cpu_count() or self.DEFAULT_WORKERS, len(valid_files), self.DEFAULT_WORKERS)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.pipeline, file_info): file_info for file_info in valid_files}
                # 统计处理结果
                success_count, error_count = 0, 0
                with tqdm(total=len(valid_files), desc="处理中", unit="文件") as p_bar:
                    for future in as_completed(future_to_file):
                        future_to_file[future]
                        try:
                            future.result()
                            success_count += 1
                        except Exception as e:
                            error_count += 1
                            logger.error(f"处理失败 {info}: {str(e)}", exc_info=True)
                        finally:
                            p_bar.update(1)
                            p_bar.set_postfix({"成功": success_count, "失败": error_count})
                logger.info(f"处理完成 - 成功: {success_count}, 失败: {error_count}, 总计: {len(valid_files)}")

        # 如果是单个的json文件, 则覆盖保存内容
        image_path = Path(image_path)
        output_path = self.output_dir / f"{image_path.parent.name}.json"
        if image_path.suffix.lower() == ".json":
            # valid_files按照key值进行排序
            save_json(output_path, valid_files)
