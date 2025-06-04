import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_image_edit.utils.image_processor import ImageProcessor

SAVE_TARGET_DIR = Path("/data1/yuyangxin/datasets/anyedit/sampled_output")
BASE_DIR = Path("/data1/yuyangxin/datasets/anyedit")


def load_json_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_json_file(json_file: Path, img_processor: ImageProcessor, save_dir: Path):
    try:
        json_data = load_json_file(json_file)
        dataset_count = {}
        for dataset_name, dataset in json_data.items():
            # 计算每个样本的0像素比例
            mask_ratios = []
            for info in tqdm(dataset, desc=f"{json_file.name} 进度", leave=False):
                gt_mask = img_processor.load_image(info["gt_mask_path"], image_type="1")[0]
                gt_mask = np.array(gt_mask)
                zero_ratio = (gt_mask == 0).sum() / gt_mask.size
                # 只保留比例大于0.1且小于0.9的样本
                if 0.1 < zero_ratio < 0.9:
                    mask_ratios.append((zero_ratio, info))
            # 按0像素比例降序排序，取前1000
            mask_ratios.sort(key=lambda x: x[0], reverse=True)
            sorted_infos = [info for _, info in mask_ratios[:1000]]
            json_data[dataset_name] = sorted_infos
            dataset_count[dataset_name] = len(sorted_infos)
        save_json_file(save_dir / json_file.name, json_data)
        return dataset_count
    except Exception as e:
        return (json_file, str(e))


def main(dir_path: Path, max_workers: int = 16, debug: bool = False):
    save_path = Path("/data1/yuyangxin/datasets/anyedit/sort_result")
    save_path.mkdir(parents=True, exist_ok=True)

    img_processor = ImageProcessor()
    json_files = list(dir_path.glob("*.json"))
    error_files = []
    dataset_count = {}

    if debug:
        # 单线程调试模式
        for json_file in tqdm(json_files, desc="单线程调试进度"):
            result = process_json_file(json_file, img_processor, save_path)
            if isinstance(result, dict):
                for k, v in result.items():
                    dataset_count.setdefault(k, 0)
                    dataset_count[k] += v
            elif result:
                error_files.append(result)
    else:
        # 多线程模式
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_json_file, json_file, img_processor, save_path) for json_file in json_files
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="总体进度"):
                result = f.result()
                if isinstance(result, dict):
                    for k, v in result.items():
                        dataset_count.setdefault(k, 0)
                        dataset_count[k] += v
                elif result:
                    error_files.append(result)

    # 输出处理结果
    print("处理完成！")
    print("各dataset总数量统计：")
    for k, v in dataset_count.items():
        print(f"{k}: {v}")

    # 输出处理出错的文件
    if error_files:
        print("以下文件处理出错:")
        for ef, err in error_files:
            print(f"{ef}: {err}")
    else:
        print("所有文件处理成功！")


if __name__ == "__main__":
    main(SAVE_TARGET_DIR, max_workers=16, debug=False)
