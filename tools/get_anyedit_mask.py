import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


def process_json_file(json_file: Path, dir_path: Path, img_processor: ImageProcessor):
    try:
        json_data = load_json_file(json_file)
        ret = {}
        for dataset_name, dataset in json_data.items():
            ret[dataset_name] = len(dataset)

            for info in tqdm(dataset, desc=f"{json_file.name} 进度", leave=False):
                real_image_path = BASE_DIR / info["real_image_path"]
                real_image = img_processor.load_image(real_image_path)[0]
                edit_method = info["edit_type"]
                ref_mask = None

                output_img_path = BASE_DIR / info["edited_image_path"]
                assert output_img_path.exists(), f"Output image not found: {output_img_path}"
                output_img = img_processor.load_image(output_img_path)[0]

                if output_img.size != real_image.size:
                    output_img = output_img.resize(real_image.size, Image.BILINEAR)

                gt_dir = dir_path / edit_method / "gt_mask_image"
                gt_dir.mkdir(parents=True, exist_ok=True)

                diff_mask = img_processor.compare_images_pixelwise(real_image, output_img, ref_mask, "HSV")
                diff_mask_path = gt_dir / f"{real_image_path.stem}.jpg"
                diff_mask.save(diff_mask_path)

                info["gt_mask_path"] = str(diff_mask_path.resolve())
        save_json_file(json_file, json_data)
        # 无错误
        return ret
    except Exception as e:
        return (json_file, str(e))


def main(dir_path: Path, max_workers: int = 8):
    img_processor = ImageProcessor()
    json_files = list(dir_path.glob("*.json"))
    error_files = []
    dataset_count = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_json_file, json_file, dir_path, img_processor) for json_file in json_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="总体进度"):
            result = f.result()
            if isinstance(result, dict):
                # 汇总每个json文件的dataset长度
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
    main(SAVE_TARGET_DIR)
