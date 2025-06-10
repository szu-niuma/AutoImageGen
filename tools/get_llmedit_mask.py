import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm  # 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_image_edit.utils.image_processor import ImageProcessor

SAVE_TARGET_DIR = Path("/home/yuyangxin/data/dataset/custom_dataset/llm_edit")


def load_json_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_json_file(img_processor: ImageProcessor):
    au_dir: Path = SAVE_TARGET_DIR / "Au"
    tp_dir = SAVE_TARGET_DIR / "Tp"
    gt_dir = SAVE_TARGET_DIR / "Gt"

    json_data = []
    # 遍历au_dir下的所有文件
    for au_file in au_dir.glob("*.jpg"):
        if not au_file.is_file():
            continue

        # 将文件移动到tp_dir
        # 获取对应的tp_dir路径
        index = au_file.stem.split("_")[-1]
        tp_file = tp_dir / f"fake_{index}.jpg"
        real_image = img_processor.load_image(au_file)[0]
        fake_image = img_processor.load_image(tp_file)[0]
        if fake_image.size != real_image.size:
            raise ValueError(f"Image size mismatch: {fake_image.size} != {real_image.size}")

        diff_mask = img_processor.compare_images_pixelwise(real_image, fake_image, None, "HSV")
        diff_mask_path = gt_dir / f"gt_{index}.jpg"
        diff_mask.save(diff_mask_path)
        json_data.append(
            {
                "real_image": au_file.absolute().as_posix(),
                "fake_image": tp_file.absolute().as_posix(),
                "gt_mask": diff_mask_path.absolute().as_posix(),
            }
        )
    save_json_file(SAVE_TARGET_DIR / "inst.json", json_data)


def main():
    img_processor = ImageProcessor()
    process_json_file(img_processor)


if __name__ == "__main__":
    main()
