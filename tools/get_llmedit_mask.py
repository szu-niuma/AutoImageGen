import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm  # 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_image_edit.utils.image_similarity import ImageSimilarity

SAVE_TARGET_DIR = Path("/home/yuyangxin/data/dataset/custom_dataset/llm_edit")


def load_json_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def process_json_file():
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
        assert tp_file.is_file(), f"对应的tp文件不存在: {tp_file}"
        pixel_mask = ImageSimilarity.compare_images_pixelwise(au_file, tp_file, gray=True)
        pixel_mask_path = gt_dir / f"pixel_gt_{index}.jpg"
        pixel_mask.save(pixel_mask_path)

        lpips_mask = ImageSimilarity.compare_images_lpips(au_file, tp_file, gray=True)
        lpips_mask_path = gt_dir / f"lpips_gt_{index}.jpg"
        lpips_mask.save(lpips_mask_path)

        json_data.append(
            {
                "real_image": au_file.absolute().as_posix(),
                "fake_image": tp_file.absolute().as_posix(),
                "pixel_gt_mask": pixel_mask_path.absolute().as_posix(),
                "lpips_gt_mask": lpips_mask_path.absolute().as_posix(),
            }
        )

        print(f"Processed {au_file.name} and saved masks to {pixel_mask_path.name} and {lpips_mask_path.name}")
    save_json_file(SAVE_TARGET_DIR / "inst.json", json_data)


def main():
    process_json_file()


if __name__ == "__main__":
    main()
