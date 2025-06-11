from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_image_edit.utils.image_similarity import ImageSimilarity
from tqdm import tqdm


def collect_image_sets(json_paths):
    real_data, fake_data = set(), set()
    for path in json_paths:
        with Path(path).open("r") as f:
            data = json.load(f)
        for item in data:
            real_data.add(item["real_image"])
            fake_data.add((item["real_image"], item["fake_image"], item["edit_method"]))
    return real_data, fake_data


def main():
    base_dir = Path("/home/yuyangxin/data/dataset/FragFake/output")
    json_files = [
        base_dir / "FragFake_train_easy.json",
        base_dir / "FragFake_train_hard.json",
    ]
    real_data, fake_data = collect_image_sets(json_files)
    print(f"Collected {len(real_data)} real images and {len(fake_data)} fake images.")

    ret = []
    for real_image in real_data:
        ret.append([real_image, "positive", 0])

    image_sim = ImageSimilarity()

    # 定义线程处理函数
    def process_fake(fake_info):
        real_image, fake_image, method = fake_info
        gt_masks = Path(f"/home/yuyangxin/data/dataset/FragFake/resource/gt_masks/{method}")
        gt_masks.mkdir(parents=True, exist_ok=True)
        resize_fake = Path(f"/home/yuyangxin/data/dataset/FragFake/resource/resize_fake/{method}")
        resize_fake.mkdir(parents=True, exist_ok=True)

        # 检查并调整尺寸
        target_img = image_sim.check_and_resize(real_image, fake_image)
        target_path = resize_fake / Path(fake_image).name
        if target_path.exists():
            return None

        target_img.save(target_path)
        lpips = image_sim.compare_images_lpips(real_image, target_path, gray=True)
        mask_path = gt_masks / Path(fake_image).name
        lpips.save(mask_path)
        return [str(target_path), str(mask_path), 1]

    # 多线程并发执行
    max_workers = 32
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_fake, fi): fi for fi in fake_data}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating masks", unit="img"):
            res = fut.result()
            if res:
                ret.append(res)

    # 保存结果
    output_path = base_dir / "FragFake.json"
    with output_path.open("w") as f:
        json.dump(ret, f, indent=4)


if __name__ == "__main__":
    main()
