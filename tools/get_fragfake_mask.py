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


def get_diff(real_img, fake_img):
    """
    获取真实图像和伪造图像之间的差异
    :param real_image: 真实图像路径
    :param fake_image: 伪造图像路径
    :param img_sim: ImageSimilarity 实例
    :return: 差异图像
    """
    lpips_diff = ImageSimilarity.compare_images_lpips(real_img, fake_img, heatmap=False, norm="zscore", gray=False)
    pixel_diff = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, heatmap=False, norm="zscore", gray=False, color_space="lab")
    res = (lpips_diff + pixel_diff) / 2
    return res


def main():
    base_dir = Path("/home/yuyangxin/data/dataset/FragFake/output")
    json_files = [
        base_dir / "FragFake_train_easy.json",
        base_dir / "FragFake_train_hard.json",
    ]
    real_data, fake_data = collect_image_sets(json_files)
    print(f"Collected {len(real_data)} real images and {len(fake_data)} fake images.")

    # 定义线程处理函数
    def process_fake(fake_info):
        real_image, fake_image, method = fake_info

        gt_masks = Path(f"/home/yuyangxin/data/dataset/FragFake/resource/gt_masks/{method}")
        gt_masks.mkdir(parents=True, exist_ok=True)
        resize_fake = Path(f"/home/yuyangxin/data/dataset/FragFake/resource/resize_fake/{method}")
        resize_fake.mkdir(parents=True, exist_ok=True)

        # 检查并调整尺寸
        real_img, fake_img = ImageSimilarity.check_and_resize(real_image, fake_image)
        target_path = resize_fake / Path(fake_image).name
        fake_img.save(target_path)

        lpips = ImageSimilarity.compare_images_lpips(real_img, fake_img, gray=False)
        pixel = ImageSimilarity.compare_images_pixelwise(real_img, fake_img, gray=False, color_space="lab")
        mask_info = (lpips + pixel) / 2
        mask_info = ImageSimilarity.to_gray(mask_info)

        mask_path = gt_masks / Path(fake_image).name
        mask_info.save(mask_path)

        return [[real_image, "Negative", 0], [str(target_path), str(mask_path), 1]]

    # 多线程并发执行
    max_workers = 64
    ret = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fi in fake_data:
            futures[executor.submit(process_fake, fi)] = fi

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating masks", unit="img"):
            ret.extend(fut.result())

    # 保存结果
    output_path = base_dir / "FragFake.json"
    with output_path.open("w") as f:
        json.dump(ret, f, indent=4)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
