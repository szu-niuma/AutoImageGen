import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image


class COCOProcessor:
    """COCO数据集处理器"""

    def __init__(self, dataset_path: str | Path, annotation_file: str | Path):
        """
        初始化COCO处理器

        Args:
            dataset_path: COCO数据集根目录路径
            annotation_file: 标注文件路径 (如: annotations/instances_train2017.json)
        """
        self.dataset_path = Path(dataset_path)
        self.annotation_file = Path(annotation_file)
        self.annotations = None
        self.categories = None
        self.images_info = None
        self._color_images_cache = None  # 添加彩色图片缓存

        # 随机种子
        random.seed(42)

    def load_annotations(self) -> bool:
        """加载COCO标注文件"""
        try:
            with self.annotation_file.open("r", encoding="utf-8") as f:
                self.annotations = json.load(f)

            self.categories = {cat["id"]: cat["name"] for cat in self.annotations["categories"]}
            self.images_info = {img["id"]: img for img in self.annotations["images"]}

            print(f"成功加载标注文件: {self.annotation_file}")
            print(f"图片数量: {len(self.images_info)}")
            print(f"类别数量: {len(self.categories)}")

            return True
        except Exception as e:
            print(f"加载标注文件失败: {e}")
            return False

    def check_image(self, image_path: Path) -> bool:
        """
        检查图片是否为彩色图片

        Args:
            image_path: 图片路径

        Returns:
            bool: True表示彩色图片，False表示灰度图片
        """
        try:
            # 检查文件大小，如果不超过200KB则返回False
            file_size = image_path.stat().st_size
            if file_size <= 300 * 1024:  # 200KB = 200 * 1024 bytes
                return False

            # 使用PIL检查
            with Image.open(image_path) as img:
                # 如果是RGB或RGBA模式，则为彩色
                if img.mode in ["RGB", "RGBA"]:
                    return True
                elif img.mode == "L":  # 灰度图
                    return False
                else:
                    # 对于其他模式，转换为RGB后检查
                    img_rgb = img.convert("RGB")
                    img_array = np.array(img_rgb)
                    # 检查R、G、B三个通道是否相同
                    return not np.allclose(img_array[:, :, 0], img_array[:, :, 1]) or not np.allclose(
                        img_array[:, :, 1], img_array[:, :, 2]
                    )
        except Exception as e:
            print(f"检查图片颜色失败 {image_path}: {e}")
            return False

    def get_image_annotations(self, image_id: int) -> List[Dict]:
        """
        获取指定图片的所有标注，去除重复的category_id
        当同一category_id有多个标注时，优先保留单独对象(iscrowd=0)

        Args:
            image_id: 图片ID

        Returns:
            List[Dict]: 标注列表，每个category_id只保留一个，优先保留单独对象
        """
        image_annotations = []
        seen_categories = set()  # 用于记录已经见过的category_id
        category_annotations = {}  # 临时存储每个category的所有标注

        # 第一步：收集所有该图片的标注，按category_id分组
        for ann in self.annotations["annotations"]:
            if ann["image_id"] == image_id:
                category_id = ann["category_id"]
                if category_id not in category_annotations:
                    category_annotations[category_id] = []
                category_annotations[category_id].append(ann)

        # 第二步：对每个category_id，优先选择iscrowd=0的标注
        for category_id, anns in category_annotations.items():
            # 优先选择iscrowd=0的标注
            selected_ann = None
            for ann in anns:
                if ann["iscrowd"] == 0:
                    selected_ann = ann
                    break

            # 如果没有iscrowd=0的标注，则选择第一个
            if selected_ann is None:
                selected_ann = anns[0]

            # 构建标注信息
            ann_info = {
                "category_name": self.categories[category_id],
                "bbox": selected_ann["bbox"],  # [x, y, width, height]
                "iscrowd": selected_ann["iscrowd"],
            }
            if "segmentation" in selected_ann:
                ann_info["segmentation"] = selected_ann["segmentation"]

            image_annotations.append(ann_info)

        return image_annotations

    def _build_color_images_index(self, max_num, image_folder: str = "train2017") -> List[Tuple[int, Dict, Path]]:
        """构建彩色图片索引，只在第一次调用时执行"""
        if self._color_images_cache is not None:
            return self._color_images_cache

        if not self.load_annotations():
            return []

        image_dir = self.dataset_path / image_folder
        if not image_dir.exists():
            print(f"图片目录不存在: {image_dir}")
            return []

        print("正在构建彩色图片索引...")
        color_images = []
        processed_count = 0

        for image_id, image_info in self.images_info.items():
            image_path = image_dir / image_info["file_name"]
            if image_path.exists():
                processed_count += 1
                if self.check_image(image_path):
                    color_images.append((image_id, image_info, image_path))

                if processed_count % 1000 == 0:
                    print(f"已检查: {processed_count} 张图片, 找到彩色图片: {len(color_images)} 张")
                if max_num and len(color_images) >= max_num:
                    print(f"已达到最大彩色图片数量需求限制: {max_num}")
                    break

        self._color_images_cache = color_images
        print(f"彩色图片索引构建完成! 总计: {len(color_images)} 张彩色图片")
        return color_images

    def random_sample_color_images(
        self,
        num_samples: int,
        target_dir: str | Path,
        image_folder: str = "train2017",
        save_labels: bool = True,
        labels_file: str = "MSCOCO.json",
    ) -> List[Tuple[Path, List[Dict]]]:
        """优化版本：使用预构建的彩色图片索引"""
        # 获取彩色图片索引
        color_images = self._build_color_images_index(max_num=num_samples * 3, image_folder=image_folder)

        if not color_images:
            print("没有找到彩色图片")
            return []

        # 创建目标目录
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 随机选择指定数量的彩色图片
        selected_images = random.sample(color_images, num_samples)

        results = []
        success_count = 0

        for image_id, image_info, image_path in selected_images:
            # 复制图片到目标目录
            target_image_path = target_dir / image_info["file_name"]
            try:
                shutil.copy2(image_path, target_image_path)
                success_count += 1
                print(f"已复制图片 {success_count}/{num_samples}: {image_info['file_name']}")

                # 获取标注信息
                annotations = self.get_image_annotations(image_id)
                results.append((target_image_path, annotations))

            except Exception as e:
                print(f"复制图片失败 {image_path}: {e}")

        print(f"抽取完成! 成功复制 {success_count} 张彩色图片到: {target_dir}")

        # 保存标签信息
        if save_labels and results:
            labels_path = target_dir / labels_file
            self.save_results(results, labels_path)

        return results

    def process_dataset(self, image_folder: str = "train2017") -> List[Tuple[Path, List[Dict]]]:
        """
        处理COCO数据集，返回彩色图片路径和对应标签

        Args:
            image_folder: 图片文件夹名称

        Returns:
            List[Tuple[Path, List[Dict]]]: (图片路径, 标注列表) 的列表
        """
        if not self.load_annotations():
            return []

        results = {}
        image_dir = self.dataset_path / image_folder

        if not image_dir.exists():
            print(f"图片目录不存在: {image_dir}")
            return []

        processed_count = 0
        color_count = 0

        for image_id, image_info in self.images_info.items():
            image_path = image_dir / image_info["file_name"]

            if not image_path.exists():
                continue

            processed_count += 1

            # 检查是否为彩色图片
            if self.check_image(image_path):
                color_count += 1
                # 获取该图片的标注
                annotations = self.get_image_annotations(image_id)
                results[image_path] = annotations

            # 每处理1000张图片输出一次进度
            if processed_count % 1000 == 0:
                print(f"已处理: {processed_count} 张图片, 彩色图片: {color_count} 张")

        print(f"处理完成! 总共处理: {processed_count} 张图片")
        print(f"彩色图片数量: {color_count} 张")
        print(f"返回结果数量: {len(results)} 条")

        return results

    def get_category_statistics(self, results: List[Tuple[Path, List[Dict]]]) -> Dict[str, int]:
        """
        统计各类别的出现次数

        Args:
            results: process_dataset返回的结果

        Returns:
            Dict[str, int]: 类别名称和出现次数的字典
        """
        category_count = {}

        for _, annotations in results:
            for ann in annotations:
                category_name = ann["category_name"]
                category_count[category_name] = category_count.get(category_name, 0) + 1

        return category_count

    def save_results(self, results: List[Tuple[Path, List[Dict]]], output_file: Path | str):
        """
        保存处理结果到JSON文件

        Args:
            results: process_dataset返回的结果
            output_file: 输出文件路径
        """
        output_file = Path(output_file)
        output_data = []

        for image_path, annotations in results:
            output_data.append({"image_path": str(image_path), "annotations": annotations})

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"结果已保存到: {output_file}")


def main(
    dataset_path: str | Path = None,
    annotation_file: str | Path = None,
    target_dir: str | Path = "sampled_images",
    num_samples: int = 2000,
):
    """示例用法"""
    # 创建处理器
    processor = COCOProcessor(dataset_path, annotation_file)

    # 随机抽取彩色图片
    results = processor.random_sample_color_images(
        num_samples=num_samples,
        target_dir=target_dir,
        image_folder="train2017",
        save_labels=True,
        labels_file="sampled_labels.json",
    )

    # 统计类别分布
    if results:
        category_stats = processor.get_category_statistics(results)
        print("\n类别统计:")
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"{category}: {count}")


if __name__ == "__main__":
    main(
        dataset_path=Path(r"E:\datasets\MSCOCO\train2017"),
        annotation_file=Path(r"E:\datasets\MSCOCO\annotations_trainval2017\annotations\instances_train2017.json"),
        target_dir=Path(r"E:\桌面\李老师科研小组\AutoImageGen\datasets\real_images"),
        num_samples=1000,
    )
