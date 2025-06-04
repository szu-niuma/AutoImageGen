import json
import os
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm  # 新增

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_image_edit.utils.image_processor import ImageProcessor

SAVE_TARGET_DIR = Path("/home/yuyangxin/data/dataset/custom_dataset/brushedit")
OUTPUTS_DIR = Path("/home/yuyangxin/data/BrushEdit/outputs")


def load_json_file(file_path: Path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(file_path: Path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# 2. 获取每个文件的mask文件
def main(dir_path: Path):
    img_processor = ImageProcessor()
    json_files = list(dir_path.glob("*.json"))
    error_files = []
    for json_file in tqdm(json_files, desc="处理进度"):  # 添加进度条
        try:
            json_data = load_json_file(json_file)
            real_image = img_processor.load_image(json_data["real_image"])[0]
            for info in json_data["creativity"]:
                edit_method = info["edit_method"]
                if info.get("inst_mask"):
                    ref_mask = img_processor.load_image(info["inst_mask"])[0]
                    # 确保 ref_mask 的尺寸与 real_image 一致
                    if ref_mask.size != real_image.size:
                        ref_mask = ref_mask.resize(real_image.size, Image.BILINEAR)
                else:
                    ref_mask = None

                # 加载输出图片
                output_img_path = OUTPUTS_DIR / edit_method / f"{json_file.stem}.jpg"
                assert output_img_path.exists(), f"Output image not found: {output_img_path}"
                output_img = img_processor.load_image(output_img_path)[0]

                # 如果output_img的尺寸与real_image不一致，则调整output_img的尺寸
                if output_img.size != real_image.size:
                    output_img = output_img.resize(real_image.size, Image.BILINEAR)

                # 获取 diff mask
                diff_mask = img_processor.compare_images_pixelwise(real_image, output_img, ref_mask, "HSV")

                # 构建保存路径
                gt_dir = SAVE_TARGET_DIR / "Gt" / edit_method
                ref_dir = SAVE_TARGET_DIR / "Ref" / edit_method
                out_dir = SAVE_TARGET_DIR / "Tp" / edit_method
                gt_dir.mkdir(parents=True, exist_ok=True)
                ref_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)

                diff_mask_path = gt_dir / f"{json_file.stem}.jpg"
                ref_mask_path = ref_dir / f"{json_file.stem}.jpg"
                output_img_save_path = out_dir / f"{json_file.stem}.jpg"

                # 保存图片
                diff_mask.save(diff_mask_path)
                ref_mask.save(ref_mask_path)
                output_img.save(output_img_save_path)

                # 更新 json 信息
                info["inst_mask"] = str(ref_mask_path)
                info["output_img"] = str(output_img_save_path)
                info["gt_mask"] = str(diff_mask_path)
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
            error_files.append(json_file)
        else:
            # 保存修改后的 json 文件
            save_json_file(json_file, json_data)

    # 处理完成后输出结果
    print("处理完成！")
    if error_files:
        print("以下文件处理出错:")
        for ef in error_files:
            print(ef)
    else:
        print("所有文件处理成功！")


if __name__ == "__main__":
    dir_path = Path("/home/yuyangxin/data/AutoImageGen/datasets/instruction/EditInstruction")
    main(dir_path)
