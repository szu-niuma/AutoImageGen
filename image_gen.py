import os
import traceback

from auto_image_edit import CreativityPipeline
from config import GLOBAL_CONFIG


def read_txt_file(file_path="/data1/yuyangxin/datasets/xinye/train.txt", base_path="/data1/yuyangxin/datasets/xinye/train/images/"):
    """
    读取txt文件，提取标签为0的图片文件名

    Args:
        file_path: txt文件路径
        base_path: 图片文件的基础路径

    Returns:
        list: 包含完整路径的图片文件名列表
    """
    result = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    parts = line.split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = parts[1]

                        # 只收集标签为0的文件
                        if label == "0":
                            full_path = base_path + filename
                            if os.path.exists(full_path):
                                result.append({
                                    "img_path": full_path,
                                    "label": 0,
                                })

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")

    # 保存为json文件
    output_file = "./xinye_train.json"
    import json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_file}")
    return result


def main():
    # 创建创意管道实例
    creativity_pipeline = CreativityPipeline(config=GLOBAL_CONFIG.get_config(), is_debug=False, max_num=1000, image_type="face")
    target_json_dir = "/home/yuyangxin/data/AutoImageGen/xinye_train.json"
    creativity_pipeline.run(target_json_dir)  # 替换为实际的图片路径或目录


if __name__ == "__main__":
    main()
    print("Image generation script executed successfully.")
