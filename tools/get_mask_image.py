import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from pycocotools import mask as maskUtils
from utils.AuthV3Util import addAuthParams

img_dir = Path("/home/yuyangxin/data/AutoImageGen/datasets/real_images")
mask_dir = Path("/home/yuyangxin/data/AutoImageGen/datasets/mask_images")
edit_instruction_dir = Path("/home/yuyangxin/data/AutoImageGen/datasets/instruction/EditInstruction")
edit_instruction_dir.mkdir(exist_ok=True)

# 您的应用ID
APP_KEY = "55e3f0d790f02dd1"
# 您的应用密钥
APP_SECRET = "KZcsPCURHNW7XzALbJhXopRrXk7L7SKV"


def translate(info):
    """
    note: 将下列变量替换为需要请求的参数
    """
    time.sleep(1)  # 避免请求过于频繁
    data = {"q": info, "from": "zh-CHS", "to": "en"}
    addAuthParams(APP_KEY, APP_SECRET, data)
    header = {"Content-Type": "application/x-www-form-urlencoded"}
    res = doCall("https://openapi.youdao.com/api", header, data, "post")
    return res.json()["translation"][0]


def doCall(url, header, params, method):
    if "get" == method:
        return requests.get(url, params)
    elif "post" == method:
        return requests.post(url, params, header)


def get_mask(json_path: str):
    json_path = Path(json_path)
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {json_path}, 错误: {e}")
        return

    save_json_path = edit_instruction_dir / json_path.name
    if save_json_path.exists():
        print(f"文件已存在，跳过处理: {save_json_path}")
        return

    img_path = img_dir / f"{json_path.stem}.jpg"
    data["real_image"] = str(img_path)

    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"图像文件未找到: {img_path}")
    img_height, img_width = img.shape[:2]

    object_info = {}
    for obj in data.get("analysis", []):
        seg = obj.get("segmentation")
        category = obj.get("category_name")
        if seg is None or category is None:
            raise ValueError(f"缺少 segmentation 或 category_name 字段: {obj}")
        try:
            # 处理 segmentation 格式
            if isinstance(seg, list):
                rles = maskUtils.frPyObjects(seg, img_height, img_width)
                rle = maskUtils.merge(rles)
            else:
                rle = seg
            mask = maskUtils.decode(rle) * 255
        except Exception as e:
            print(f"分割掩码处理失败: {category}, 错误: {e}")
            continue
        safe_category = category.replace(" ", "_")
        out_path = mask_dir / f"{json_path.stem}_{safe_category}.png"
        try:
            cv2.imwrite(str(out_path), mask)
            object_info[category] = str(out_path)
        except Exception as e:
            raise IOError(f"保存掩码图像失败: {out_path}, 错误: {e}")

    # 整合 creativity 信息
    for info in data.get("creativity", []):
        edited_object = info.get("edited_object")
        # 判断 edited_object的语言是否为中文. 如果是中文，则翻译为英文
        if edited_object and re.search(r"[\u4e00-\u9fff]", edited_object):
            edited_object = translate(edited_object).lower()
            # 如果edited_object以the开头，则去掉the
            if edited_object.startswith("the "):
                edited_object = edited_object[4:]
        if not edited_object or edited_object not in object_info:
            if info["edit_method"] == "Object_Generation":
                continue  # 如果是对象生成方法，跳过
            raise ValueError(f"编辑对象掩码未找到: {edited_object}")
        else:
            info["inst_mask"] = object_info[edited_object]
        # 将"edit_prompt"翻译为英文
        info["edit_prompt_en"] = translate(info.get("edit_prompt", ""))

    # 保存到 EditInstruction 目录
    try:
        with open(edit_instruction_dir / f"{json_path.stem}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        raise IOError(f"保存编辑指令 JSON 文件失败: {save_json_path}, 错误: {e}")


if __name__ == "__main__":
    creativity_pipeline_dir = Path("/home/yuyangxin/data/AutoImageGen/datasets/instruction/CreativityPipeline")
    json_files = list(creativity_pipeline_dir.glob("*.json"))
    for json_file in json_files:
        print(f"处理文件: {json_file}")
        get_mask(json_file)
    print("所有文件处理完成。")
