import io
import json
import random
from pathlib import Path
from typing import Any, Dict

from loguru import logger
from PIL import Image, ImageColor, ImageDraw, ImageFont

additional_colors = [color_name for (color_name, color_code) in ImageColor.colormap.items()]


def load_json(file_path: Path, raise_exception: bool = True) -> Dict[str, Any]:
    """加载JSON文件"""

    def handle_error(message: str, exception: Exception = None):
        if raise_exception and exception:
            logger.error(message)
            raise exception
        return {}

    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.is_file():
        return handle_error(f"{file_path} 不存在", FileNotFoundError(f"{file_path} 不存在"))

    try:
        with file_path.open("r", encoding="utf-8") as json_file:
            content = json_file.read().strip()
            if not content:
                return handle_error(f"{file_path} 文件为空", ValueError(f"{file_path} 文件为空"))

            data = json.loads(content, strict=False)
            return data
    except (json.JSONDecodeError, IOError, OSError) as e:
        return handle_error(f"处理 {file_path} 时发生错误: {e}", e)


def save_json(save_path: Path, result):
    """保存JSON文件"""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    try:
        # 确保目标目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=4, sort_keys=True, ensure_ascii=False)
        logger.info(f"成功保存 JSON 文件到 {save_path}")
    except (IOError, OSError) as e:
        logger.error(f"写入文件 {save_path} 时发生错误: {e}")
    except TypeError as e:
        logger.error(f"JSON 序列化失败: {e}")


def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "orange",
        "pink",
        "purple",
        "brown",
        "gray",
        "beige",
        "turquoise",
        "cyan",
        "magenta",
        "lime",
        "navy",
        "maroon",
        "teal",
        "olive",
        "coral",
        "lavender",
        "violet",
        "gold",
        "silver",
    ] + additional_colors

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json.loads(bounding_boxes)):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0] / 1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1] / 1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2] / 1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.show()
