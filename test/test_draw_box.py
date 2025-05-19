import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PIL import Image

from llm_object.processor.image_processor import ImageProcessor
from llm_object.prompts.models.object_model import ObjectModels, ResponseObject

# [{"box_2d": [397, 518, 573, 753], "label": "basket"}, {"box_2d": [296, 668, 398, 831], "label": "ball"}]
if __name__ == "__main__":
    infos = ResponseObject(
        objects=[
            ObjectModels(label="basket", box_2d=[397, 518, 573, 753]),
            ObjectModels(label="ball", box_2d=[296, 668, 398, 831]),
        ]
    )
    image = Image.open(r"E:\桌面\demo\resource\test_image\VID_20250408_193434_frame_000090.jpg")
    image = ImageProcessor().draw_box(image, infos, show_image=True)
