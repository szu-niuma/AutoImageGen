from config import GLOBAL_CONFIG
from auto_image_edit.image_analysis import ImageAnalysis
from auto_image_edit.image_creativity import ImageCreativity
from auto_image_edit.image_editor import ImageEditor

if __name__ == "__main__":
    instruction_generator = ImageAnalysis(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    image_path = r"E:\桌面\demo\resource\airplane.jpg"
    result = instruction_generator.run(image_path)

    image_creativity = ImageCreativity(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    result = image_creativity.run(result)

    image_editor = ImageEditor(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    result["image_edit"] = image_editor.run(result)
