from config import GLOBAL_CONFIG
from llm_object.image_analysis import ImageAnalysis
from llm_object.image_creativity import ImageCreativity
from llm_object.image_editor import ImageEditor

if __name__ == "__main__":
    instruction_generator = ImageAnalysis(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    image_path = r"E:\桌面\demo\resource\airplane.jpg"
    result = instruction_generator.run(image_path)

    image_creativity = ImageCreativity(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    result = image_creativity.run(result)

    image_editor = ImageEditor(GLOBAL_CONFIG.get_config(), "output", is_debug=True, max_num=1)
    result["image_edit"] = image_editor.run(result)
