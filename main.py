from auto_image_edit.creativity_pipeline import CreativityPipeline
from config import GLOBAL_CONFIG

if __name__ == "__main__":
    instruction_generator = CreativityPipeline(GLOBAL_CONFIG.get_config(), "output", is_debug=True)
    image_path = r".\resource\airplane.jpg"
    result = instruction_generator.run(image_path)
