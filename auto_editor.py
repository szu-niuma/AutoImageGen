import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger
from tqdm import tqdm

from auto_image_edit import BallSegments, ImageProcessor
from config import GLOBAL_CONFIG


class ObjectRecognition:
    def __init__(self, output_dir, model_name="chatgpt", is_debug=False, max_num=2000, ref_img_path=None):
        if model_name == "chatgpt":
            self.llm = ChatOpenAI(**GLOBAL_CONFIG.get_config())
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        self.example_dir: Path = Path(__file__).parents[0] / "examples"
        self.max_num = max_num
        self.model_name = model_name
        self.is_debug = is_debug
        self.ref_img_path = ref_img_path

        if self.is_debug is True:
            logger.debug("开启Debug模式")

        self.image_processor = ImageProcessor()
        self.ball_segments = BallSegments(self.llm)

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def evaluate(self, target_image_path: Path):
        edited_img, _, _ = self.image_processor.load_image(target_image_path)
        edited_img_base64 = self.image_processor.get_base64(edited_img)
        image_info = self.get_image_info(edited_img_base64)
        try:
            response = self.ball_segments.run(image_info)
        except Exception as e:
            error_info = f"Error processing {target_image_path}: {e}"
            logger.error(traceback.format_exc())
            return error_info
        else:
            logger.info(response.model_dump_json())
            # 将结果序列化为JSON字符串格式
            if self.is_debug:
                target_img = self.image_processor.draw_box(edited_img, response, show_image=True)
                target_img_path = self.output_dir / f"{target_image_path.stem}_result.png"
                target_img.save(target_img_path, format="PNG")
                logger.debug(f"保存结果图片: {target_img_path}")
            return response.model_dump_json()

    def get_image_info(self, edited_img_base64):
        if self.ref_img_path:
            _, ref_img, _ = self.image_processor.load_image(self.ref_img_path)
            ref_img_base64 = self.image_processor.get_base64(ref_img)
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{ref_img_base64}"},
                },
                {
                    "type": "text",
                    "text": "上述图片是需要识别的物体的参考效果，该图中红框=篮筐，绿框=篮球",
                },
            ]
        else:
            content = []
        content.extend(
            [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{edited_img_base64}"},
                },
                {
                    "type": "text",
                    "text": "Detect the 2d bounding boxes of the basket and the ball (with “label” as topping description”)",
                },
            ]
        )
        return HumanMessage(content=content)

    def run(self, dataset_path):
        # 读取json文件
        dataset_path = Path(dataset_path)
        # 判断文件是否是目录还是文件
        if dataset_path.is_file():
            # 如果是文件，直接读取
            image_path_list = [dataset_path]
        elif dataset_path.is_dir():
            image_path_list = ImageProcessor.get_dataset(dataset_path)
        else:
            raise ValueError(f"无效的路径: {dataset_path}")

        # 使用线程池并行处理任务
        ret = []
        if self.is_debug:
            # 单线程处理
            for image_path in image_path_list:
                ret.append(self.evaluate(image_path))
        else:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_path = {
                    executor.submit(self.evaluate, image_path): image_path for image_path in image_path_list
                }
                for future in tqdm(as_completed(future_to_path), total=len(future_to_path), desc="Processing images"):
                    img_path = future_to_path[future]
                    try:
                        result = future.result()
                        ret.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {img_path}: {e}")
                        logger.error(traceback.format_exc())

        # 保存为json文件
        with open(self.output_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(ret, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 创建实例并运行评估
    object_rec = ObjectRecognition("./output", is_debug=True, ref_img_path=None)
    object_rec.run(r"E:\桌面\demo\resource\test_image\VID_20250408_193434_frame_000090.jpg")
