import traceback

from auto_image_edit.analysis_pipeline import AnalysisPipeline
from config import GLOBAL_CONFIG


def main():
    # 创建创意管道实例
    creativity_pipeline = AnalysisPipeline(config=GLOBAL_CONFIG.get_config(), is_debug=False)
    target_json_dir = "/home/yuyangxin/data/finetune-qwen/resource/datasets/controllable_edit/FragFake_train_hard.json"
    creativity_pipeline.run(target_json_dir)  # 替换为实际的图片路径或目录


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
