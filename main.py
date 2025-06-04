import traceback

from auto_image_edit.diff_pipeline import DiffPipeline
from config import GLOBAL_CONFIG


def main():
    # 创建创意管道实例
    creativity_pipeline = DiffPipeline(
        config=GLOBAL_CONFIG.get_config(),
        is_debug=False,  # 是否启用调试模式
    )
    target_json_dir = "/home/yuyangxin/data/dataset/coverage/ins.json"
    creativity_pipeline.run(target_json_dir)  # 替换为实际的图片路径或目录


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
