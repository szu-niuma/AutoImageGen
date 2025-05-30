import traceback

from auto_image_edit.creativity_pipeline import CreativityPipeline
from config import GLOBAL_CONFIG


def main():
    # 创建创意管道实例
    creativity_pipeline = CreativityPipeline(
        config=GLOBAL_CONFIG.get_config(),
        out_dir=r"E:\桌面\李老师科研小组\AutoImageGen\datasets\instruction",  # 替换为实际的输出目录
        max_num=10,  # 可选，处理的最大图片数量
        is_debug=False,  # 是否启用调试模式
    )

    target_json = r"E:\桌面\李老师科研小组\AutoImageGen\datasets\sampled_labels.json"
    creativity_pipeline.run(target_json)  # 替换为实际的图片路径或目录


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
