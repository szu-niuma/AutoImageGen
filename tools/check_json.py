# 选择文件夹下的所有json文件
# 读取json文件内容
# 如果"creativity"字段不存在，或者为空, 则删除这个json文件

import json
import os


def check_and_delete_json_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # 检查"creativity"字段
                if "creativity" not in data or not data["creativity"]:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")


if __name__ == "__main__":
    folder = r"E:\桌面\李老师科研小组\AutoImageGen\datasets\instruction\CreativityPipeline"
    check_and_delete_json_files(folder)
