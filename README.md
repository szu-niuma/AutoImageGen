# 项目简介-DEMO
本项目基于 LangChain 与 OpenAI 接口，结合图像预处理与视频关键帧提取，实现对图像中篮球及相关设施的实例分割与识别，并将结果保存为 JSON 格式。



# 主要功能
视频关键帧提取：processor/video_process.py

图像加载与预处理、掩码操作：utils.py & image_processor.py

基于 LLM 的实例分割提示词封装：prompts/ball_seg.txt 与类 BallSegments

主流程入口：object_recognition.py

# 配置说明
## 在./config文件夹下创建.env文件, 写入如下配置条目
`
OPENAI_API_KEY=your_api_key`

`
OPENAI_BASE_URL=your_base_url`

`
OPENAI_MODEL_NAME=your_model_name`



# prompt
system prompt: `./llm_object/prompts/object_detection.txt`

user prompt: `Detect the 2d bounding boxes of the basket and the ball (with “label” as topping description”)`

# TODO
1. 将图像输入可控图像编辑软件实现对应的结果
2. 将图像输入大模型中生成对应的编辑结果