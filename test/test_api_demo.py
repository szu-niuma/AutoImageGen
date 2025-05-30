import base64

from openai import OpenAI

api_base = "https://jeniya.cn/v1"
api_key = "sk-WY6YbQoyZMR425I2tAOLcxO9PEBCjERdTR3yV5PgaZfKAsdH"
client = OpenAI(api_key=api_key, base_url=api_base)

prompt = """
帮我去掉这个飞机
"""

result = client.images.edit(
    model="gpt-image-1",
    image=[
        open(r"E:\桌面\demo\resource\airplane.jpg", "rb"),
    ],
    prompt=prompt,
    mask=[
        open(r"E:\桌面\demo\resource\airplane_mask.png", "rb"),
    ],
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("./gift-basket.png", "wb") as f:
    f.write(image_bytes)
