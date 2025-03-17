import requests
import json

import base64
from io import BytesIO
from PIL import Image
# API 服务器地址
api_url = "http://127.0.0.1:8000/v1/chat/completions"

def encode_base64_image(image:Image.Image, image_format="PNG") -> str:
    im_file = BytesIO()
    image.save(im_file, format=image_format)
    im_bytes = im_file.getvalue()
    im_64 = base64.b64encode(im_bytes).decode("utf-8")
    return im_64

def encode_base64_image_url(image:Image.Image, image_format="PNG") -> str:
    return f"data:image/{image_format};base64,{encode_base64_image(image, image_format)}"
# image = Image.open("/map-vepfs/yuansheng/LongContext/lmms-eval/test_0.jpg")
# image_url = encode_base64_image_url(image)
payload = {
    "model": "/data/yuansheng/checkpoint/Qwen2.5-VL-7B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这个图"},
                {"type": "text", "text": "是什么？"},
                {"type": "text", "text": "详细解释图内函数与各个之间的交点等，以及相关的数学知识"},
                {"type": "text", "text": "重复上面的解释十遍"}
            ]
        }
    ],
    "max_tokens": 2,
    "temperature": 0,
    "top_p": None
}



# 发送请求
response = requests.post(api_url, json=payload)

# 判断请求是否成功，并打印返回信息
if response.ok:
    print("Response:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
else:
    print("Request failed with status code:", response.status_code)
    print(response.text)
