import requests
import time

start = time.time()

# 请求 /detect_and_compare 接口
url = "http://127.0.0.1:5000/detect_and_compare"
text = "断"  # 示例汉字
img_path = "test1.bmp"  # 示例图片路径

with open(img_path, "rb") as img_file:
    files = {"text": (None, text), "img": (img_path, img_file, "image/jpeg")}
    response = requests.post(url, files=files)
    end = time.time()
    print("Status:", response.status_code)
    print("耗时: {:.3f} 秒".format(end - start))
    print("Response:", response.json())
