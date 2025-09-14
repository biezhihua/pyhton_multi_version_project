import requests
import time

start = time.time()

# 请求 /detect_and_compare 接口
url = "http://127.0.0.1:5000/detect_and_compare"
chengyu = "华而不实"  # 示例汉字
check_text = "华"  # 示例汉字
img_path = r"E:\Projects\pyhton_multi_version_project\my_project\module_64\imgs_uploaded\华而不实\20250914_145318_478500.jpg"  # 示例图片路径

with open(img_path, "rb") as img_file:
    files = {"chengyu": (None, chengyu), "check_text": (None, check_text), "img": (img_path, img_file, "image/jpeg")}
    response = requests.post(url, files=files)
    end = time.time()
    print("Status:", response.status_code)
    print("耗时: {:.3f} 秒".format(end - start))
    print("Response:", response.json())
