import cv2
import numpy as np
from PIL import Image
import cv2
import numpy as np
from detect_server import detect_and_compare_local

if __name__ == "__main__":
    chengyu = "华而不实"  # 示例汉字
    check_text = "华"  # 示例汉字
    img_path = r"E:\Projects\pyhton_multi_version_project\my_project\module_64\imgs_uploaded\华而不实\20250914_145318_478500.jpg"  # 示例图片路径

    # 使用OpenCV读取图片为numpy数组
    img = Image.open(img_path)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"图片读取失败: {img_path}")
    else:
        # 调用本地检测与对比方法，返回识别结果和相似度
        result = detect_and_compare_local(chengyu, check_text, img_cv)
        # 输出结果，包含检测框、置信度、类别和与文字图片的相似度
        print(result)
