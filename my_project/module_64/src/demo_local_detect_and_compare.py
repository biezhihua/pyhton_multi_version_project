import cv2
from detect_server import detect_and_compare_local

if __name__ == "__main__":
    chengyu = "断章取义"
    # 要检测的文字
    text = "断"
    # 要检测的图片路径
    img_path = "test1.bmp"

    # 使用OpenCV读取图片为numpy数组
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"图片读取失败: {img_path}")
    else:
        # 调用本地检测与对比方法，返回识别结果和相似度
        result = detect_and_compare_local(text, img_cv)
        # 输出结果，包含检测框、置信度、类别和与文字图片的相似度
        print(result)
