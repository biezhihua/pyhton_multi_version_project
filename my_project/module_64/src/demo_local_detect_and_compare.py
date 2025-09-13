import cv2
from image_server import detect_and_compare_local

if __name__ == "__main__":
    text = '断'
    img_path = 'test1.bmp'
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        print(f"图片读取失败: {img_path}")
    else:
        result = detect_and_compare_local(text, img_cv)
        print(result)
