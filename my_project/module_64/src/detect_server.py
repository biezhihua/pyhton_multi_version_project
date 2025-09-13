from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import onnxruntime
import time

from PIL import Image, ImageDraw, ImageFont
import torch
import os

app = FastAPI()


def text_to_image(char, font_path="simsun.ttc", image_size=(100, 100), font_size=75):
    """
    生成单字符图片，保存到本地临时文件，并返回 numpy 数组。
    """
    import tempfile

    img = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"警告: 无法加载字体 {font_path}, 使用默认字体")
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2
    draw.text((x, y), char, fill="black", font=font)
    # 保存到本地临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=f"tmp_{char}_img_", dir=".") as tmp:
        img.save(tmp.name)
        print(f"生成的字符图片已保存到: {tmp.name}")
    # 返回 numpy 数组
    return np.array(img)


def check_img_sim_by_data(image1, image2) -> float:
    onnx_model_path = "mhxy_text_sim_model.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    input1 = cv2.resize(image1, (105, 105)).astype(np.float32) / 255
    input2 = cv2.resize(image2, (105, 105)).astype(np.float32) / 255

    input1 = np.transpose(input1, (2, 0, 1))
    input2 = np.transpose(input2, (2, 0, 1))

    input1 = np.expand_dims(input1, axis=0)
    input2 = np.expand_dims(input2, axis=0)

    outputs = ort_session.run(None, {"x1": input1, "x2": input2})
    return float(outputs[0][0][0])


# 新API: /detect_and_compare
def detect_and_compare_local(text, img_cv):
    """
    本地调用验证方法。
    参数:
        text: 要检测的字符
        img_cv: OpenCV图片（numpy数组）
    返回:
        检测框、置信度、类别和与文字图片的相似度
    """
    try:
        # 1. 生成文字图片（直接内存，作为对比模板）
        char_img_cv = text_to_image(text)
        if char_img_cv is None:
            return {"error": "生成文字图片失败"}

        # 2. 检查输入图片是否有效
        if img_cv is None:
            return {"error": "上传图片无效"}

        # 3. 加载YOLOv5模型（本地已训练模型）并进行目标检测
        model = torch.hub.load(
            repo_or_dir="E:\Projects\yolov5",  # YOLOv5源码路径
            model="custom",
            path="mhxy_text.pt",
            force_reload=True,
            source="local",
        )
        # 设置模型参数
        model.conf = 0.1  # 置信度阈值
        model.iou = 0.45  # IOU阈值
        model.classes = None  # 所有类别
        model.agnostic = False  # 类别无关NMS
        model.multi_label = False
        model.max_det = 1000  # 最大检测数

        # 4. 设置推理设备（优先使用GPU）
        device = torch.device("cuda:0" if torch.cuda.is_available() and "cuda" in "cuda:0" else "cpu")
        model.to(device)

        # 5. 检测图片，获取所有检测框

        start = time.time()
        results = model(img_cv, size=640)
        boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]
        end = time.time()
        print("图片检测耗时: {:.3f} 秒".format(end - start))

        # 6. 遍历所有检测框，分别与文字图片对比相似度
        compare_results = []
        max_sim = -float("inf")
        max_crop_img = None
        max_box = None
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            crop_img = img_cv[int(y1) : int(y2), int(x1) : int(x2)]
            if crop_img.size == 0:
                continue
            try:
                # 计算相似度分数
                start = time.time()
                sim = check_img_sim_by_data(char_img_cv, crop_img)
                end = time.time()
                print("相似度检测耗时: {:.3f} 秒".format(end - start) + " sim: {:.3f}".format(sim))
            except Exception as e:
                sim = None
            compare_results.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "cls": int(cls),
                    "similarity": sim,
                }
            )
            # 记录最大sim值的区域
            if sim is not None and sim > max_sim:
                max_sim = sim
                max_crop_img = crop_img
                max_box = [int(x1), int(y1), int(x2), int(y2)]

        # 7. 保存sim值最高的区域截图到本地
        if max_crop_img is not None:
            import tempfile
            import datetime

            filename = f"tmp_{text}_max_sim_crop_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(filename, max_crop_img)
            print(f"已保存sim值最高的区域截图: {filename}, box: {max_box}, sim: {max_sim}")

        # 8. 返回所有检测结果
        return {"results": compare_results}
    except Exception as e:
        return {"error": str(e)}


# FastAPI接口封装
@app.post("/detect_and_compare")
async def detect_and_compare(text: str = File(...), img: UploadFile = File(...)):
    """
    text: 字符
    img: 图片
    img_size: 图片尺寸（不传递给本地方法，仅用于API参数）
    detect_rect: 识别区域，格式为"x1,y1,x2,y2"
    """
    try:
        img_bytes = await img.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        result = detect_and_compare_local(text, img_cv)
        if "error" in result:
            return JSONResponse(status_code=500, content=result)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run("detect_server:app", host="0.0.0.0", port=5000)
