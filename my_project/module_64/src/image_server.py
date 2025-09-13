from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import onnxruntime

from PIL import Image, ImageDraw, ImageFont
import torch
import os

app = FastAPI()


def text_to_image(char, font_path="simsun.ttc", image_size=(100, 100), font_size=75):
    """生成单字符图片并返回 numpy 数组"""
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
    # 转为 numpy 数组
    return np.array(img)


def check_img_by_data(image1, image2):
    onnx_model_path = "model.onnx"
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


@app.post("/compare")
async def compare_images(
    image1: UploadFile = File(...), image2: UploadFile = File(...)
):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1_np = np.frombuffer(img1_bytes, np.uint8)
        img2_np = np.frombuffer(img2_bytes, np.uint8)
        img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            return JSONResponse(
                status_code=400, content={"error": "Invalid image data."}
            )
        result = check_img_by_data(img1, img2)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 新API: /detect_and_compare


def detect_and_compare_local(text, img_cv):
    """
    本地调用验证方法，text为字符，img_cv为OpenCV图片（numpy数组），返回识别与对比结果。
    """
    try:
        # 1. 生成文字图片（直接内存）
        char_img_cv = text_to_image(text)
        if char_img_cv is None:
            return {"error": "生成文字图片失败"}

        # 2. 检查图片
        if img_cv is None:
            return {"error": "上传图片无效"}

        # 3. 加载YOLOv5模型并识别
        model = torch.hub.load(
            repo_or_dir="E:\Projects\yolov5",
            model="custom",
            path="mhxy_text.pt",
            force_reload=True,
            source="local",
        )

        model.conf = 0.1
        model.iou = 0.45
        model.classes = None
        model.agnostic = False
        model.multi_label = False
        model.max_det = 1000

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and "cuda" in "cuda:0" else "cpu"
        )
        model.to(device)

        results = model(img_cv, size=640)
        boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        compare_results = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            crop_img = img_cv[int(y1) : int(y2), int(x1) : int(x2)]
            if crop_img.size == 0:
                continue
            try:
                sim = check_img_by_data(char_img_cv, crop_img)
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
        return {"results": compare_results}
    except Exception as e:
        return {"error": str(e)}


# FastAPI接口封装
@app.post("/detect_and_compare")
async def detect_and_compare(text: str = File(...), img: UploadFile = File(...)):
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
    uvicorn.run("image_server:app", host="0.0.0.0", port=5000)
