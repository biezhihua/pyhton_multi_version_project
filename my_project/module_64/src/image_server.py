

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import onnxruntime

app = FastAPI()

def check_img_by_data(image1, image2):
    onnx_model_path = 'model.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    input1 = cv2.resize(image1, (105, 105)).astype(np.float32) / 255
    input2 = cv2.resize(image2, (105, 105)).astype(np.float32) / 255

    input1 = np.transpose(input1, (2, 0, 1))
    input2 = np.transpose(input2, (2, 0, 1))

    input1 = np.expand_dims(input1, axis=0)
    input2 = np.expand_dims(input2, axis=0)

    outputs = ort_session.run(None, {'x1': input1, 'x2': input2})
    return float(outputs[0][0][0])

@app.post('/compare')
async def compare_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    try:
        img1_bytes = await image1.read()
        img2_bytes = await image2.read()
        img1_np = np.frombuffer(img1_bytes, np.uint8)
        img2_np = np.frombuffer(img2_bytes, np.uint8)
        img1 = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)
        if img1 is None or img2 is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image data."})
        result = check_img_by_data(img1, img2)
        return {"result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("image_server:app", host="0.0.0.0", port=5000)
