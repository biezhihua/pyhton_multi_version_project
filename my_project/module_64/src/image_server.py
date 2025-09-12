
from flask import Flask, request, jsonify
import cv2
import numpy as np
import onnx
import onnxruntime
import tempfile
import os

app = Flask(__name__)

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

@app.route('/compare', methods=['POST'])
def compare_images():
    file1 = request.files.get('image1')
    file2 = request.files.get('image2')
    if not file1 or not file2:
        return jsonify({'error': 'Both image1 and image2 are required.'}), 400

    # 读取图片数据为 numpy 数组
    file1_bytes = np.frombuffer(file1.read(), np.uint8)
    file2_bytes = np.frombuffer(file2.read(), np.uint8)
    img1 = cv2.imdecode(file1_bytes, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(file2_bytes, cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return jsonify({'error': 'Invalid image data.'}), 400

    # 调用模型进行比较
    try:
        result = check_img_by_data(img1, img2)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
