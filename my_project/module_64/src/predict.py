import cv2
import onnx
import onnxruntime
import numpy as np

# 导出的ONNX模型的路径
onnx_model_path = 'model.onnx'

# 加载ONNX模型
onnx_model = onnx.load(onnx_model_path)

# 创建ONNX Runtime推理会话
ort_session = onnxruntime.InferenceSession(onnx_model_path)

image1 = cv2.imread(r'1.jpg')
image2 = cv2.imread(r'2.jpg')

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 将图片转换为numpy数组
input1 = cv2.resize(image1, (105, 105)).astype(np.float32) / 255
input2 = cv2.resize(image2, (105, 105)).astype(np.float32) / 255

# 确保输入的形状是 (C, H, W)
input1 = np.transpose(input1, (2, 0, 1))
input2 = np.transpose(input2, (2, 0, 1))

# 添加 batch 维度
input1 = np.expand_dims(input1, axis=0)
input2 = np.expand_dims(input2, axis=0)

outputs = ort_session.run(None, {'x1': input1, 'x2': input2})


# outputs 是一个列表，包含模型的所有输出
# outputs[0] 获取第一个输出（假设模型只有一个输出）
# outputs[0][0] 获取批次中的第一个样本的输出
# outputs[0][0][0] 获取输出的具体值
# 输出
print(outputs[0][0][0])
