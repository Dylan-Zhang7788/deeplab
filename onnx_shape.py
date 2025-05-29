import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession("/home/ems/zhangdi_ws/deeplab/end2end-m.onnx")

# 获取输入信息
inputs_info = session.get_inputs()
print("Input(s) of the model:")
for input in inputs_info:
    print(f"Name: {input.name}, Shape: {input.shape}, Type: {input.type}")