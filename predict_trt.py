import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
# /usr/src/tensorrt/bin/trtexec --onnx=end2end_512.onnx --fp16 --saveEngine=rdrnet_0529.engine
# --minShapes=input:1x3x768x1024 --optShapes=input:1x3x768x1024 --maxShapes=input:1x3x768x1024
# 用这一行命令把onnx转成trt

# 常量定义
MODEL_PATH = "seg_model_resnet.trt"
TEST_IMG_DIR = "test_img"
OUTPUT_DIR = "test_results"
INPUT_H = 768 # 根据你的模型输入尺寸调整
INPUT_W = 1024  # 根据你的模型输入尺寸调整

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载TensorRT引擎
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# 分配缓冲区的辅助函数
def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

# 执行推理
def do_inference(engine, h_input, d_input, h_output, d_output, img):
    np.copyto(h_input, img.ravel())
    
    # 创建执行上下文
    context = engine.create_execution_context()                                                                                                                                                                                                                                                                                                                                                                              
    
    # 传输输入数据到GPU
    cuda.memcpy_htod(d_input, h_input)
    
    # 执行推理
    context.execute_v2(bindings=[int(d_input), int(d_output)])
    
    # 传输输出回CPU
    cuda.memcpy_dtoh(h_output, d_output)

    output_shape = engine.get_binding_shape(1)  # 获取模型输出形状
    output = h_output.reshape(output_shape)     # 自动处理 batch=1 的情况
    
    return output

# 主处理函数
def process_images():
    # 加载TensorRT引擎
    engine = load_engine(MODEL_PATH)
    h_input, d_input, h_output, d_output = allocate_buffers(engine)
    
    # 遍历测试图像
    for img_name in os.listdir(TEST_IMG_DIR):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        if not os.path.isfile(img_path):
            continue
            
        # 读取图像
        img = Image.open(img_path).convert('RGB')
        # img = img.resize((512, 512))
        img = np.array(img).astype(np.float32) / 255.0  # shape: (H, W, 3)

        if img is None:
            continue
            
        orig_h, orig_w = img.shape[:2]
        # 预处理图像
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std  # 广播机制自动处理通道维度
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0) 
        
        # 执行推理
        output = do_inference(engine, h_input, d_input, h_output, d_output, img)
        
        onnx_output = output[0]  # shape: (1, 2, H, W)
        pred_mask = np.argmax(onnx_output, axis=0)  # shape: (H, W)

        # 6. 将 mask 映射成彩色（0=黑，1=白）
        colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        colored_mask[pred_mask == 1] = [255, 255, 255]  # 白色前景
        # colored_mask = np.resize(colored_mask, (768, 1024, 3)) 

        # 7. 半透明叠加（透明度 alpha 可调）
        alpha = 0.5
        original_image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)
        output_path = os.path.join(OUTPUT_DIR, f"overlay_{img_name}")
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"✅ 已保存结果: {output_path}")

    # 释放资源
    d_input.free()
    d_output.free()

if __name__ == "__main__":
    process_images()
    print("Processing completed.")