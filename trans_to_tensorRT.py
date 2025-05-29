import torch
import tensorrt as trt
import numpy as np
import cv2
import os
from torch2trt import torch2trt
import torch.nn as nn
import argparse
import network
from datasets import Cityscapes
from time import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--engine_path", type=str, default='model.engine')
    return parser

# 2. 转换为TensorRT
def convert_to_tensorrt(model, input_shape=(1, 3, 512, 512)):
    # 创建示例输入
    x = torch.randn(input_shape).cuda()
    
    # 转换模型 (fp16模式可提升速度)
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=True,
        max_workspace_size=1 << 30,  # 1GB
        log_level=trt.Logger.INFO
    )
    return model_trt

# 3. 保存/加载TensorRT引擎
def save_engine(engine, path):
    with open(path, 'wb') as f:
        f.write(engine.serialize())

def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

# 4. 推理测试
def inference_test(model, img_path):
    # 预处理图像
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda()
    img_tensor = img_tensor / 255.0  # 归一化
    
    # PyTorch推理
    torch.cuda.synchronize()
    t0 = time()
    with torch.no_grad():
        output_pytorch = model(img_tensor)
    torch.cuda.synchronize()
    print(f"PyTorch推理时间: {time()-t0:.4f}s")
    
    # TensorRT推理 (如果已转换)
    if hasattr(model, 'engine'):
        torch.cuda.synchronize()
        t0 = time()
        output_trt = model(img_tensor)
        torch.cuda.synchronize()
        print(f"TensorRT推理时间: {time()-t0:.4f}s")
        
        # 验证输出一致性
        diff = torch.max(torch.abs(output_pytorch - output_trt))
        print(f"最大输出差异: {diff.item():.6f}")

if __name__ == "__main__":
    # 配置参数
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    
    # 初始化TensorRT
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    
    # 1. 加载PyTorch模型
    print("加载PyTorch模型...")
    opts.num_classes = 2
    decode_fn = Cityscapes.decode_target

    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'),weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    
    # 2. 转换为TensorRT
    print("\n转换为TensorRT引擎...")
    model_trt = convert_to_tensorrt(model)
    save_engine(model_trt.engine, opts.engine_path)
    
    # 3. 加载TensorRT引擎 (可选)
    print("\n加载TensorRT引擎...")
    engine = load_engine(trt_runtime, opts.engine_path)
    
    # 4. 性能测试
    print("\n运行推理测试...")
    inference_test(model_trt, opts.input)