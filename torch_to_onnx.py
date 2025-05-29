import torch
import torch.nn as nn
import numpy as np
import onnx
import os
import onnxruntime as ort
import argparse
import network
from datasets import Cityscapes
from torchvision import transforms as T
from PIL import Image


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

opts = get_argparser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

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
    model.to(device)
    print("Resume model from %s" % opts.ckpt)
    del checkpoint
model.eval()

# 2. 准备输入数据（假设模型输入是 1×3×512×512）
transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
img = Image.open("test_img/2.jpg").convert('RGB')
img = transform(img).unsqueeze(0) # To tensor of NCHW
img = img.to('cpu')
# dummy_input = torch.randn(1, 3, 768, 1024, requires_grad=False).float() 
dummy_input = img

# 3. 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "seg_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("✅ ONNX模型导出成功")

# 4. 加载 ONNX 模型并推理
ort_session = ort.InferenceSession("seg_model.onnx")

# 注意要转成 numpy 并保持 dtype 为 float32
onnx_input = dummy_input.numpy()
ort_outs = ort_session.run(None, {"input": onnx_input})
onnx_output = ort_outs[0]

# 5. 用 PyTorch 推理原始模型
with torch.no_grad():
    torch_output = model(dummy_input).numpy()

# 6. 误差比较
max_diff = np.max(np.abs(torch_output - onnx_output))
print(f"✅ 最大输出误差: {max_diff:.6f}")

if max_diff < 1e-4:
    print("✅ PyTorch 与 ONNX 模型一致 ✔️")
else:
    print("⚠️ 存在较大误差，请检查模型中是否包含不支持的操作 ❗")
