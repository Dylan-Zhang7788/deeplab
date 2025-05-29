import os
import cv2
import numpy as np
import onnxruntime as ort
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import cv2




from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob


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
    return parser

opts = get_argparser().parse_args()
if opts.dataset.lower() == 'voc':
    opts.num_classes = 21
    decode_fn = VOCSegmentation.decode_target
elif opts.dataset.lower() == 'cityscapes':
    # opts.num_classes = 19
    opts.num_classes = 2
    decode_fn = Cityscapes.decode_target

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

# Setup dataloader
image_files = []
if os.path.isdir(opts.input):
    for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
        files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
        if len(files)>0:
            image_files.extend(files)
elif os.path.isfile(opts.input):
    image_files.append(opts.input)

# Set up model (all models are 'constructed at network.modeling)
model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
if opts.separable_conv and 'plus' in opts.model:
    network.convert_to_separable_conv(model.classifier)
utils.set_bn_momentum(model.backbone, momentum=0.01)

if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'),weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("Resume model from %s" % opts.ckpt)
    del checkpoint
else:
    print("[!] Retrain")
    model = nn.DataParallel(model)
    model.to(device)

#denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

if opts.crop_val:
    transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
else:
    transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
if opts.save_val_results_to is not None:
    os.makedirs(opts.save_val_results_to, exist_ok=True)




















# 1. 设置路径
input_dir = "test_img"                   # 输入图片文件夹
output_dir = "test_results_onnx_0529"            # 输出结果文件夹
os.makedirs(output_dir, exist_ok=True)   # 如果输出文件夹不存在则创建

# 2. 创建 ONNX 推理会话
ort_session = ort.InferenceSession("seg_model_resnet.onnx")

# 3. 遍历所有图片
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        continue

    img_path = os.path.join(input_dir, filename)
    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"❌ 无法读取图片: {img_path}")
        continue

    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0) # To tensor of NCHW
    img = img.to(device)
    with torch.no_grad():
        model = model.eval()
        
        
        model_output = model(img)
        pred = model_output.max(1)[1].cpu().numpy()[0] # HW


    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    # 4. 预处理
    img_np = Image.open(img_path).convert('RGB')

    # # 将图像转换为numpy数组并归一化到[0, 1]
    img_np = np.array(img).astype(np.float32)  # shape: (H, W, 3)

    img_cv = cv2.imread(img_path).astype(np.float32) / 255.0  # (H, W, 3), BGR
    img_np = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    

    # 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std  # 广播机制自动处理通道维度

    # 转换为 NCHW 格式并添加 batch 维度
    img_np = np.transpose(img_np, (2, 0, 1))  # (3, H, W)
    img_np = np.expand_dims(img_np, axis=0) 

    # 5. 推理
    ort_outs = ort_session.run(None, {"input": img_np.astype(np.float32)})
    onnx_output = ort_outs[0]  # (1, 2, H, W)
    pred_mask = np.argmax(onnx_output, axis=1)[0]  # (H, W)

    # 6. Resize mask 回原图大小
    mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 7. 将 mask 映射成彩色（0=黑，1=白）
    colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    colored_mask[mask_resized == 1] = [255, 255, 255]  # 白色前景

    # 8. 半透明叠加
    alpha = 0.7
    overlay = cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)

    # 9. 保存结果
    output_path = os.path.join(output_dir, f"overlay_{filename}")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"✅ 已保存结果: {output_path}")


    max_diff = np.max(np.abs(model_output.numpy()  - onnx_output))
    print(f"✅ 最大输出误差: {max_diff:.6f}")

print("🎉 全部图片处理完成！")
