#!/bin/bash

visdom -port 28333 &

sleep 5

# 运行 Python 脚本
python main.py \
  --model deeplabv3plus_mobilenet \
  --dataset cityscapes \
  --separable_conv \
  --enable_vis \
  --vis_port 28333 \
  --gpu_id 0 \
  --crop_val \
  --lr 0.01 \
  --crop_size 513 \
  --batch_size 128 \
  --output_stride 16 \
  --loss_type focal_loss \
  --data_root ./datasets/data/cityscapes_processed/