import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

# 1. 加载测试图片
img_path = "1.jpg"
original_image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
orig_h, orig_w = image_rgb.shape[:2]

# 2. 预处理：resize + normalize + HWC->CHW
input_tensor = cv2.resize(image_rgb, (1024, 1024)).astype(np.float32) / 255.0
input_tensor = input_tensor.transpose(2, 0, 1)  # HWC -> CHW
input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dim
input_tensor = np.ascontiguousarray(input_tensor)

# 3. 创建 ONNX 推理会话
ort_session = ort.InferenceSession("seg_model.onnx")

# 4. 推理
ort_outs = ort_session.run(None, {"input": input_tensor})
onnx_output = ort_outs[0]  # shape: (1, 2, H, W)
pred_mask = np.argmax(onnx_output, axis=1)[0]  # shape: (H, W)

# 5. Resize mask 回原图大小
mask_resized = cv2.resize(pred_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

# 6. 将 mask 映射成彩色（0=黑，1=白）
colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
colored_mask[mask_resized == 1] = [255, 255, 255]  # 白色前景

# 7. 半透明叠加（透明度 alpha 可调）
alpha = 0.5
overlay = cv2.addWeighted(image_rgb, 1.0, colored_mask, alpha, 0)

# 8. 显示和保存结果
plt.imshow(overlay)
plt.title("Overlay Segmentation")
plt.axis("off")
plt.show()

cv2.imwrite("overlay_result.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
print("✅ 已保存覆盖结果图 overlay_result.jpg")