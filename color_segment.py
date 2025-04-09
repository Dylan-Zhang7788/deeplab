import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def segment_grass(image, k=3):
    """
    使用颜色聚类方法分割图像中的特定绿色草地
    
    参数:
        image: 输入图像(numpy数组)
        k: 聚类中心数量(默认为3)
    
    返回:
        segmented: 分割后的图像(草地为白色，其他为黑色)
        visualization: 可视化结果
    """
    # 转换为RGB颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 将图像转换为2D数组(像素列表)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # 定义K-means聚类标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # 执行K-means聚类
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # 将中心值转换为8位整型
    centers = np.uint8(centers)
    
    # 获取每个像素的标签
    labels = labels.flatten()
    
    # 将目标绿色RGB(115, 122, 55)转换为HSV
    target_green_rgb = np.uint8([[[115, 122, 55]]])
    target_green_hsv = cv2.cvtColor(target_green_rgb, cv2.COLOR_RGB2HSV)[0][0]
    
    # 计算HSV颜色范围
    # H范围: ±15, S和V范围: ±40 (可根据需要调整)
    hue_range = 15
    sat_val_range = 40
    
    lower_green = np.array([
        max(0, target_green_hsv[0] - hue_range),
        max(0, target_green_hsv[1] - sat_val_range),
        max(0, target_green_hsv[2] - sat_val_range)
    ])
    
    upper_green = np.array([
        min(179, target_green_hsv[0] + hue_range),
        min(255, target_green_hsv[1] + sat_val_range),
        min(255, target_green_hsv[2] + sat_val_range)
    ])
    
    print(f"使用的HSV颜色范围 - 下限: {lower_green}, 上限: {upper_green}")
    
    # 将聚类中心转换为HSV
    hsv_centers = cv2.cvtColor(np.array([centers]), cv2.COLOR_RGB2HSV)[0]
    
    # 找出哪些聚类中心在目标绿色范围内
    grass_clusters = []
    for i, hsv_center in enumerate(hsv_centers):
        if (hsv_center[0] >= lower_green[0] and hsv_center[0] <= upper_green[0] and
            hsv_center[1] >= lower_green[1] and hsv_center[1] <= upper_green[1] and
            hsv_center[2] >= lower_green[2] and hsv_center[2] <= upper_green[2]):
            grass_clusters.append(i)
    
    # 如果没有找到绿色聚类，尝试找到最接近目标绿色的聚类
    if not grass_clusters:
        distances = [np.linalg.norm(hsv_center - target_green_hsv) for hsv_center in hsv_centers]
        closest_cluster = np.argmin(distances)
        grass_clusters.append(closest_cluster)
        print("警告: 没有找到完全匹配的绿色聚类，使用最接近的聚类")
    
    # 创建草地掩码
    mask = np.zeros_like(labels)
    for cluster in grass_clusters:
        mask[labels == cluster] = 1
    
    # 将掩码调整为图像形状
    mask = mask.reshape((h, w))
    mask = mask.astype(np.uint8)
    
    # 使用形态学操作改善掩码
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 创建分割结果(草地为白色，其他为黑色)
    segmented = np.zeros_like(image_rgb)
    segmented[mask == 1] = [255, 255, 255]
    
    # 创建可视化结果
    visualization = image_rgb.copy()
    visualization[mask == 1] = [115, 122, 55]  # 使用指定的绿色值标记草地
    
    return segmented, visualization

def process_folder(input_folder, output_folder, k=3):
    """
    处理文件夹中的所有图像
    
    参数:
        input_folder: 输入图像文件夹路径
        output_folder: 输出结果文件夹路径
        k: 聚类中心数量
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取输入文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"警告: 在文件夹 {input_folder} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像需要处理")
    
    for i, filename in enumerate(image_files, 1):
        # 读取图像
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        
        if image is None:
            print(f"无法读取图像: {filename}, 跳过")
            continue
        
        # 处理图像
        segmented, visualization = segment_grass(image, k=k)
        
        # 准备输出文件名
        base_name = os.path.splitext(filename)[0]
        
        # 保存分割结果
        segmented_path = os.path.join(output_folder, f"{base_name}_segmented.jpg")
        cv2.imwrite(segmented_path, cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))
        
        # 保存可视化结果
        vis_path = os.path.join(output_folder, f"{base_name}_visualization.jpg")
        cv2.imwrite(vis_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        print(f"处理进度: {i}/{len(image_files)} - {filename}")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "test_img"  # 输入图像文件夹
    output_folder = "test_result"  # 输出结果文件夹
    
    # 处理文件夹中的所有图像
    process_folder(input_folder, output_folder, k=4)
    
    print("处理完成! 结果已保存到", output_folder)