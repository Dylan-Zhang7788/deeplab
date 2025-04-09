import os
import glob

# 指定要查找文件的目录（默认为当前目录）
directory = 'datasets/data/forest/train/GT_color/'  # 可以替换为你的目标目录路径

# 查找所有以 _Clipped.png 结尾的文件
files = glob.glob(os.path.join(directory, '*_Clipped.png'))

# 遍历文件并重命名
for file_path in files:
    # 获取文件所在目录和文件名
    dir_name, file_name = os.path.split(file_path)
    
    # 生成新的文件名
    new_file_name = file_name.replace('_Clipped.png', '_mask.png')
    
    # 生成新的文件路径
    new_file_path = os.path.join(dir_name, new_file_name)
    
    # 重命名文件（覆盖式重命名）
    os.rename(file_path, new_file_path)
    print(f'Renamed: {file_path} -> {new_file_path}')

print("All files renamed successfully!")