import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def get_rgb(event):
    # 获取点击位置的坐标
    x, y = event.x, event.y
    # 获取该位置的 RGB 值
    rgb = img.getpixel((x, y))
    print(f"位置: ({x}, {y}) - RGB值: {rgb}")

# 创建主窗口
root = tk.Tk()
root.title("点击获取RGB值")

# 打开文件对话框选择图像
file_path = filedialog.askopenfilename()
if not file_path:
    print("未选择图像文件。")
    exit()

# 打开图像并转换为适合显示的格式
img = Image.open(file_path).convert('RGB')
tk_img = ImageTk.PhotoImage(img)

# 创建画布并显示图像
canvas = tk.Canvas(root, width=img.width, height=img.height)
canvas.pack()
canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)

# 绑定鼠标左键点击事件
canvas.bind("<Button-1>", get_rgb)

# 运行主循环
root.mainloop()
