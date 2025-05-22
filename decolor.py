import os
import shutil
from PIL import Image
from tqdm import tqdm

# ===== 参数配置 =====
IMG_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ===== 路径配置 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'train')


# ===== 黑白转换并覆盖 =====
print("===== 将图像转换为灰度（黑白）图并覆盖保存 =====")
for root, dirs, _ in os.walk(train_dir):
    for subdir in dirs:
        folder = os.path.join(root, subdir)
        for file in os.listdir(folder):
            if not file.lower().endswith(IMG_SUFFIXES):
                continue
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    img_gray = img.convert('L')  # 转换为灰度图
                    img_gray.save(path)          # 覆盖保存
                    print(f"已转换：{path}")
            except Exception as e:
                print(f"处理失败（灰度转换）：{path}，错误：{e}")
