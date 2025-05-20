import os
import shutil
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# ===== 参数配置 =====
GRAYSCALE_DIFF_THRESHOLD = 1
SATURATION_THRESHOLD = 30
HASH_DIFF_THRESHOLD = 5

IMG_SUFFIXES = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ===== 路径配置 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'train')
delete_dir = os.path.join(train_dir, 'delete')
os.makedirs(delete_dir, exist_ok=True)


def move_to_delete(src_path):
    rel_path = os.path.relpath(src_path, train_dir)
    dst_path = os.path.join(delete_dir, rel_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.move(src_path, dst_path)
    print(f"已移动到：{dst_path}")


# ===== 第一部分：删除黑白图像 =====
print("===== 第一部分：删除黑白图像 =====")
for root, dirs, _ in os.walk(train_dir):
    if delete_dir in root:
        continue  # 跳过已移动的内容
    for subdir in dirs:
        folder = os.path.join(root, subdir)
        for file in tqdm(os.listdir(folder)):
            if not file.lower().endswith(IMG_SUFFIXES):
                continue
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_np = np.array(img)
                    diff_rg = np.abs(img_np[:, :, 0] - img_np[:, :, 1])
                    diff_rb = np.abs(img_np[:, :, 0] - img_np[:, :, 2])
                    diff_gb = np.abs(img_np[:, :, 1] - img_np[:, :, 2])
                    mean_diff = (np.mean(diff_rg) + np.mean(diff_rb) + np.mean(diff_gb)) / 3
                    if mean_diff < GRAYSCALE_DIFF_THRESHOLD:
                        img.close()
                        move_to_delete(path)
            except Exception as e:
                print(f"处理失败（黑白判断）：{path}，错误：{e}")

# ===== 第二部分：删除色彩单调图像 =====
print("\n===== 第二部分：删除色彩单调图像（按色彩饱和度） =====")
for root, dirs, _ in os.walk(train_dir):
    if delete_dir in root:
        continue
    for subdir in dirs:
        folder = os.path.join(root, subdir)
        for file in tqdm(os.listdir(folder)):
            if not file.lower().endswith(IMG_SUFFIXES):
                continue
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_hsv = img.convert('HSV')
                    hsv = np.array(img_hsv)
                    saturation = hsv[..., 1]
                    mean_sat = saturation.mean()

                    print(f"{path} - 平均饱和度: {mean_sat:.2f}")

                    mean_sat = saturation.mean()
                    std_sat = saturation.std()

                    # 同时判断：整体饱和度高、但变化小 → 很可能是“纯色页”
                    if mean_sat > 60 and std_sat < 10:
                        move_to_delete(path)

            except Exception as e:
                print(f"处理失败（色彩饱和度判断）：{path}，错误：{e}")


# ===== 第三部分：使用哈希去重保留唯一图像 =====
print("\n===== 第三部分：使用哈希去重图像 =====")
hash_func = imagehash.phash

for root, dirs, _ in os.walk(train_dir):
    if delete_dir in root:
        continue
    for subdir in dirs:
        subfolder = os.path.join(root, subdir)
        hash_dict = {}
        visited = set()
        similar_groups = []

        print(f"\n处理子文件夹：{subfolder}")
        image_files = [f for f in os.listdir(subfolder) if f.lower().endswith(IMG_SUFFIXES)]

        print("计算图片哈希中...")
        for filename in tqdm(image_files):
            filepath = os.path.join(subfolder, filename)
            try:
                with Image.open(filepath) as img:
                    h = hash_func(img)
                    hash_dict[filepath] = h
            except Exception:
                continue

        filepaths = list(hash_dict.keys())
        print("查找相似图片中...")
        for i in tqdm(range(len(filepaths))):
            if filepaths[i] in visited:
                continue
            group = [filepaths[i]]
            visited.add(filepaths[i])
            for j in range(i + 1, len(filepaths)):
                if filepaths[j] in visited:
                    continue
                if hash_dict[filepaths[i]] - hash_dict[filepaths[j]] <= HASH_DIFF_THRESHOLD:
                    group.append(filepaths[j])
                    visited.add(filepaths[j])
            if len(group) > 1:
                similar_groups.append(group)

        print(f"发现 {len(similar_groups)} 组相似图片，开始移动多余文件...")
        for group in similar_groups:
            for filepath in group[1:]:  # 每组保留第一张
                try:
                    move_to_delete(filepath)
                except Exception:
                    continue

        print("该子文件夹处理完成。")

print("\n所有处理完成。")
