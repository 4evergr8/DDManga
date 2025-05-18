import os
from PIL import Image
import imagehash
from tqdm import tqdm

# 设置目标文件夹为当前目录下的 "train"
base_folder = os.path.join(os.getcwd(), 'train')
hash_func = imagehash.phash
threshold = 8

# 遍历 train 目录下的每个子文件夹
for root, dirs, _ in os.walk(base_folder):
    for subdir in dirs:
        subfolder = os.path.join(root, subdir)
        hash_dict = {}
        visited = set()
        similar_groups = []

        print(f"\n处理子文件夹：{subfolder}")

        # 获取图片列表
        image_files = [f for f in os.listdir(subfolder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

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
                if hash_dict[filepaths[i]] - hash_dict[filepaths[j]] <= threshold:
                    group.append(filepaths[j])
                    visited.add(filepaths[j])
            if len(group) > 1:
                similar_groups.append(group)

        print(f"发现 {len(similar_groups)} 组相似图片，开始删除...")
        for group in similar_groups:
            for filepath in group[1:]:  # 每组保留第一张
                try:
                    os.remove(filepath)
                except Exception:
                    continue

        print("该子文件夹处理完成。")

print("\n所有子文件夹处理完成。")
