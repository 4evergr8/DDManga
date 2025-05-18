import os
import random
from PIL import Image
import imagehash

# 获取脚本所在目录
folder = os.path.dirname(os.path.abspath(__file__))

# 参数
size_tolerance = 1024*1024  # 文件大小相差在1KB以内视为可能相似
hash_threshold = 5     # 感知哈希的最大海明距离

# 收集图片路径和大小
image_info = []
for f in os.listdir(folder):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        path = os.path.join(folder, f)
        size = os.path.getsize(path)
        image_info.append((path, size))

# 按文件大小排序
image_info.sort(key=lambda x: x[1])

# 分组：文件大小相近的分一组
groups = []
current_group = [image_info[0]]
for i in range(1, len(image_info)):
    if abs(image_info[i][1] - current_group[-1][1]) <= size_tolerance:
        current_group.append(image_info[i])
    else:
        groups.append(current_group)
        current_group = [image_info[i]]
if current_group:
    groups.append(current_group)

# 对每组内进行哈希比较，聚类并删除重复图
for group in groups:
    if len(group) == 1:
        continue

    hash_list = []
    for path, _ in group:
        try:
            img = Image.open(path).convert('RGB')
            h = imagehash.phash(img)
            hash_list.append((path, h))
        except:
            continue

    used = set()
    for i in range(len(hash_list)):
        if i in used:
            continue
        cluster = [hash_list[i][0]]
        used.add(i)
        for j in range(i + 1, len(hash_list)):
            if j in used:
                continue
            if hash_list[i][1] - hash_list[j][1] <= hash_threshold:
                cluster.append(hash_list[j][0])
                used.add(j)
        if len(cluster) > 1:
            keep = random.choice(cluster)
            for path in cluster:
                if path != keep:
                    os.remove(path)
                    print(f"已删除：{path}")
