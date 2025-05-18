import os
import shutil
from PIL import Image, ImageFile, ImageChops
import warnings



def is_color_image(img, threshold=10):
    # 强制转成RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    r, g, b = img.split()
    diff_rg = ImageChops.difference(r, g)
    diff_rb = ImageChops.difference(r, b)
    diff_gb = ImageChops.difference(g, b)

    # 计算差异的平均值
    def mean_diff(diff_img):
        hist = diff_img.histogram()
        pixels = sum(hist)
        total = sum(i * hist[i] for i in range(256))
        return total / pixels if pixels else 0

    mean_rg = mean_diff(diff_rg)
    mean_rb = mean_diff(diff_rb)
    mean_gb = mean_diff(diff_gb)

    # 如果任意两个通道平均差异超过阈值，则认为是彩色图
    if mean_rg > threshold or mean_rb > threshold or mean_gb > threshold:
        return True
    else:
        return False



def unique_filename(dst_dir, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_name = filename
    while os.path.exists(os.path.join(dst_dir, new_name)):
        new_name = f"{base}_{counter}{ext}"
        counter += 1
    return new_name

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'muk-monsieur')
    a_dir = os.path.join(base_dir, 'a')
    b_dir = os.path.join(base_dir, 'b')

    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    for folder, _, files in os.walk(data_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    if img.width < 256 or img.height < 256:
                        continue

                    if is_color_image(img):
                        dst_file = unique_filename(b_dir, file)
                        img.close()
                        shutil.move(path, os.path.join(b_dir, dst_file))
                    else:
                        dst_file = unique_filename(a_dir, file)
                        img.close()
                        shutil.move(path, os.path.join(a_dir, dst_file))

            except Exception as e:
                print(f"处理失败: {path}，错误：{e}")


if __name__ == '__main__':
    main()
