import os
import zipfile
import cv2
from tqdm import tqdm

def unzip_all_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.zip', '.pth')):
                zip_path = os.path.join(root, file)
                print(f'解压：{zip_path}')
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root, pwd=b'666')
                    os.remove(zip_path)  # ✅ 解压完成后删除压缩包
                    print(f'已删除：{zip_path}')
                except Exception as e:
                    print(f'⚠️ 解压失败：{zip_path}，错误：{e}')

def is_image_valid_cv2(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False  # 尺寸太小，可能是无效图像
        return True
    except Exception as e:
        print(f'⚠️ 读取失败：{image_path}，错误：{e}')
        return False

def find_and_save_images(data_path, output_txt_path):
    all_img_path = []
    skipped_count = 0

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                abs_path = os.path.abspath(os.path.join(root, file))
                if is_image_valid_cv2(abs_path):  # ✅ 用 OpenCV 检测图像
                    all_img_path.append(abs_path)
                else:
                    print(f'⚠️ 无效图像：{abs_path}')
                    skipped_count += 1

    all_img_path.sort()

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for path in tqdm(all_img_path, desc=f"生成 {output_txt_path}"):
            f.write(path + '\n')

    print(f'✅ 共记录有效图像：{len(all_img_path)} 张，跳过损坏图像：{skipped_count} 张')

def main():
    val_path = 'val'
    unzip_all_in_folder(val_path)
    val_txt = os.path.join(val_path, 'val.txt')
    print(f'生成 {val_txt} 中...')
    find_and_save_images(val_path, val_txt)

    train_root = 'train'
    print('解压 train 中的所有 zip 包...')
    unzip_all_in_folder(train_root)

    train_txt = os.path.join(train_root, 'train.txt')
    print(f'生成 {train_txt} 中...')
    find_and_save_images(train_root, train_txt)

    print('🎉 全部完成！')

if __name__ == '__main__':
    main()
