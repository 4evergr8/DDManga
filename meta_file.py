import os
import zipfile
import cv2
from PIL import Image


def unzip_all_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.zip', '.pth')):
                zip_path = os.path.join(root, file)
                print(f'解压：{zip_path}')
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root, pwd=b'666')
                    os.remove(zip_path)
                    print(f'已删除压缩包：{zip_path}')
                except Exception as e:
                    print(f'⚠️ 解压失败：{zip_path}，错误：{e}')

def is_image_valid(image_path):
    try:
        with Image.open(image_path) as img:
            _ = img.convert("RGB")
    except Exception:
        return False
    return True



def delete_invalid_images(data_path):
    deleted_count = 0
    total_checked = 0

    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                abs_path = os.path.abspath(os.path.join(root, file))
                total_checked += 1
                if not is_image_valid(abs_path):
                    try:
                        os.remove(abs_path)
                        print(f'🗑️ 已删除无效图像：{abs_path}')
                        deleted_count += 1
                    except Exception as e:
                        print(f'⚠️ 删除失败：{abs_path}，错误：{e}')

    print(f'✅ 检查完成：共检查 {total_checked} 张图像，删除 {deleted_count} 张无效图像')

def main():
    val_path = 'val'
    unzip_all_in_folder(val_path)
    print(f'检查并清理 {val_path} 中的无效图像...')
    delete_invalid_images(val_path)

    train_root = 'train'
    print('解压 train 中的所有 zip 包...')
    unzip_all_in_folder(train_root)
    print(f'检查并清理 {train_root} 中的无效图像...')
    delete_invalid_images(train_root)

    print('🎉 全部完成！')

if __name__ == '__main__':
    main()
