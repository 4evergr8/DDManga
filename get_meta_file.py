import os
import zipfile
from tqdm import tqdm

def unzip_all_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.zip', '.pth')):
                zip_path = os.path.join(root, file)
                print(f'解压：{zip_path}')
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root, pwd=b'666')

def find_and_save_images(data_path, output_txt_path):
    all_img_path = []

    for root, _, files in os.walk(data_path):
        for ext in ['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG', 'webp']:
            for file in files:
                if file.lower().endswith(f'.{ext}'):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    all_img_path.append(abs_path)

    all_img_path.sort()

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for path in tqdm(all_img_path, desc=f"处理 {data_path}"):
            f.write(path + '\n')

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

    print('全部完成。')

if __name__ == '__main__':
    main()
