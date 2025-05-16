import os
from tqdm import tqdm

def find_and_save_images(data_path, output_name):
    all_img_path = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_img_path.append(os.path.join(root, file))

    with open(output_name, 'w') as f:
        for path in tqdm(all_img_path, desc=f"处理 {data_path}"):
            f.write(path + '\n')

def main():
    for split in ['train', 'val']:
        data_path = split
        output_name = os.path.join(data_path, f'{split}.txt')
        print(f'生成 {output_name} 中...')
        find_and_save_images(data_path, output_name)
    print('全部完成。')

if __name__ == '__main__':
    main()
