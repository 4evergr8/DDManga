import os
import cv2
import numpy as np

def xdog(img, sigma=0.3, k=1.6, gamma=0.98, epsilon=0.01, phi=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    diff = g1 - gamma * g2
    diff = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon)))
    result = (diff * 255).astype(np.uint8)
    return result

def process_folder(input_folder, output_folder, epsilons=[0.005, 0.01, 0.02, 0.05]):
    os.makedirs(output_folder, exist_ok=True)
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']

    for filename in os.listdir(input_folder):
        if os.path.splitext(filename)[-1].lower() not in image_exts:
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的文件: {filename}")
            continue

        for epsilon in epsilons:
            xdog_img = xdog(img, epsilon=epsilon)
            base_name, ext = os.path.splitext(filename)
            out_filename = f"{base_name}_eps{epsilon}{ext}"
            out_path = os.path.join(output_folder, out_filename)
            cv2.imwrite(out_path, xdog_img)
            print(f"处理完成: {out_filename}")

if __name__ == '__main__':
    input_dir = '.'  # 当前文件夹
    output_dir = './xdog_output'
    epsilons = [0.005, 0.01, 0.02, 0.05]
    process_folder(input_dir, output_dir, epsilons)
