import os
import cv2
import numpy as np

def xdog(img, sigma=0.3, k=1.6, gamma=0.98, epsilon=0.005, phi=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    diff = g1 - gamma * g2
    diff = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon)))
    result = (diff * 255).astype(np.uint8)
    return result

def process_folder(input_folder, output_folder):
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

        xdog_img = xdog(img)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, xdog_img)
        print(f"处理完成: {filename}")

if __name__ == '__main__':
    input_dir = '.'  # 当前文件夹
    output_dir = './xdog_output'
    process_folder(input_dir, output_dir)


