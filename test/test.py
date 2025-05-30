import os
import cv2
import numpy as np

def xdog(img, sigma=0.3, k=1.6, gamma=0.98, epsilon=0.01, phi=10):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    diff = g1 - gamma * g2
    diff = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon)))
    result = (diff * 255).astype(np.uint8)
    return result

def process_folder(input_folder, output_folder, phi_values):
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

        for phi in phi_values:
            xdog_img = xdog(img, phi=phi)
            base_name, ext = os.path.splitext(filename)
            out_path = os.path.join(output_folder, f"{base_name}_phi{phi}{ext}")
            cv2.imwrite(out_path, xdog_img)
            print(f"处理完成: {filename} phi={phi}")

if __name__ == '__main__':
    input_dir = '.'  # 当前文件夹
    output_dir = './xdog_output'
    phi_list = [5, 10, 15, 20]  # 你需要输出的不同phi值
    process_folder(input_dir, output_dir, phi_list)
