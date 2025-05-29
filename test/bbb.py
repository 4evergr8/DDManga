import cv2
import numpy as np
import os
from glob import glob

def xdog_filter(img, sigma=0.5, k=4.5, p=21, epsilon=-0.1, phi=10.0):
    img = img.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    diff = g1 - g2
    diff = diff / np.max(np.abs(diff))
    xdog = 1.0 + np.tanh(phi * (diff + epsilon))
    return (xdog * 255).clip(0, 255).astype(np.uint8)

def process_and_save(input_path, output_path, alpha=0.6):
    img = cv2.imread(input_path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xdog = xdog_filter(gray)

    mixed = cv2.addWeighted(xdog, alpha, gray, 1 - alpha, 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, mixed)

def process_all_images(input_folder='.', output_folder='xdog_mix_output', alpha=0.6):
    exts = ['*.jpg', '*.png', '*.jpeg']
    paths = []
    for ext in exts:
        paths += glob(os.path.join(input_folder, ext))

    for path in paths:
        filename = os.path.basename(path)
        out_path = os.path.join(output_folder, filename)
        process_and_save(path, out_path, alpha)

if __name__ == "__main__":
    process_all_images()
