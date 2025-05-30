import os
import cv2
import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY


class GreyRestorationPipeline:
    def __init__(self, model_path, input_size=256):
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载GreyModel
        self.model = ARCH_REGISTRY.get('XDoG2GrayNet')().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['params'], strict=True)
        self.model.eval()

    @torch.no_grad()
    def process(self, img):
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.input_size, self.input_size))

        # 先计算XDoG图
        xdog_img = xdog(img_resized)  # 已经是灰度图uint8

        # 灰度图
        gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # 拼接成2通道输入，归一化到0~1
        input_tensor = np.stack([xdog_img, gray_img], axis=0).astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)  # (1,1,H,W)
        output_resized = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        out_img = output_resized.squeeze().cpu().numpy()
        out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
        return out_img


def xdog(img, sigma=0.3, k=1.6, gamma=0.98, epsilon=0.005, phi=20):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(img, (0, 0), sigma)
    g2 = cv2.GaussianBlur(img, (0, 0), sigma * k)
    diff = g1 - gamma * g2
    diff = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon)))
    result = (diff * 255).astype(np.uint8)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='pretrain', help='权重文件目录')
    parser.add_argument('--input', type=str, default='test', help='输入图片目录')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--input_size', type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_files = [f for f in os.listdir(args.model_dir) if f.endswith('.pth')]
    assert model_files, "未找到任何.pth权重文件"

    image_list = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.png'))]
    assert image_list, "输入文件夹中没有图片"

    for model_file in model_files:
        model_path = os.path.join(args.model_dir, model_file)
        print(f"\n加载模型: {model_file}")
        restorer = GreyRestorationPipeline(model_path, input_size=args.input_size)

        for file_name in tqdm(image_list, desc=f"推理中（{model_file}）"):
            img_path = os.path.join(args.input, file_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue
            result = restorer.process(img)
            save_name = os.path.splitext(file_name)[0] + f"_{os.path.splitext(model_file)[0]}.png"
            cv2.imwrite(os.path.join(args.output, save_name), result)


if __name__ == '__main__':
    main()
