import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2

# XDoG滤波函数，和训练时保持一致
def xdog_filter(image, k=1.6, gamma=0.98, epsilon=0.1, phi=20):
    image = np.array(image).astype(np.float32) / 255.0
    blur1 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5 * k)
    dog = blur1 - gamma * blur2
    xdog = 1.0 + np.tanh(phi * (dog - epsilon))
    return (xdog * 255).clip(0, 255).astype(np.uint8)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from multiscale_pyramid_unet import XDoGToGrayNet  # 确保路径正确
model = XDoGToGrayNet(base_channels=64)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# 输入图片路径
input_path = "test_image.jpg"
output_path = "result_gray.png"

# 读取图片并转灰度
img = Image.open(input_path).convert("L")
img_np = np.array(img)

# 计算 XDoG 输入
xdog_np = xdog_filter(img_np)

# 转为tensor，保持训练时通道和归一化方式一致
to_tensor = T.ToTensor()
xdog_tensor = to_tensor(Image.fromarray(xdog_np)).unsqueeze(0).to(device)  # 1x1xHxW

# 模型推理
with torch.no_grad():
    output_tensor = model(xdog_tensor)

# 处理输出张量，转为灰度图像保存
output_img = output_tensor.squeeze().cpu().numpy()
output_img = np.clip(output_img * 255, 0, 255).astype(np.uint8)
Image.fromarray(output_img).save(output_path)

print(f"推理完成，结果已保存至 {output_path}")
