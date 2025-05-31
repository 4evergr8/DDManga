import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np

# XDoG滤波
def xdog_filter(image, k=1.6, gamma=0.98, epsilon=0.1, phi=20):
    image = np.array(image).astype(np.float32) / 255.0
    blur1 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5 * k)
    dog = blur1 - gamma * blur2
    xdog = 1.0 + np.tanh(phi * (dog - epsilon))
    return (xdog * 255).clip(0, 255).astype(np.uint8)

# 数据集（含同步随机裁剪）
class XDoGDataset(Dataset):
    def __init__(self, image_paths, crop_size=512):
        self.image_paths = image_paths
        self.crop_size = crop_size
        self.to_tensor = T.ToTensor()
        self.to_grayscale = T.Grayscale()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        gray = self.to_grayscale(img)
        gray_np = np.array(gray)
        xdog_np = xdog_filter(gray_np)

        h, w = gray_np.shape
        if h < self.crop_size or w < self.crop_size:
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            gray_np = np.pad(gray_np, ((0, pad_h), (0, pad_w)), mode='reflect')
            xdog_np = np.pad(xdog_np, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = gray_np.shape

        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        gray_crop = gray_np[top:top + self.crop_size, left:left + self.crop_size]
        xdog_crop = xdog_np[top:top + self.crop_size, left:left + self.crop_size]

        gray_tensor = self.to_tensor(Image.fromarray(gray_crop))
        xdog_tensor = self.to_tensor(Image.fromarray(xdog_crop))
        return xdog_tensor, gray_tensor

# 获取图片路径
def get_image_paths(folder):
    exts = ['png', 'jpg', 'jpeg']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, '**', f'*.{ext}'), recursive=True))
    return files

# 数据加载器
def get_loader(data_dir, batch_size, crop_size=512):
    paths = get_image_paths(data_dir)
    dataset = XDoGDataset(paths, crop_size=crop_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader

# 训练函数
def train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs, save_interval_iters=100):
    model.to(device)
    best_val_loss = float('inf')
    total_iters = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for iter_idx, (xdog, target) in enumerate(train_loader, start=1):
            total_iters += 1
            xdog, target = xdog.to(device), target.to(device)
            pred = model(xdog)
            loss = loss_fn(pred, target)

            if torch.isnan(loss):
                print(f"跳过 NaN，Iter [{total_iters}]")
                print(f"  输入 xdog 是否含 NaN: {torch.isnan(xdog).any().item()}")
                print(f"  标签 target 是否含 NaN: {torch.isnan(target).any().item()}")
                print(f"  输出 pred 是否含 NaN: {torch.isnan(pred).any().item()}")
                print(f"  当前图片路径: {train_loader.dataset.image_paths[iter_idx % len(train_loader.dataset)]}")
                del pred, loss
                torch.cuda.empty_cache()
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xdog.size(0)

            if total_iters % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Iter [{total_iters}] - Loss: {loss.item():.4f}")

            if total_iters % save_interval_iters == 0:
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for val_xdog, val_target in val_loader:
                        val_xdog, val_target = val_xdog.to(device), val_target.to(device)
                        val_pred = model(val_xdog)
                        val_loss += loss_fn(val_pred, val_target).item() * val_xdog.size(0)
                val_loss /= len(val_loader.dataset)
                print(f"Iter {total_iters} - Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                    torch.save(save_model.state_dict(), "best_model.pth")
                    print(f"保存最优模型，Iter {total_iters}，Val Loss: {val_loss:.4f}")

                model.train()

            # ✅ 额外保存快照模型
            if total_iters % 2000 == 0:
                save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save(save_model.state_dict(), f"iter_{total_iters}.pth")
                print(f"📦 已保存快照模型：iter_{total_iters}.pth")

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")


# 主程序
if __name__ == "__main__":
    from multiscale_pyramid_unet import XDoGToGrayNet, XDoGToGrayLoss  # 请确保模块路径正确

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XDoGToGrayNet(base_channels=64)

    # 多GPU封装
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU")
        model = torch.nn.DataParallel(model)

    model.to(device)

    loss_fn = XDoGToGrayLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    batch_size = 2
    crop_size = 512
    epochs = 200000
    save_interval = 100

    train_loader = get_loader('/kaggle/temp/train', batch_size=batch_size, crop_size=crop_size)
    val_loader = get_loader('/kaggle/temp/val', batch_size=batch_size, crop_size=crop_size)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs, save_interval)
