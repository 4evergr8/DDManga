import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import numpy as np

# XDoG滤波
def xdog_filter(image, k=1.6, gamma=0.98, epsilon=0.1, phi=200):
    image = np.array(image).astype(np.float32) / 255.0
    blur1 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigmaX=0.5 * k)
    dog = blur1 - gamma * blur2
    xdog = 1.0 + np.tanh(phi * (dog - epsilon))
    return (xdog * 255).clip(0, 255).astype(np.uint8)

# 数据集
class XDoGDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
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
        xdog = Image.fromarray(xdog_np)
        xdog_tensor = self.to_tensor(xdog)
        gray_tensor = self.to_tensor(gray)
        return xdog_tensor, gray_tensor

# 加载图片路径
def get_image_paths(folder):
    exts = ['png', 'jpg', 'jpeg']
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(folder, '**', f'*.{ext}'), recursive=True))
    return files
def pad_collate_fn(batch):
    xdogs, targets = zip(*batch)
    max_h = max([img.shape[1] for img in xdogs])
    max_w = max([img.shape[2] for img in xdogs])

    padded_xdogs = []
    padded_targets = []
    for xdog, target in zip(xdogs, targets):
        pad_h = max_h - xdog.shape[1]
        pad_w = max_w - xdog.shape[2]
        xdog_padded = torch.nn.functional.pad(xdog, (0, pad_w, 0, pad_h), value=1.0)
        target_padded = torch.nn.functional.pad(target, (0, pad_w, 0, pad_h), value=1.0)
        padded_xdogs.append(xdog_padded)
        padded_targets.append(target_padded)
    return torch.stack(padded_xdogs), torch.stack(padded_targets)

# 加载数据
def get_loader(data_dir, batch_size):
    paths = get_image_paths(data_dir)
    dataset = XDoGDataset(paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate_fn)
    return loader


# 训练函数
def train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs, save_interval=5):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xdog, target in train_loader:
            xdog, target = xdog.to(device), target.to(device)
            pred = model(xdog)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xdog.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xdog, target in val_loader:
                xdog, target = xdog.to(device), target.to(device)
                pred = model(xdog)
                loss = loss_fn(pred, target)
                val_loss += loss.item() * xdog.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"保存最优模型，Val Loss: {val_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            save_path = f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"保存周期模型：{save_path}")

# 主程序
if __name__ == "__main__":
    from multiscale_pyramid_unet import XDoGToGrayNet, XDoGToGrayLoss  # 请确保模块路径正确

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XDoGToGrayNet(base_channels=64)
    loss_fn = XDoGToGrayLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 8
    epochs = 20
    save_interval = 5

    train_loader = get_loader('/content/gdrive/MyDrive/DDColor/train', batch_size=batch_size)
    val_loader = get_loader('/content/gdrive/MyDrive/DDColor/val', batch_size=batch_size)

    train(model, loss_fn, optimizer, train_loader, val_loader, device, epochs, save_interval)
