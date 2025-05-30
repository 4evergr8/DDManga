import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class XDoG2GrayNet(nn.Module):
    """支持任意分辨率的XDoG到灰度图网络，基于UNet结构，无FC层"""

    def __init__(self, in_ch=2, out_ch=1, base_feat=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat, base_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_feat, base_feat*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat*2, base_feat*2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_feat*2, base_feat*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat*4, base_feat*4, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_feat*4, base_feat*8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat*8, base_feat*8, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose2d(base_feat*8, base_feat*4, 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_feat*8, base_feat*4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat*4, base_feat*4, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(base_feat*4, base_feat*2, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_feat*4, base_feat*2, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat*2, base_feat*2, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(base_feat*2, base_feat, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_feat*2, base_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_feat, base_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(base_feat, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # b,c,h,w
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return out
