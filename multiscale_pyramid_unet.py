import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    """基础卷积块：卷积 + 批标准化 + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块：增强特征学习能力"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)

class PyramidEncoderBlock(nn.Module):
    """金字塔编码器块：多尺度特征提取"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 主分支
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        
        # 多尺度分支
        self.scale_conv1 = ConvBlock(out_channels, out_channels//4, 1, 0)  # 1x1卷积
        self.scale_conv3 = ConvBlock(out_channels, out_channels//4, 3, 1)  # 3x3卷积
        self.scale_conv5 = ConvBlock(out_channels, out_channels//4, 5, 2)  # 5x5卷积
        self.scale_conv7 = ConvBlock(out_channels, out_channels//4, 7, 3)  # 7x7卷积
        
        # 特征融合
        self.fusion_conv = ConvBlock(out_channels * 2, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        # 主分支处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        
        # 多尺度特征提取
        scale1 = self.scale_conv1(x)
        scale3 = self.scale_conv3(x)
        scale5 = self.scale_conv5(x)
        scale7 = self.scale_conv7(x)
        
        # 多尺度特征融合
        multi_scale = torch.cat([scale1, scale3, scale5, scale7], dim=1)
        
        # 主特征与多尺度特征融合
        fused = torch.cat([x, multi_scale], dim=1)
        fused = self.fusion_conv(fused)
        
        # 池化用于下一层
        pooled = self.pool(fused)
        
        return fused, pooled  # 返回跳跃连接特征和下采样特征

class PyramidDecoderBlock(nn.Module):
    """金字塔解码器块：多尺度特征恢复"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 上采样
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        
        # 跳跃连接融合
        self.skip_conv = ConvBlock(skip_channels, out_channels)
        
        # 主处理分支
        self.conv1 = ConvBlock(out_channels * 2, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        
        # 多尺度细化
        self.refine_conv1 = ConvBlock(out_channels, out_channels//4, 1, 0)
        self.refine_conv3 = ConvBlock(out_channels, out_channels//4, 3, 1)
        self.refine_conv5 = ConvBlock(out_channels, out_channels//4, 5, 2)
        self.refine_conv7 = ConvBlock(out_channels, out_channels//4, 7, 3)
        
        # 最终融合
        self.final_conv = ConvBlock(out_channels * 2, out_channels)
        
    def forward(self, x, skip):
        # 上采样
        x = self.upconv(x)
        
        # 处理跳跃连接
        skip = self.skip_conv(skip)
        
        # 确保尺寸匹配（处理任意分辨率）
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # 融合跳跃连接
        x = torch.cat([x, skip], dim=1)
        
        # 主处理
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual(x)
        
        # 多尺度细化
        refine1 = self.refine_conv1(x)
        refine3 = self.refine_conv3(x)
        refine5 = self.refine_conv5(x)
        refine7 = self.refine_conv7(x)
        
        # 细化特征融合
        refined = torch.cat([refine1, refine3, refine5, refine7], dim=1)
        
        # 最终融合
        final = torch.cat([x, refined], dim=1)
        final = self.final_conv(final)
        
        return final

class AdaptivePyramidUNet(nn.Module):
    """
    自适应多尺度金字塔U-Net
    专门用于XDoG边缘图到灰度图的转换，支持任意分辨率
    """
    def __init__(self, base_channels=64, num_levels=None):
        super().__init__()
        self.base_channels = base_channels
        
        # 输入处理
        self.input_conv = ConvBlock(1, base_channels)  # XDoG输入为单通道
        
        # 动态构建编码器
        self.encoders = nn.ModuleList()
        self.encoder_channels = []
        
        # 构建多层编码器
        in_ch = base_channels
        for i in range(5):  # 最多5层，根据输入自适应
            out_ch = base_channels * (2 ** i)
            self.encoders.append(PyramidEncoderBlock(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(in_ch, in_ch * 2),
            ResidualBlock(in_ch * 2),
            ConvBlock(in_ch * 2, in_ch * 2),
            ResidualBlock(in_ch * 2)
        )
        
        # 动态构建解码器
        self.decoders = nn.ModuleList()
        for i in range(4, -1, -1):  # 反向构建解码器
            in_ch = self.encoder_channels[i] * 2 if i == 4 else self.encoder_channels[i+1]
            skip_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i]
            self.decoders.append(PyramidDecoderBlock(in_ch, skip_ch, out_ch))
        
        # 输出层
        self.output_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels//2),
            ConvBlock(base_channels//2, base_channels//4),
            nn.Conv2d(base_channels//4, 1, 1),  # 输出单通道灰度图
            nn.Sigmoid()  # 限制输出范围[0,1]
        )
        
    def _get_pyramid_levels(self, input_size):
        """根据输入尺寸动态确定金字塔层数"""
        min_size = min(input_size[2], input_size[3])  # H, W中的最小值
        max_levels = int(math.log2(min_size // 8))     # 确保最小特征图不小于8x8
        return min(max_levels, 5)  # 最多5层
        
    def forward(self, x):
        # 获取输入尺寸信息
        batch_size, channels, height, width = x.size()
        
        # 确保输入尺寸适合网络处理（padding到32的倍数）
        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            need_crop = True
        else:
            need_crop = False
        
        # 确定使用的金字塔层数
        pyramid_levels = self._get_pyramid_levels(x.size())
        
        # 输入处理
        x = self.input_conv(x)
        
        # 编码器前向传播
        skip_connections = []
        current = x
        
        for i in range(pyramid_levels):
            skip, current = self.encoders[i](current)
            skip_connections.append(skip)
        
        # 瓶颈层
        current = self.bottleneck(current)
        
        # 解码器前向传播
        for i in range(pyramid_levels):
            skip = skip_connections[pyramid_levels - 1 - i]
            current = self.decoders[i](current, skip)
        
        # 输出层
        output = self.output_conv(current)
        
        # 裁剪到原始尺寸
        if need_crop:
            output = output[:, :, :height, :width]
        
        return output

class XDoGToGrayNet(AdaptivePyramidUNet):
    """
    XDoG到灰度图转换的专用网络
    继承自AdaptivePyramidUNet，添加任务特定的优化
    """
    def __init__(self, base_channels=64):
        super().__init__(base_channels)
        
        # 边缘感知注意力模块
        self.edge_attention = nn.Sequential(
            nn.Conv2d(1, base_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels//4, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, xdog_input):
        # 计算边缘注意力权重
        edge_attention = self.edge_attention(xdog_input)
        
        # 主网络前向传播
        gray_output = super().forward(xdog_input)
        
        # 应用边缘注意力（保持XDoG边缘信息）
        # 在边缘区域更多保留原始信息，在非边缘区域更多依赖网络生成
        attended_output = gray_output * (1 - edge_attention) + xdog_input * edge_attention
        
        return attended_output

# 损失函数定义
class XDoGToGrayLoss(nn.Module):
    """专门为XDoG到灰度转换设计的损失函数"""
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # L1损失权重
        self.beta = beta    # 感知损失权重  
        self.gamma = gamma  # 边缘保持损失权重
        
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def edge_loss(self, pred, target):
        """边缘保持损失"""
        # Sobel算子检测边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
        
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2)
        
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2)
        
        return self.mse_loss(pred_edge, target_edge)
        
    def forward(self, pred, target):
        # 主要L1损失
        l1 = self.l1_loss(pred, target)
        
        # 感知损失（简化版，使用MSE）
        perceptual = self.mse_loss(pred, target)
        
        # 边缘保持损失
        edge = self.edge_loss(pred, target)
        
        total_loss = self.alpha * l1 + self.beta * perceptual + self.gamma * edge
        
        return total_loss

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = XDoGToGrayNet(base_channels=64)
    
    # 创建损失函数
    criterion = XDoGToGrayLoss()
    
    # 测试不同分辨率的输入
    test_sizes = [(256, 256), (512, 512), (768, 432), (1024, 1024)]
    
    print("=== XDoG到灰度图转换网络测试 ===")
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    model.eval()
    with torch.no_grad():
        for h, w in test_sizes:
            # 模拟XDoG输入（单通道边缘图）
            xdog_input = torch.randn(1, 1, h, w)
            
            # 前向传播
            gray_output = model(xdog_input)
            
            print(f"输入尺寸: {xdog_input.shape} -> 输出尺寸: {gray_output.shape}")
            print(f"输出值范围: [{gray_output.min():.3f}, {gray_output.max():.3f}]")
            
            # 测试损失计算
            target = torch.randn(1, 1, h, w).clamp(0, 1)  # 模拟目标灰度图
            loss = criterion(gray_output, target)
            print(f"损失值: {loss.item():.4f}")
            print("-" * 50)