import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryRenderingModule(nn.Module):
    def __init__(self):
        super(BoundaryRenderingModule, self).__init__()

    def forward(self, x):
        """
        输入：
            x: Tensor，形状为 (B, C, D, H, W)，B 是 batch size，C 是通道数，D 是深度，H 和 W 是图像的高度和宽度
        输出：
            boundary: Tensor，形状为 (B, C, D, H, W)，提取出的边界图，C 表示每个通道的边界图
        """
        B, C, D, H, W = x.shape
        boundaries = []

        # 定义一个 3x3x3 的结构元素
        structure = torch.ones((1, 1, 3, 3, 3), device=x.device, dtype=torch.float32)

        for i in range(C):
            # 提取当前通道
            channel = x[:, i:i+1, :, :, :]  # 形状为 (B, 1, D, H, W)

            # 归一化
            channel = (channel - channel.min()) / (channel.max() - channel.min())

            # 二值化
            channel = channel > 0.5  # 将灰度图像映射为二进制图像

            # 膨胀
            dilated = F.conv3d(channel.float(), structure, padding=1)
            dilated = dilated >= 1.0  # 膨胀结果二值化

            # 腐蚀
            # eroded = F.conv3d(channel.float(), structure.flip(2, 3, 4), padding=1)
            # eroded = eroded >= 1.0  # 腐蚀结果二值化

            # 计算边界: 膨胀 - 原图
            boundary = dilated.float() - channel.float()

            boundaries.append(boundary)

        # 将所有通道的边界图组合成一个张量
        boundary = torch.cat(boundaries, dim=1)  # 形状为 (B, C, D, H, W)

        return boundary

