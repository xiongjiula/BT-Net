import torch
import torch.nn.functional as F


# 3D形态学膨胀与布尔差集损失计算
def binary_dilation_pytorch(x, structure):
    # 使用3D卷积实现膨胀操作
    dilation_kernel = torch.ones_like(structure, device=x.device, dtype=torch.float32)
    dilated = F.conv3d(x.float(), dilation_kernel, padding=1)
    dilated = dilated >= 1.0  # 膨胀结果二值化
    return dilated


def morphological_boundary_loss(pred, target):
    # 先将pred和target转换为0和1的二进制图像
    pred_bin = pred > 0.5  # 转换为二进制
    target_bin = target > 0.5

    # 定义3x3x3的结构元素
    structure = torch.ones((1, 1, 3, 3, 3), device=pred.device, dtype=torch.float32)

    # 使用PyTorch实现膨胀操作
    pred_dilated = binary_dilation_pytorch(pred_bin, structure)
    target_dilated = binary_dilation_pytorch(target_bin, structure)

    # 计算膨胀后的图像与原图之间的差异
    pred_boundary = pred_dilated.float() - pred_bin.float()
    target_boundary = target_dilated.float() - target_bin.float()

    # 计算边界损失
    boundary_loss = F.mse_loss(pred_boundary, target_boundary)

    return boundary_loss

