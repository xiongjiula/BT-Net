""" Full assembly of the parts to form the complete network """
from .multiScaleTransformer_parts import MultiScaleTransformer
from .boundary_parts import BoundaryRenderingModule
from .unet_parts import *


class BoundaryTransUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BoundaryTransUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.outc1 = (OutConv(64, n_classes))
        self.outc2 = (OutConv(256, n_classes))
        self.outc3 = (OutConv(64 * 5, n_classes))

        self.boundary_module = BoundaryRenderingModule()
        self.multiScaleTransformer = MultiScaleTransformer([512, 256, 128], [256, 256, 256], image_size=[(256, 256, 256), (64, 64, 64)])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        boundary_input = self.outc1(d1)      # (B, C, D, H, W)

        # boundary_input = boundary_input.permute(3, 0, 1, 2).unsqueeze(0)  # (B, C, D, H, W)

        # 获取边界渲染模块的输出
        pred1 = self.boundary_module(boundary_input)  #  (B, C, D, H, W)

        # 获取融合后的特征图
        mul_feature_map = self.multiScaleTransformer(d4, d3, d2) #  (B, C, D, H, W)
        pred2 = self.outc2(mul_feature_map)

        # 将边界渲染图与特征融合图拼接
        # 拼接时，确保两者在通道维度拼接
        predf = torch.cat((pred1, pred2), dim=1)  # 在通道维度拼接
        # 拼接融合后的特征图
        # concatenated_output = torch.cat((concatenated_output, mul_feature_map), dim=1)

        predf = self.outc3(predf)
        return predf

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

