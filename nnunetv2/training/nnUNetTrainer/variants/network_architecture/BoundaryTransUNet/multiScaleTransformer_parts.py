import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_parts import TransformerDown, Transformer, TransformerUp


class MultiScaleTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048,
                 patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        # 3个不同尺度的Transformer
        self.transformer1 = TransformerDown(in_channels[0], out_channels[0], image_size[0], depth, dmodel, mlp_dim, patch_size,
                                            heads, dim_head, dropout, emb_dropout)
        self.transformer2 = Transformer(dmodel, depth, heads, dim_head, mlp_dim, dropout, patch_size)
        self.transformer3 = TransformerUp(in_channels[2], out_channels[2], image_size[1], depth, dmodel, mlp_dim, patch_size,
                                            heads, dim_head, dropout, emb_dropout)

        # 可学习的权重，用于加权求和
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, x1, x2, x3):
        # 通过不同尺度的Transformer
        out1 = self.transformer1(x1)
        out2 = self.transformer2(x2)
        out3 = self.transformer3(x3)

        # 上采样至目标尺寸
        # out1 = self.upsample1(out1)
        # out2 = self.upsample2(out2)
        # out3 = self.upsample3(out3)

        # 加权求和
        weighted_features = out1 * self.weights[0] + out2 * self.weights[1] + out3 * self.weights[2]

        # 通过卷积层输出特征图
        # final_out = self.final_conv(weighted_sum)
        return weighted_features
