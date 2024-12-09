import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
np.set_printoptions(threshold=1000)


def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return self.norm(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#  ====================================================================================================================

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

#  ====================================================================================================================

class TransformerDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_depth, image_height, image_width = pair(image_size)
        patch_depth, patch_height, patch_width = pair(patch_size)
        assert image_depth % patch_depth == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_depth * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_depth, p2=patch_height, p3=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (d h w) c -> b c d h w', d=image_depth // patch_depth, h=image_height // patch_height, w=image_width // patch_width),
        )
        # 最大池化和双卷积
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.double_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, c, d, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)

        # 最大池化
        out = self.maxpool(out)
        # 双卷积
        out = self.double_conv(out)
        return out




class TransformerUp(nn.Module):
    """Upscaling with transpose convolutions and double conv"""

    def __init__(self, in_channels, out_channels, image_size, depth=2, dmodel=1024, mlp_dim=2048, patch_size=2, heads=6, dim_head=128, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_depth, image_height, image_width = pair(image_size)
        patch_depth, patch_height, patch_width = pair(patch_size)
        assert image_depth % patch_depth == 0 and image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_depth * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)', p1=patch_depth, p2=patch_height, p3=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (d h w) c -> b c d h w', d=image_depth // patch_depth, h=image_height // patch_height, w=image_width // patch_width),
        )

        # 添加上采样和双卷积
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


def forward(self, x):
        x = self.to_patch_embedding(x)  # (b, c, d, h, w) -> (b, num_patches, dim_patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        # transformer layer
        ax = self.transformer(x)
        out = self.recover_patch_embedding(ax)

        # 上采样
        out = self.upsample(out)
        # 双卷积
        out = self.double_conv(out)

        return out
