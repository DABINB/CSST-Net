"""
A High-Precision Aeroengine Bearing Fault Diagnosis Based on Spatial Enhancement Convolution and Vision Transformer
********************************************************************
*                                                                  *
* Copyright © 2024 All rights reserved                             *
* Written by Mr.Wangbin                                            *
* [December 18,2024]                                               *
*                                                                  *
********************************************************************
"""
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torchsummary import summary


class CustomUnpool(nn.Module):
    def __init__(self, factor):
        super(CustomUnpool, self).__init__()
        self.factor = factor

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        output_height = height * self.factor
        output_width = width * self.factor
        return F.interpolate(x, size=(output_height, output_width), mode='nearest')

class SpatialAttention(nn.Module):
    def __init__(self, reduction_factor):
        super(SpatialAttention, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size = reduction_factor)
        self.conv = nn.Conv2d(2, 1, kernel_size = 3, padding = 1)
        self.gelu = nn.GELU()
        self.custom_unpool = CustomUnpool(reduction_factor)

    def forward(self, x):
        pooled_input = self.pool(x)
        avg_out = torch.mean(pooled_input, dim=1, keepdim=True)
        max_out, _ = torch.max(pooled_input, dim=1, keepdim=True)
        cat_map = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.conv(cat_map)
        attention_map = self.gelu(attention_map)
        upscaled_attention_map = self.custom_unpool(attention_map)
        return upscaled_attention_map

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DownsampleLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = self.gelu(out)
        return out

class SConvLayer(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor):
        super(SConvLayer, self).__init__()
        self.groups = groups

        #5*5 depthwise convolution
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups = in_channels)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()

        # 1x1 pointwise convolution
        self.conv2 = nn.Conv2d(in_channels, hidden, kernel_size=1, groups = groups)
        self.batch_norm2 = nn.BatchNorm2d(hidden)
        # Channel shuffle operation
        self.channel_shuffle = self._channel_shuffle()
        # 1x1 pointwise convolution
        self.conv3 = nn.Conv2d(hidden, in_channels, kernel_size=1, groups=groups)
        self.batch_norm3 = nn.BatchNorm2d(in_channels)
        #spatial_attention
        self.spatial_attention = SpatialAttention(reduction_factor)

    def _channel_shuffle(self):
        def shuffle(x):
            batch_size, num_channels, height, width = x.size()
            channels_per_group = num_channels // self.groups
            x = x.view(batch_size, self.groups, channels_per_group, height, width)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            x = x.view(batch_size, -1, height, width)
            return x
        return shuffle

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.gelu(out)

        out = self.channel_shuffle(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.gelu(out)

        att = self.spatial_attention(out)
        out = out * att
        output = out + residual
        return output


class SConv(nn.Module):
    def __init__(self, in_channels, hidden, groups, reduction_factor, depth):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                SConvLayer(in_channels=in_channels, hidden=hidden,  groups= groups, reduction_factor=reduction_factor)
                )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out = self.nn1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.nn2(out)
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.nn1 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)                                                       #1 50 576
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)       #1 8 50 24
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.nn1(out)
        out = self.dropout(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            out = attention(x, mask = mask)
            out = mlp(out)
        return out

class CSST_Net(nn.Module):
    def __init__(self,*, num_classes, num_patches, dim, depth, heads, mlp_dim, linear_dim, dropout=0.1):
        super(CSST_Net, self).__init__()
        self.downsample_1 = DownsampleLayer(in_channels=3, out_channels=12, kernel_size=2, stride=2)
        self.sconv_1 = SConv(in_channels=12, hidden=48, groups=4, reduction_factor=16, depth=1)
        self.downsample_2 = DownsampleLayer(in_channels=12, out_channels=24, kernel_size=2, stride=2)
        self.sconv_2 = SConv(in_channels=24, hidden=96, groups=4, reduction_factor=8, depth=1)
        self.downsample_3 = DownsampleLayer(in_channels=24, out_channels=48, kernel_size=2, stride=2)
        self.sconv_3 = SConv(in_channels=48, hidden=192, groups=4, reduction_factor=4, depth=1)
        self.downsample_4 = DownsampleLayer(in_channels=48, out_channels=96, kernel_size=2, stride=2)
        self.sconv_4 = SConv(in_channels=96, hidden=384, groups=4, reduction_factor=2, depth=1)
        self.downsample_5 = DownsampleLayer(in_channels=96, out_channels=192, kernel_size=2, stride=2)
        self.patch_conv = nn.Conv2d(dim, dim, 1, stride=1)
        self.pos_embedding = nn.Parameter(torch.empty(1, (num_patches + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, linear_dim)
        self.nn2 = nn.Linear(linear_dim, num_classes)

    def forward(self, x, mask = None):
        out = self.downsample_1(x)
        out = self.sconv_1(out)
        out = self.downsample_2(out)
        out = self.sconv_2(out)
        out = self.downsample_3(out)
        out = self.sconv_3(out)
        out = self.downsample_4(out)
        out = self.sconv_4(out)
        out = self.downsample_5(out)
        out = self.patch_conv(out)

        out = rearrange(out, 'b c h w -> b (h w) c')
        cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        out += self.pos_embedding
        out = self.dropout(out)
        out = self.transformer(out, mask)

        out = self.to_cls_token(out[:, 0])
        out = self.nn1(out)
        output = self.nn2(out)
        return output

if __name__ == "__main__":
    input_data = torch.randn(1,3,224,224).cuda()
    Model = CSST_Net(
        num_classes = 4,
        num_patches = 49,
        dim = 192,
        depth = 1,
        heads =8,
        mlp_dim = 768,
        linear_dim = 64,
        dropout =0.1).cuda()
    output = Model(input_data)
    print("The shape of output is", output.shape)
    print(Model)
    summary(Model,(3,224,224))
