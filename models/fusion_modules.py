import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import normalize

import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


class FTF(nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=128,
                 num_heads=4,  # 减少注意力头数
                 dropout=0.1,
                 expansion=2):  # 降低扩展倍数
        super().__init__()

        # 高效通道压缩
        self.channel_compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 1)
        )

        # 轻量位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, out_channels, 16, 16))  # 固定尺寸编码

        # 空间下采样
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 高效Transformer层
        self.transformer = EfficientTransformer(
            dim=out_channels,
            heads=num_heads,
            dim_head=32,  # 固定头维度
            dropout=dropout
        )

        # 深度可分离卷积重组
        self.reconstruct = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 通道压缩
        x = self.channel_compress(x)  # [B, 128, H, W]

        # 空间下采样 (减少序列长度)
        x_down = self.downsample(x)  # [B, 128, H, W]
        _, _, h, w = x_down.shape

        # 自适应位置编码
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode='bilinear', align_corners=True)
        x_down = x_down + pos_embed

        # 序列化处理
        x_seq = x_down.flatten(2).permute(0, 2, 1)  # [B, h*w, C]

        # 高效Transformer处理
        x_seq = self.transformer(x_seq)

        # 恢复空间维度
        x_trans = x_seq.permute(0, 2, 1).view(B, -1, h, w)

        # 空间上采样
        x_up = self.upsample(x_trans)

        # 残差连接
        x = x + x_up

        # 轻量特征重组
        return self.reconstruct(x)


class EfficientTransformer(nn.Module):
    """轻量级Transformer模块"""

    def __init__(self, dim, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        self.dim_head = dim_head  # 存储头维度
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子保持浮点

        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        res = x
        x = self.norm1(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 修正维度参数为self.dim_head
        q, k, v = map(lambda t: t.reshape(t.shape[0], -1, self.heads, self.dim_head).transpose(1, 2), qkv)

        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(x.shape[0], -1, self.heads * self.dim_head)
        out = self.to_out(out)

        x = res + out
        x = x + self.ff(self.norm2(x))
        return x


### MoE Gate ###
class MoEGate(nn.Module):
    """Dynamic gating network for generating expert weights (Noisy Top1 MoE version)."""

    def __init__(self, in_channels, num_experts, temperature=1.0, training_state=True, k=1):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.training = training_state
        self.k = k  # Number of top experts to select

        # Gating network: input features → expert weights
        self.gate = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # Compute raw logits [B, num_experts]
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        logits = self.gate(x)

        if self.training:
            # Add noise to the logits
            noise = torch.randn_like(logits) * F.softplus(logits / self.temperature)
            logits = logits + noise

        # Select top-k elements
        topk_values, topk_indices = torch.topk(logits, k=self.k, dim=1)

        # Create a boolean mask for the top-k elements
        mask = torch.zeros_like(logits, dtype=torch.bool)  # 使用布尔类型
        mask.scatter_(1, topk_indices, True)  # 直接标记 True/False

        masked_logits = logits.masked_fill(~mask, -float('inf'))  # 未被选中的位置设为 -inf

        # 应用 Softmax
        weights = F.softmax(masked_logits / self.temperature, dim=1)  # 输出中未被选中的位置权重严格为 0

        return weights


#### MoE ####
class MoE(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(MoE, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_experts = 5
        self.gate = MoEGate(in_channels=self.in_channel, num_experts=self.num_experts)
        self.atts = nn.ModuleList([NIN(self.in_channel // 2) for _ in range(self.num_experts)])

    def forward(self, x):
        weights = self.gate(x)  # Top1 one-hot 权重

        # 计算所有专家的输出 [B, num_experts, C, H, W]
        expert_outputs = torch.stack([att(x) for att in self.atts], dim=1)

        # 使用权重融合专家输出
        weights = weights.view(x.shape[0], self.num_experts, 1, 1, 1)  # 扩展维度
        output = (expert_outputs * weights).sum(dim=1)
        return output


################################################# CBAM ############################################################
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAMLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.combine = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.assemble = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        y = self._forward_se(x)
        z = self._forward_spatial(x)
        return y * z.expand_as(y)

    def _forward_se(self, x):
        b, c, h, w = x.size()
        # visualize_rgb_ir_features(x)
        # x1 = x[:, :x.shape[1] // 2, :, :]
        # x2 = x[:, x.shape[1] // 2:, :, :]
        #
        # x = torch.cat([x1, x2], dim=1)

        x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)

        y = torch.sigmoid(x_avg + x_max)  # 修改处
        y1 = y[:, :y.shape[1] // 2]
        y2 = y[:, y.shape[1] // 2:]

        # r_x = torch.cat([x1, (torch.mean(y2) / torch.mean(y1)) * x2], dim=1)
        # plot_y = y[0,:,0,0].detach().cpu().numpy()
        # plot_y = (plot_y - np.nanmin(plot_y)) / (np.nanmax(plot_y) - np.nanmin(plot_y))
        # plot_x = np.arange(256)
        # fig, ax = plt.subplots()
        # markerline1, stemlines, _ = plt.stem(plot_x[:128], plot_y[:128], 'k')
        # plt.setp(markerline1, 'color', 'k', 'markerfacecolor', 'k', 'mec', 'k')
        # markerline2, stemlines, _ = plt.stem(plot_x[128:], plot_y[128:], 'crimson')
        # plt.setp(markerline2, 'color', 'crimson', 'markerfacecolor', 'crimson', 'mec', 'crimson')
        # plt.savefig('cam/{i}.png'.format(i=x.shape[-2]))

        return self.combine(x * y)

    def _forward_spatial(self, x):
        # x1 = x[:, :x.shape[1] // 2, :, :]
        # x2 = x[:, x.shape[1] // 2:, :, :]
        x1_avg = torch.mean(x, 1, True)
        x1_max, _ = torch.max(x, 1, True)
        y = torch.cat((x1_avg, x1_max), 1)
        y = torch.sigmoid(self.assemble(y))
        return y


# class EFM(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(EFM, self).__init__()
#
#         # self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.PReLU(),
#             nn.Linear(channel // reduction, channel),
#         )
#         self.combine = nn.Conv2d(channel, channel // 2, kernel_size=1)
#         self.assemble = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
#
#     def forward(self, x):
#         y = self._forward_se(x)
#         z = self._forward_spatial(x)
#         return y * z.expand_as(y)
#
#     def _forward_se(self, x):
#         b, c, h, w = x.size()
#         # visualize_rgb_ir_features(x)
#         x1 = x[:, :x.shape[1] // 2, :, :]
#         x2 = x[:, x.shape[1] // 2:, :, :]
#
#         x = torch.cat([x1, x2], dim=1)
#
#         x_avg = self.fc(torch.mean(x, dim=(2, 3)).view(b, c)).view(b, c, 1, 1)
#         # x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
#
#         y = torch.tanh(x_avg)
#         # y1 = y[:, :y.shape[1] // 2]
#         # y2 = y[:, y.shape[1] // 2:]
#
#         # r_x = torch.cat([x1, (torch.mean(y2) / torch.mean(y1)) * x2], dim=1)
#         # plot_y = y[0,:,0,0].detach().cpu().numpy()
#         # plot_y = (plot_y - np.nanmin(plot_y)) / (np.nanmax(plot_y) - np.nanmin(plot_y))
#         # plot_x = np.arange(256)
#         # fig, ax = plt.subplots()
#         # markerline1, stemlines, _ = plt.stem(plot_x[:128], plot_y[:128], 'k')
#         # plt.setp(markerline1, 'color', 'k', 'markerfacecolor', 'k', 'mec', 'k')
#         # markerline2, stemlines, _ = plt.stem(plot_x[128:], plot_y[128:], 'crimson')
#         # plt.setp(markerline2, 'color', 'crimson', 'markerfacecolor', 'crimson', 'mec', 'crimson')
#         # plt.savefig('cam/{i}.png'.format(i=x.shape[-2]))
#
#         return self.combine(x * y)
#
#     def _forward_spatial(self, x):
#         x1 = x[:, :x.shape[1] // 2, :, :]
#         x2 = x[:, x.shape[1] // 2:, :, :]
#         x1_avg = torch.mean(x1, 1, True)
#         x2_avg = torch.mean(x2, 1, True)
#         y = torch.cat((x1_avg, x2_avg), 1)
#         y = torch.sigmoid(self.assemble(y))
#         return y


################################### ECA Attention #####################################

class channel_attention_block(nn.Module):
    """ Implements a Channel Attention Block """

    def __init__(self, in_channels):

        super(channel_attention_block, self).__init__()

        adaptive_k = self.channel_att_kernel_calc(in_channels)

        self.pool_types = ["max", "avg"]

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=adaptive_k, padding=(adaptive_k - 1) // 2, bias=False)

        self.combine = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        channel_att_sum = None

        for pool_type in self.pool_types:

            if pool_type == "avg":

                avg_pool = self.avg_pool(x)
                channel_att_raw = self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            elif pool_type == "max":

                max_pool = self.max_pool(x)
                channel_att_raw = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

            if channel_att_sum is None:

                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        gate = self.sigmoid(channel_att_sum).expand_as(x)

        return self.combine(x * gate)

    def channel_att_kernel_calc(self, num_channels, gamma=2, b=1):
        b = 1
        gamma = 2
        t = int(abs((math.log(num_channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        return k


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attention_block(nn.Module):
    """ Implements a Spatial Attention Block """

    def __init__(self):
        super(spatial_attention_block, self).__init__()

        kernel_size = 7

        self.compress = ChannelPool()

        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_compress = self.compress(x)

        x_out = self.spatial(x_compress)

        gate = self.sigmoid(x_out)

        return x * gate


class attention_block(nn.Module):

    def __init__(self, in_channels):
        super(attention_block, self).__init__()

        self.channel_attention_block = channel_attention_block(in_channels=in_channels)

        self.spatial_attention_block = spatial_attention_block()

    def forward(self, x):
        x_out = self.channel_attention_block(x)
        x_out_1 = self.spatial_attention_block(x_out)

        return x_out_1


################################################# Shuffle Attention ############################################################

class shuffle_attention_block(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, in_channels, groups=16):
        super(shuffle_attention_block, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.cbias = nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))
        self.sweight = nn.parameter.Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.sbias = nn.parameter.Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))

        self.combine = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)

        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)

        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        # Reduce the Channels
        out = self.combine(out)

        return out


class LCF(nn.Module):
    def __init__(self, in_channels=256, out_channels=128, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 分割输入为可见光（前128通道）和红外（后128通道）
        self.split_channels = in_channels // 2  # 128

        # 轻量化交叉注意力（参数共享）
        self.shared_attn = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels // reduction, 1, bias=False),  # 128→32
            nn.ReLU(),
            nn.Conv2d(self.split_channels // reduction, self.split_channels, 1, bias=False)  # 32→128
        )

        # 跨模态交互的深度可分离卷积
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, padding=1, groups=self.split_channels, bias=False),
            # 深度卷积
            nn.Conv2d(self.split_channels, self.split_channels, 1, bias=False)  # 点卷积
        )

        # 通道压缩与融合（256→128）
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # 256→128
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        # Step 1: 分割输入为可见光和红外特征 [B,256,H,W] → [B,128,H,W] x2
        rgb_feat = x[:, :self.split_channels, :, :]  # 前128通道为可见光
        thermal_feat = x[:, self.split_channels:, :, :]  # 后128通道为红外

        # Step 2: 交叉注意力计算
        attn_rgb = self.shared_attn(rgb_feat)  # [B,128,H,W]
        attn_thermal = self.shared_attn(thermal_feat)  # [B,128,H,W]
        cross_attn = torch.sigmoid(attn_rgb + attn_thermal)  # 注意力权重

        # Step 3: 模态交互增强
        enhanced_rgb = self.depthwise_conv(rgb_feat * cross_attn)
        enhanced_thermal = self.depthwise_conv(thermal_feat * cross_attn)

        # Step 4: 拼接特征并压缩通道 [B,256,H,W] → [B,128,H,W]
        fused_feat = torch.cat([enhanced_rgb, enhanced_thermal], dim=1)  # [B,256,H,W]
        out = self.fusion_conv(fused_feat)  # [B,128,H,W]
        return out


from mpl_toolkits.axes_grid1 import make_axes_locatable


def visualize_rgb_ir_features(x):
    """
    Visualize the sum of RGB and IR channels from a given 4D tensor.

    Parameters:
    x (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
                      The first half of the channels are assumed to be RGB features,
                      and the second half are assumed to be IR features.
    """
    if x.dim() != 4:
        raise ValueError("Input tensor must be 4-dimensional with shape (batch_size, channels, height, width).")

    batch_size, channels, height, width = x.shape

    # 分割x为rgb和ir特征
    x1 = x[:, :channels // 2, :, :]  # RGB特征
    x2 = x[:, channels // 2:, :, :]  # IR特征

    # 沿着通道维度求和
    rgb_sum = x1.sum(dim=1)  # 形状为(batch_size, height, width)
    ir_sum = x2.sum(dim=1)  # 形状为(batch_size, height, width)

    # 取第一个样本进行可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(rgb_sum[0].cpu().numpy(), cmap='viridis')  # 使用viridis颜色映射
    axes[0].set_title('Sum of RGB Channels')

    im2 = axes[1].imshow(ir_sum[0].cpu().numpy(), cmap='viridis')  # 使用viridis颜色映射
    axes[1].set_title('Sum of IR Channels')

    # 创建一个共享的颜色条
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax)

    plt.tight_layout()
    plt.show()


from torchvision.ops import DeformConv2d


class EFM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(EFM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
        )
        self.combine = nn.Conv2d(channel, channel // 2, kernel_size=1)
        self.assemble1 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.assemble2 = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self._forward_spatial(x)
        # loss_mi = self._compute_mi(x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :])
        y = self._forward_se(x)
        return y

    def _forward_se(self, x):
        b, c, h, w = x.size()
        # visualize_rgb_ir_features(x)
        # x1 = x[:, :x.shape[1] // 2, :, :]
        # x2 = x[:, x.shape[1] // 2:, :, :]
        #
        # x = torch.cat([x1, x2], dim=1)
        x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        y = torch.tanh(x_avg + x_max)
        # y1 = y[:, :y.shape[1] // 2]
        # y2 = y[:, y.shape[1] // 2:]

        # r_x = torch.cat([x1, (torch.mean(y2) / torch.mean(y1)) * x2], dim=1)
        # plot_y = y[0,:,0,0].detach().cpu().numpy()
        # plot_y = (plot_y - np.nanmin(plot_y)) / (np.nanmax(plot_y) - np.nanmin(plot_y))
        # plot_x = np.arange(256)
        # fig, ax = plt.subplots()
        # markerline1, stemlines, _ = plt.stem(plot_x[:128], plot_y[:128], 'k')
        # plt.setp(markerline1, 'color', 'k', 'markerfacecolor', 'k', 'mec', 'k')
        # markerline2, stemlines, _ = plt.stem(plot_x[128:], plot_y[128:], 'crimson')
        # plt.setp(markerline2, 'color', 'crimson', 'markerfacecolor', 'crimson', 'mec', 'crimson')
        # plt.savefig('cam/{i}.png'.format(i=x.shape[-2]))
        return self.combine(x * y)

    def _forward_spatial(self, x):
        x1 = x[:, :x.shape[1] // 2, :, :]
        x2 = x[:, x.shape[1] // 2:, :, :]

        x1_avg = torch.mean(x1, dim=1, keepdim=True)
        x1_max, _ = torch.max(x1, dim=1, keepdim=True)
        x2_avg = torch.mean(x2, dim=1, keepdim=True)
        x2_max, _ = torch.max(x2, dim=1, keepdim=True)
        att1 = torch.sigmoid(self.assemble1(torch.cat([x1_avg, x1_max], dim=1)))
        att2 = torch.sigmoid(self.assemble2(torch.cat([x2_avg, x2_max], dim=1)))
        # att = torch.softmax(torch.cat([att1, att2], dim=1), dim=1)
        # att1 = att[:, 0, :, :].unsqueeze(dim=1)
        # att2 = att[:, 1, :, :].unsqueeze(dim=1)
        # visualize_rgb_ir_features(torch.cat([att1, att2], dim=1).cpu().detach())
        # att1 = torch.where(att1 < 0.1, torch.zeros_like(att1), att1)
        # att2 = torch.where(att2 < 0.1, torch.zeros_like(att2), att2)
        # visualize_rgb_ir_features(torch.cat([att1, att2], dim=1).cpu().detach())
        y = torch.cat((x1 * att1.expand_as(x1), x2 * att2.expand_as(x2)), 1)
        return y


# class EFM(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(EFM, self).__init__()
#         self.channel = channel
#         self.reduction = reduction
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         # 跨模态通道注意力分支
#         self.fc1 = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(),
#             nn.Linear(channel // reduction, channel // 2),
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(channel, channel // reduction),
#             nn.ReLU(),
#             nn.Linear(channel // reduction, channel // 2),
#         )
#
#         # 跨模态空间注意力卷积（输入通道数4包含双模态统计量）
#         self.assemble1 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
#         self.assemble2 = nn.Conv2d(4, 1, kernel_size=7, padding=3)
#
#         self.combine = nn.Conv2d(channel, channel // 2, kernel_size=1)
#
#     def forward(self, x):
#         x = self._forward_spatial(x)
#         x1, x2 = x.chunk(2, dim=1)
#         loss_mi = self._compute_mi(x1, x2)
#         y = self._forward_se(x)
#         return y, loss_mi
#
#     def _forward_se(self, x):
#         b, c, h, w = x.size()
#         # 跨模态通道注意力
#         x_avg = self.avg_pool(x).view(b, c)
#         x_max = self.max_pool(x).view(b, c)
#
#         # 双模态交叉注意力
#         x1_att = torch.tanh(self.fc1(x_avg)) + torch.tanh(self.fc1(x_max))
#         x2_att = torch.tanh(self.fc2(x_avg)) + torch.tanh(self.fc2(x_max))
#
#         # 通道维度交互融合
#         y = x * torch.cat([x1_att.view(b, -1, 1, 1),
#                            x2_att.view(b, -1, 1, 1)], dim=1)
#         return self.combine(y)
#
#     def _forward_spatial(self, x):
#         # 双模态特征分解
#         x1, x2 = x.chunk(2, dim=1)
#
#         # 跨模态空间统计量
#         x1_stats = torch.cat([x1.mean(dim=1, keepdim=True),
#                               x1.max(dim=1, keepdim=True)[0]], dim=1)
#         x2_stats = torch.cat([x2.mean(dim=1, keepdim=True),
#                               x2.max(dim=1, keepdim=True)[0]], dim=1)
#
#         # 跨模态注意力生成
#         att1 = torch.sigmoid(self.assemble1(
#             torch.cat([x1_stats, x2_stats], dim=1)))
#         att2 = torch.sigmoid(self.assemble2(
#             torch.cat([x2_stats, x1_stats], dim=1)))
#
#         return torch.cat([x1 * att1.expand_as(x1),
#                           x2 * att2.expand_as(x2)], dim=1)
#
#     def _compute_mi(self, x1, x2):
#         # 轻量化互信息计算（基于全局特征对比）
#         b, c = x1.shape[0], x1.shape[1]
#
#         # 空间压缩+通道标准化
#         x1 = F.adaptive_avg_pool2d(x1, 1).view(b, c)  # [B,C]
#         x2 = F.adaptive_avg_pool2d(x2, 1).view(b, c)
#         x1 = F.normalize(x1, p=2, dim=1)
#         x2 = F.normalize(x2, p=2, dim=1)
#
#         # 计算相似度矩阵
#         sim_matrix = torch.einsum('nc,mc->nm', x1, x2)  # [B,B]
#
#         # 构造对比目标
#         labels = torch.arange(b, dtype=torch.long, device=x1.device)
#
#         # 对称式对比损失
#         loss_i2v = F.cross_entropy(sim_matrix / 0.07, labels)
#         loss_v2i = F.cross_entropy(sim_matrix.t() / 0.07, labels)
#         return (loss_i2v + loss_v2i) / 2


class NIN(nn.Module):
    def __init__(self, in_channels=128):
        super(NIN, self).__init__()
        # 定义1x1卷积核
        self.conv_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_tir = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 定义用于计算动态权重系数的共享非线性函数
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc = nn.Sequential(
        #     nn.Linear(in_channels, in_channels // 2),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_channels // 2, in_channels),
        # )
        # self.flownet = FlowNet()
        # 新增对齐层
        # self.align_conv = nn.Sequential(
        #     nn.Conv2d(2 * in_channels, 64, kernel_size=3, padding=1),  # 输入是拼接的特征
        #     nn.ReLU(),
        #     nn.Conv2d(64, 2, kernel_size=3, padding=1)  # 输出2通道的偏移量 (x,y)
        # )
        # #
        # # 初始化最后一层卷积为零，使初始偏移量为0
        # nn.init.zeros_(self.align_conv[-1].weight)
        # nn.init.zeros_(self.align_conv[-1].bias)

    # def forward(self, x):
    #     # 分割RGB和TIR特征
    #     x_rgb = x[:, :x.shape[1] // 2, :, :]
    #     x_tir = x[:, x.shape[1] // 2:, :, :]
    #
    #     # 应用1x1卷积和残差连接
    #     D_rgb = x_rgb + self.conv_rgb(x_rgb)
    #     D_tir = x_tir + self.conv_tir(x_tir)
    #
    #     # === 新增的空间对齐模块 ===
    #     # 1. 拼接RGB和TIR特征作为输入
    #     concat_features = torch.cat([D_rgb, D_tir], dim=1)
    #
    #     # 2. 预测空间偏移量 (归一化到[-0.1, 0.1]范围内)
    #     spatial_offset = self.align_conv(concat_features)  # [B, 2, H, W]
    #     spatial_offset = 0.1 * torch.tanh(spatial_offset)  # 限制偏移范围
    #
    #     # 3. 创建归一化网格 [-1, 1]
    #     B, _, H, W = D_tir.shape
    #     grid_y, grid_x = torch.meshgrid(
    #         torch.linspace(-1, 1, H, device=x.device),
    #         torch.linspace(-1, 1, W, device=x.device),
    #         indexing='ij'
    #     )
    #     base_grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # [1, H, W, 2]
    #     base_grid = base_grid.expand(B, H, W, 2)  # [B, H, W, 2]
    #
    #     # 4. 应用偏移并采样
    #     warped_grid = base_grid + spatial_offset.permute(0, 2, 3, 1)  # 偏移量转置为[B, H, W, 2]
    #     D_tir_aligned = torch.nn.functional.grid_sample(
    #         D_tir,
    #         warped_grid,
    #         mode='bilinear',
    #         padding_mode='border',
    #         align_corners=True
    #     )
    #     # === 对齐模块结束 ===
    #
    #     # 计算动态权重系数 (使用对齐后的TIR特征)
    #     alpha_rgb = torch.sigmoid(self.transform(D_rgb))
    #     alpha_tir = torch.sigmoid(self.transform(D_tir_aligned))
    #
    #     # 融合对齐后的特征
    #     D_fuse = alpha_rgb * D_rgb + alpha_tir * D_tir_aligned
    #     return D_fuse

    def forward(self, x):
        x_rgb = x[:, :x.shape[1] // 2, :, :]
        x_tir = x[:, x.shape[1] // 2:, :, :]
        # 应用1x1卷积和残差连接
        D_rgb = x_rgb + self.conv_rgb(x_rgb)
        D_tir = x_tir + self.conv_tir(x_tir)
        # 计算动态权重系数
        alpha_rgb = torch.sigmoid(self.transform(D_rgb))
        alpha_tir = torch.sigmoid(self.transform(D_tir))
        # alpha_rgb = torch.sigmoid(self._forward_se(D_rgb))
        # alpha_tir = torch.sigmoid(self._forward_se(D_tir))
        # 融合特征
        # flow = self.flownet(D_rgb, D_tir)
        # D_rgb = afwf(D_rgb, flow)
        # D_tir = self.spalign(D_rgb, D_tir)
        D_fuse = alpha_rgb * D_rgb + alpha_tir * D_tir
        return D_fuse

    # def _forward_se(self, x):
    #     b, c, h, w = x.size()
    #     x_avg = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
    #     x_max = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
    #     y = torch.sigmoid(x_avg + x_max)
    #     return y


# 光流估计模型
class FlowNet(nn.Module):
    """基于CNN的光流估计网络"""

    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 7, padding=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 5, padding=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, 2, 4, stride=2, padding=1)
        )

        # 可选的细化模块
        self.refine = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 3, padding=1)
        )

    def forward(self, src, tgt):
        # 拼接源图像和目标图像
        x = torch.cat([src, tgt], dim=1)

        # 通过编码器-解码器
        features = self.encoder(x)
        flow = self.decoder(features)

        # 可选的细化步骤
        refined_flow = self.refine(flow)

        # 残差连接
        return flow + refined_flow


# 特征对齐函数
def afwf(feat_src, flow):
    """
    使用光流场批量对齐源特征图到目标特征图

    参数:
        feat_src: 源特征图 (待对齐), 形状 [B, C, H, W]
        flow: 光流场, 形状 [B, 2, H, W] (flow[:,0]=x方向位移, flow[:,1]=y方向位移)

    返回:
        aligned_feat: 对齐后的特征图, 形状 [B, C, H, W]
    """
    B, C, H, W = feat_src.shape
    device = feat_src.device

    # 1. 创建基础网格 (物理坐标) - 支持批量处理
    # 生成归一化前的坐标网格
    x = torch.arange(0, W, device=device).float()
    y = torch.arange(0, H, device=device).float()

    # 创建网格并扩展到批量维度
    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')  # [H, W]
    grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)  # [B, H, W]

    # 2. 应用光流位移
    grid_x_warped = grid_x + flow[:, 0]  # x + Δx
    grid_y_warped = grid_y + flow[:, 1]  # y + Δy

    # 3. 归一化到 [-1, 1]
    grid_x_normalized = 2.0 * grid_x_warped / (W - 1) - 1.0
    grid_y_normalized = 2.0 * grid_y_warped / (H - 1) - 1.0

    # 4. 组合网格并调整维度 [B, H, W, 2]
    sample_grid = torch.stack((grid_x_normalized, grid_y_normalized), dim=-1)

    # 5. 双线性采样 (align_corners=True 确保角点对齐)
    aligned_feat = F.grid_sample(
        feat_src,
        sample_grid,
        mode='bilinear',  # 双线性插值
        padding_mode='zeros',  # 边界外填充0
        align_corners=True  # 确保角点对齐
    )

    return aligned_feat


class AdaptiveFrequencyFilter(nn.Module):
    def __init__(self, channels=3, reduction=4):
        super().__init__()
        self.channels = channels

        # 参数生成网络保持不变
        self.param_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels * 2, 1),
            nn.Sigmoid()
        )

        # 修改基滤波器初始化尺寸（匹配典型输入尺寸的频域特征）
        self.base_filter = nn.Parameter(torch.randn(1, channels, 32, 17))  # 假设输入尺寸256x256时rfft2后为256x129

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.base_filter)
        nn.init.kaiming_normal_(self.param_generator[1].weight, mode='fan_out')
        nn.init.kaiming_normal_(self.param_generator[3].weight, mode='fan_out')

    def forward(self, x):
        b, c, h, w = x.shape

        # 生成动态参数（保持不变）
        params = self.param_generator(x).view(b, 2, c, 1, 1)
        alpha, beta = params[:, 0], params[:, 1]

        # 先进行FFT变换获取频域特征尺寸
        x_fft = torch.fft.rfft2(x, norm='ortho')
        h_fft, w_fft = x_fft.size(-2), x_fft.size(-1)

        # 动态调整基滤波器尺寸（关键修改点）
        base_filter = F.interpolate(
            self.base_filter,
            size=(h_fft, w_fft),  # 直接匹配频域特征尺寸
            mode='bilinear',
            align_corners=False
        ).repeat(b, 1, 1, 1)

        # 参数调节（保持通道维度广播）
        adapted_filter = base_filter * alpha + beta

        # 应用滤波器（此时尺寸已匹配）
        real = x_fft.real * adapted_filter
        imag = x_fft.imag * adapted_filter
        filtered_fft = torch.complex(real, imag)

        # 逆变换（保持原始空间尺寸）
        filtered = torch.fft.irfft2(filtered_fft, s=(h, w), norm='ortho')

        return x + filtered


class AFF(nn.Module):
    def __init__(self, channels=128, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        self.aff_rgb = AdaptiveFrequencyFilter(self.channels, self.reduction)
        self.aff_thermal = AdaptiveFrequencyFilter(self.channels, self.reduction)
        self.conv = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, x):
        x1 = x[:, :x.shape[1] // 2, :, :]
        x2 = x[:, x.shape[1] // 2:, :, :]
        x1 = self.aff_rgb(x1)
        x2 = self.aff_thermal(x2)
        # print(x1.shape,x2.shape)
        y = self.conv(torch.cat([x1, x2], dim=1))
        return y


class EMoEGate(nn.Module):
    """Dynamic gating network for generating expert weights (Modified version with random selection)."""

    def __init__(self, in_channels, num_experts, temperature=1.0, training_state=True,
                 k=1, random_select_prob=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        self.training = training_state
        self.k = k  # Number of top experts to select
        self.random_select_prob = random_select_prob  # Probability of random selection

        # Gating network: input features → expert weights
        self.gate = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        # Compute raw logits [B, num_experts]
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = nn.Flatten()(x)
        logits = self.gate(x)

        # Initialize selected indices as top1
        _, selected_indices = torch.topk(logits, k=1, dim=1)

        # Training-only random selection
        if self.training and self.random_select_prob > 0:
            # Generate random selection flags
            rand_val = torch.rand(logits.size(0), 1, device=logits.device)
            random_select_mask = rand_val < self.random_select_prob

            if random_select_mask.any():
                # Create uniform probability distribution excluding top1
                probs = torch.ones_like(logits)
                probs.scatter_(1, selected_indices, 0)  # Set top1 position to 0
                probs = probs / probs.sum(dim=1, keepdim=True)  # Normalize

                # Sample random indices from non-top1 experts
                random_indices = torch.multinomial(probs, num_samples=1)

                # Replace top1 with random selection for flagged samples
                selected_indices = torch.where(
                    random_select_mask,
                    random_indices,
                    selected_indices
                )

        # Create boolean mask for selected expert
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, selected_indices, True)

        # Apply mask to logits
        masked_logits = logits.masked_fill(~mask, -float('inf'))

        # Compute final weights
        weights = F.softmax(masked_logits / self.temperature, dim=1)

        return weights
