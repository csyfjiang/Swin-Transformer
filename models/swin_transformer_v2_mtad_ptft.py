# --------------------------------------------------------
# Swin Transformer V2 with AlzheimerMMoE
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FeatureLevelAttention(nn.Module):
    """Feature-level attention mechanism for gating network"""

    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.dim = dim

        # Feature transformation network
        self.linear1 = nn.Linear(self.dim, int(self.dim * 0.5))
        self.feature_attention = nn.Linear(int(self.dim * 0.5), int(self.dim * 0.5))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(int(self.dim * 0.5), num_experts)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        attention_scores = self.feature_attention(x)
        attention_weights = F.softmax(attention_scores, dim=-1)
        x = x * attention_weights

        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AlzheimerMMoE(nn.Module):
    """Multi-gate Mixture of Experts for Alzheimer Disease Analysis"""

    def __init__(self, dim, hidden_dim, num_experts=4, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # 4个共享专家：shared, AD-focused, MCI-focused, CN-focused
        self.experts = nn.ModuleList([
            Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
            for _ in range(num_experts)
        ])

        # 诊断任务的门控网络
        self.diagnosis_gate = FeatureLevelAttention(dim, num_experts)
        # 变化任务的门控网络
        self.change_gate = FeatureLevelAttention(dim, num_experts)

        # 任务特定的特征变换层
        self.diagnosis_transform = nn.Linear(dim, dim)
        self.change_transform = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lbls_diagnosis=None, lbls_change=None, is_pretrain=True, temperature=1.0):
        """
        Args:
            x: 输入特征 [B, L, dim]
            lbls_diagnosis: 诊断标签 [B] (0: CN, 1: MCI, 2: AD)
            lbls_change: 变化标签 [B] (0: Stable, 1: Conversion, 2: Reversion)
            is_pretrain: 是否为预训练阶段
            temperature: softmax温度参数
        Returns:
            tuple: (diagnosis_output, change_output)
        """
        batch_size, seq_len, feature_dim = x.shape
        device = x.device  # 获取输入tensor的设备
        dtype = x.dtype  # 获取输入tensor的数据类型

        # 计算所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # [B, L, dim]
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, L, dim, num_experts]

        # === 诊断任务路由 ===
        if is_pretrain and lbls_diagnosis is not None:
            # 预训练阶段：基于诊断标签的先验路由
            diagnosis_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                            device=device, dtype=dtype)
            diagnosis_weights[:, :, 0] = 0.4  # shared expert权重

            # 根据诊断标签激活对应专家
            for i, lbl in enumerate(lbls_diagnosis):
                if lbl == 2:  # AD
                    diagnosis_weights[i, :, 1] = 0.6
                elif lbl == 1:  # MCI
                    diagnosis_weights[i, :, 2] = 0.6
                elif lbl == 0:  # CN
                    diagnosis_weights[i, :, 3] = 0.6
        else:
            # 微调阶段：使用学习的门控网络
            diagnosis_transformed_x = self.diagnosis_transform(x)
            diagnosis_gate_logits = self.diagnosis_gate(diagnosis_transformed_x)
            diagnosis_weights = F.softmax(diagnosis_gate_logits / temperature, dim=-1)

        # === 变化任务路由 ===
        if is_pretrain and lbls_change is not None:
            # 预训练阶段：基于变化标签的先验路由
            change_weights = torch.zeros(batch_size, seq_len, self.num_experts,
                                         device=device, dtype=dtype)
            change_weights[:, :, 0] = 0.4  # shared expert权重

            # 根据变化标签激活对应专家
            for i, lbl in enumerate(lbls_change):
                if lbl == 2:  # Reversion (认知改善)
                    change_weights[i, :, 3] = 0.6  # 倾向于使用CN专家
                elif lbl == 1:  # Conversion (认知恶化)
                    change_weights[i, :, 1] = 0.6  # 倾向于使用AD专家
                elif lbl == 0:  # Stable (稳定)
                    # 根据当前诊断状态决定
                    if lbls_diagnosis is not None:
                        if lbls_diagnosis[i] == 2:  # 当前AD，稳定
                            change_weights[i, :, 1] = 0.6
                        elif lbls_diagnosis[i] == 1:  # 当前MCI，稳定
                            change_weights[i, :, 2] = 0.6
                        elif lbls_diagnosis[i] == 0:  # 当前CN，稳定
                            change_weights[i, :, 3] = 0.6
                    else:
                        # 如果没有诊断标签，平均分配给所有专家
                        change_weights[i, :, 1:] = 0.2
        else:
            # 微调阶段：使用学习的门控网络
            change_transformed_x = self.change_transform(x)
            change_gate_logits = self.change_gate(change_transformed_x)
            change_weights = F.softmax(change_gate_logits / temperature, dim=-1)

        # 计算任务特定的专家输出加权组合
        # 诊断任务输出
        diagnosis_weights_expanded = diagnosis_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        diagnosis_output = torch.sum(expert_outputs * diagnosis_weights_expanded, dim=-1)  # [B, L, dim]

        # 变化任务输出
        change_weights_expanded = change_weights.unsqueeze(2)  # [B, L, 1, num_experts]
        change_output = torch.sum(expert_outputs * change_weights_expanded, dim=-1)  # [B, L, dim]

        # 应用dropout
        diagnosis_output = self.dropout(diagnosis_output)
        change_output = self.dropout(change_output)

        return diagnosis_output, change_output

    def get_gate_weights(self, x, temperature=1.0):
        """获取两个门控网络的权重分布（用于分析和可视化）"""
        diagnosis_transformed_x = self.diagnosis_transform(x)
        change_transformed_x = self.change_transform(x)

        diagnosis_gate_logits = self.diagnosis_gate(diagnosis_transformed_x)
        change_gate_logits = self.change_gate(change_transformed_x)

        diagnosis_weights = F.softmax(diagnosis_gate_logits / temperature, dim=-1)
        change_weights = F.softmax(change_gate_logits / temperature, dim=-1)

        return {
            'diagnosis_weights': diagnosis_weights,
            'change_weights': change_weights,
            'diagnosis_logits': diagnosis_gate_logits,
            'change_logits': change_gate_logits
        }


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')
        ).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # 修复：确保clamp操作中的tensor在同一设备上
        logit_scale = torch.clamp(
            self.logit_scale,
            max=torch.log(torch.tensor(1. / 0.01, device=self.logit_scale.device, dtype=self.logit_scale.dtype))
        ).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock_ADMMoE(nn.Module):
    """Swin Transformer Block with Alzheimer Multi-gate MoE."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0, is_pretrain=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_pretrain = is_pretrain

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_diagnosis = norm_layer(dim)  # 诊断任务的norm
        self.norm2_change = norm_layer(dim)  # 变化任务的norm

        # 使用AlzheimerMMoE替换标准MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mmoe = AlzheimerMMoE(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            num_experts=4,
            act_layer=act_layer,
            drop=drop
        )

        # 计算attention mask
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, lbls_diagnosis=None, lbls_change=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN with MMoE
        x = shortcut + self.drop_path(x)

        # 分别对两个任务应用norm
        x_diagnosis_norm = self.norm2_diagnosis(x)
        x_change_norm = self.norm2_change(x)

        # 使用平均norm结果作为MMoE输入
        mmoe_input = (x_diagnosis_norm + x_change_norm) / 2
        diagnosis_output, change_output = self.mmoe(
            mmoe_input,
            lbls_diagnosis=lbls_diagnosis,
            lbls_change=lbls_change,
            is_pretrain=self.is_pretrain,
            temperature=1.0
        )

        # 应用drop_path
        x_diagnosis = x + self.drop_path(diagnosis_output)
        x_change = x + self.drop_path(change_output)

        return x_diagnosis, x_change

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mmoe (估算，实际需要考虑MMoE的复杂度)
        flops += 4 * H * W * self.dim * self.dim * self.mlp_ratio  # 双任务的计算量
        # norm2 (两个任务)
        flops += 2 * self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """Patch Merging Layer."""

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)
        return x

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayerMMoE(nn.Module):
    """A basic Swin Transformer layer for one stage with Alzheimer MMoE."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, is_pretrain=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.is_pretrain = is_pretrain

        # build blocks with AlzheimerMMoE
        self.blocks = nn.ModuleList([
            SwinTransformerBlock_ADMMoE(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size,
                is_pretrain=is_pretrain)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, lbls_diagnosis=None, lbls_change=None):
        # 需要追踪两个任务的特征
        x_diagnosis, x_change = x, x

        for blk in self.blocks:
            if self.use_checkpoint:
                x_diagnosis, x_change = checkpoint.checkpoint(blk, x_diagnosis, lbls_diagnosis, lbls_change)
            else:
                x_diagnosis, x_change = blk(x_diagnosis, lbls_diagnosis, lbls_change)

        if self.downsample is not None:
            x_diagnosis = self.downsample(x_diagnosis)
            x_change = self.downsample(x_change)

        return x_diagnosis, x_change

    def set_pretrain_mode(self, is_pretrain):
        """设置预训练模式"""
        self.is_pretrain = is_pretrain
        for blk in self.blocks:
            blk.is_pretrain = is_pretrain

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class ClinicalPriorEncoder(nn.Module):
    """
    临床先验编码器
    将3维的prior向量编码到与图像特征相同的维度
    """

    def __init__(self, prior_dim=3, hidden_dim=128, output_dim=384, dropout=0.1):
        """
        Args:
            prior_dim: 输入prior向量维度 (默认3)
            hidden_dim: MLP隐藏层维度
            output_dim: 输出维度，需要与Stage 2的特征维度匹配
            dropout: dropout率
        """
        super().__init__()

        # 简单的3层MLP
        self.mlp = nn.Sequential(
            nn.Linear(prior_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, prior):
        """
        Args:
            prior: [B, 3] 临床先验向量
        Returns:
            encoded_prior: [B, output_dim] 编码后的特征
        """
        return self.mlp(prior)


class ClinicalImageFusion(nn.Module):
    """
    临床特征与图像特征融合模块
    支持多种融合策略
    """

    def __init__(self, image_dim, clinical_dim, fusion_type='adaptive', dropout=0.1):
        """
        Args:
            image_dim: 图像特征维度
            clinical_dim: 临床特征维度（应该与image_dim相同）
            fusion_type: 融合类型 ('adaptive', 'concat', 'add', 'hadamard')
            dropout: dropout率
        """
        super().__init__()

        assert clinical_dim == image_dim, "Clinical and image dimensions must match"

        self.fusion_type = fusion_type
        self.image_dim = image_dim

        if fusion_type == 'adaptive':
            # 自适应融合：学习融合权重
            self.gate = nn.Sequential(
                nn.Linear(image_dim * 2, image_dim),
                nn.ReLU(inplace=True),
                nn.Linear(image_dim, 2),
                nn.Softmax(dim=1)
            )
            self.fusion_proj = nn.Linear(image_dim, image_dim)

        elif fusion_type == 'concat':
            # 拼接后投影回原维度
            self.fusion_proj = nn.Sequential(
                nn.Linear(image_dim * 2, image_dim),
                nn.LayerNorm(image_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )

        elif fusion_type == 'add':
            # 简单相加，带可学习的缩放因子
            self.clinical_scale = nn.Parameter(torch.ones(1))
            self.image_scale = nn.Parameter(torch.ones(1))

        elif fusion_type == 'hadamard':
            # Hadamard积（逐元素乘积）+ 残差连接
            self.fusion_proj = nn.Linear(image_dim, image_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, clinical_features):
        """
        Args:
            image_features: [B, L, C] 图像特征 (L = H*W)
            clinical_features: [B, C] 临床特征（已经通过encoder编码到C维）
        Returns:
            fused_features: [B, L, C] 融合后的特征
        """
        B, L, C = image_features.shape

        # 将临床特征扩展到与图像特征相同的空间维度
        # clinical_features: [B, C] -> [B, 1, C] -> [B, L, C]
        clinical_features_expanded = clinical_features.unsqueeze(1).expand(B, L, C)

        if self.fusion_type == 'adaptive':
            # 计算自适应融合权重
            combined = torch.cat([image_features, clinical_features_expanded], dim=-1)
            weights = self.gate(combined.mean(dim=1))  # [B, 2]

            # 应用权重
            image_weight = weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
            clinical_weight = weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]

            fused = image_weight * image_features + clinical_weight * clinical_features_expanded
            fused = self.fusion_proj(fused)

        elif self.fusion_type == 'concat':
            # 拼接并投影
            combined = torch.cat([image_features, clinical_features_expanded], dim=-1)
            fused = self.fusion_proj(combined)

        elif self.fusion_type == 'add':
            # 加权相加
            fused = self.image_scale * image_features + self.clinical_scale * clinical_features_expanded

        elif self.fusion_type == 'hadamard':
            # Hadamard积 + 残差
            product = image_features * clinical_features_expanded
            fused = image_features + self.fusion_proj(product)

        return self.dropout(fused)


class SwinTransformerV2_AlzheimerMMoE(nn.Module):
    """
    扩展版Swin Transformer V2 with Alzheimer MMoE
    集成临床先验信息
    """

    def __init__(self,
                 # 原有参数
                 img_size=224, patch_size=4, in_chans=3,
                 num_classes_diagnosis=3, num_classes_change=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 is_pretrain=True,
                 # 新增参数
                 use_clinical_prior=True,
                 prior_dim=3,
                 prior_hidden_dim=128,
                 fusion_stage=2,  # 在哪个stage后融合
                 fusion_type='adaptive',
                 **kwargs):

        super().__init__()

        # ===== 保存所有原有初始化代码 =====
        self.num_classes_diagnosis = num_classes_diagnosis
        self.num_classes_change = num_classes_change
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.is_pretrain = is_pretrain

        # 临床先验相关参数
        self.use_clinical_prior = use_clinical_prior
        self.fusion_stage = fusion_stage

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ===== 临床先验编码器 =====
        if self.use_clinical_prior:
            # 计算融合stage的特征维度
            # 注意：如果在stage i后融合，实际维度是 embed_dim * 2^(i+1)
            # 因为PatchMerging会使维度翻倍
            if fusion_stage < self.num_layers - 1:
                # 如果不是最后一个stage，需要考虑PatchMerging的维度翻倍
                fusion_dim = int(embed_dim * 2 ** (fusion_stage + 1))
            else:
                # 最后一个stage没有PatchMerging
                fusion_dim = int(embed_dim * 2 ** fusion_stage)

            # 临床先验编码器
            self.clinical_encoder = ClinicalPriorEncoder(
                prior_dim=prior_dim,
                hidden_dim=prior_hidden_dim,
                output_dim=fusion_dim,
                dropout=drop_rate
            )

            # 融合模块
            self.clinical_fusion = ClinicalImageFusion(
                image_dim=fusion_dim,
                clinical_dim=fusion_dim,
                fusion_type=fusion_type,
                dropout=drop_rate
            )

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayerMMoE(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                pretrained_window_size=pretrained_window_sizes[i_layer],
                is_pretrain=is_pretrain)
            self.layers.append(layer)

        # Task-specific norms and heads
        self.norm_diagnosis = norm_layer(self.num_features)
        self.norm_change = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.head_diagnosis = nn.Linear(self.num_features, num_classes_diagnosis)
        self.head_change = nn.Linear(self.num_features, num_classes_change)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None):
        """
        Args:
            x: 图像输入
            clinical_prior: [B, 3] 临床先验向量
            lbls_diagnosis: 诊断标签
            lbls_change: 变化标签
        """
        # Patch embedding
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # 编码临床先验
        if self.use_clinical_prior and clinical_prior is not None:
            clinical_features = self.clinical_encoder(clinical_prior)
        else:
            clinical_features = None

        # 通过各层
        x_diagnosis, x_change = x, x

        for i_layer, layer in enumerate(self.layers):
            # 通过当前层
            x_diagnosis, x_change = layer(x_diagnosis, lbls_diagnosis, lbls_change)

            # 在指定stage后融合临床特征
            if self.use_clinical_prior and clinical_features is not None and i_layer == self.fusion_stage:
                # 对两个任务分支都进行融合
                x_diagnosis = self.clinical_fusion(x_diagnosis, clinical_features)
                x_change = self.clinical_fusion(x_change, clinical_features)

        # Final normalization and pooling
        x_diagnosis = self.norm_diagnosis(x_diagnosis)
        x_change = self.norm_change(x_change)

        x_diagnosis = self.avgpool(x_diagnosis.transpose(1, 2))
        x_change = self.avgpool(x_change.transpose(1, 2))

        x_diagnosis = torch.flatten(x_diagnosis, 1)
        x_change = torch.flatten(x_change, 1)

        return x_diagnosis, x_change

    def forward(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None, return_features=False):
        """
        前向传播
        Args:
            x: 输入图像
            clinical_prior: [B, 3] 临床先验向量
            lbls_diagnosis: 诊断标签
            lbls_change: 变化标签
            return_features: 是否返回特征
        """
        features_diagnosis, features_change = self.forward_features(
            x, clinical_prior, lbls_diagnosis, lbls_change
        )

        # 双任务输出
        output_diagnosis = self.head_diagnosis(features_diagnosis)
        output_change = self.head_change(features_change)

        if return_features:
            return output_diagnosis, output_change, features_diagnosis, features_change
        else:
            return output_diagnosis, output_change

    def get_expert_utilization(self, x, clinical_prior=None, lbls_diagnosis=None, lbls_change=None):
        """获取专家利用率（用于分析MMoE的工作情况）"""
        gate_weights_list = []

        # 设置hook来收集门控权重
        def hook_fn(module, input, output):
            if hasattr(module, 'mmoe') and hasattr(module.mmoe, 'get_gate_weights'):
                # 获取MMoE模块的门控权重
                mmoe_input = input[0]  # 假设输入是tuple的第一个元素
                gate_weights = module.mmoe.get_gate_weights(mmoe_input)
                gate_weights_list.append(gate_weights)

        hooks = []
        for layer in self.layers:
            for block in layer.blocks:
                if hasattr(block, 'mmoe'):
                    hook = block.register_forward_hook(hook_fn)
                    hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            _ = self.forward(x, clinical_prior, lbls_diagnosis, lbls_change)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        return gate_weights_list

    def flops(self):
        """计算FLOPs"""
        flops = 0
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        # 两个分类头的FLOPs
        flops += self.num_features * self.num_classes_diagnosis
        flops += self.num_features * self.num_classes_change
        return flops


def check_cuda_consistency(model, *inputs):
    """检查模型和数据的CUDA设备一致性"""
    model_device = next(model.parameters()).device

    print(f"模型设备: {model_device}")

    for i, data in enumerate(inputs):
        if isinstance(data, torch.Tensor):
            data_device = data.device
            print(f"输入{i}设备: {data_device}")
            assert model_device == data_device, f"设备不匹配: 模型在 {model_device}, 输入{i}在 {data_device}"
        elif data is None:
            print(f"输入{i}: None")
        else:
            print(f"输入{i}: 非tensor类型")


# ===== 使用示例 =====
if __name__ == "__main__":
    # 确保使用CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型并移动到GPU
    model = SwinTransformerV2_AlzheimerMMoE(
        img_size=256,
        patch_size=4,
        in_chans=3,
        num_classes_diagnosis=3,
        num_classes_change=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=16,
        mlp_ratio=4.,
        # 临床先验参数
        use_clinical_prior=True,
        prior_dim=3,
        prior_hidden_dim=128,
        fusion_stage=2,
        fusion_type='adaptive'
    ).to(device)

    # 测试输入并移动到GPU
    batch_size = 4
    img = torch.randn(batch_size, 3, 256, 256).to(device)
    clinical_prior = torch.randn(batch_size, 3).to(device)

    print("\n===== 设备一致性检查 =====")
    check_cuda_consistency(model, img, clinical_prior)

    print("\n===== 维度跟踪 =====")
    print(f"输入图像形状: {img.shape}, 设备: {img.device}")
    print(f"临床先验形状: {clinical_prior.shape}, 设备: {clinical_prior.device}")

    # 测试临床编码器
    fusion_dim = int(96 * 2 ** 2)  # Stage 2的特征维度 = 384
    encoder = ClinicalPriorEncoder(prior_dim=3, output_dim=fusion_dim).to(device)
    encoded_prior = encoder(clinical_prior)
    print(f"\n编码后的临床特征形状: {encoded_prior.shape}, 设备: {encoded_prior.device}")

    # 测试融合模块
    test_image_features = torch.randn(batch_size, 256, fusion_dim).to(device)
    print(f"\n图像特征形状 (Stage 2内): {test_image_features.shape}, 设备: {test_image_features.device}")

    fusion = ClinicalImageFusion(image_dim=fusion_dim, clinical_dim=fusion_dim).to(device)
    fused_features = fusion(test_image_features, encoded_prior)
    print(f"融合后的特征形状: {fused_features.shape}, 设备: {fused_features.device}")

    # 前向传播测试
    print("\n===== 完整模型测试 =====")

    # 使用临床先验
    with torch.cuda.amp.autocast():  # 使用混合精度
        diag_out, change_out = model(img, clinical_prior=clinical_prior)

    print(f"诊断输出形状: {diag_out.shape}, 设备: {diag_out.device}")
    print(f"变化输出形状: {change_out.shape}, 设备: {change_out.device}")

    # 不使用临床先验的情况
    with torch.cuda.amp.autocast():
        diag_out2, change_out2 = model(img, clinical_prior=None)

    print(f"\n不使用临床先验:")
    print(f"诊断输出形状: {diag_out2.shape}, 设备: {diag_out2.device}")
    print(f"变化输出形状: {change_out2.shape}, 设备: {change_out2.device}")

    # GPU内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        cached = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"\nGPU内存使用:")
        print(f"已分配: {allocated:.2f}GB")
        print(f"缓存: {cached:.2f}GB")

    print("\n===== 模型参数统计 =====")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 验证所有参数都在GPU上
    params_on_cuda = all(p.device.type == 'cuda' for p in model.parameters())
    print(f"所有参数都在CUDA上: {params_on_cuda}")

    print("\n===== 维度计算说明 (256x256输入) =====")
    print("Swin Transformer维度变化:")
    print(f"输入: 256x256")
    print(f"Patch Embed后: 64x64, dim=96")
    print(f"Stage 0: 64x64, dim=96 (融合点 fusion_stage=0)")
    print(f"├─ 后接PatchMerging: 32x32, dim=192")
    print(f"Stage 1: 32x32, dim=192 (融合点 fusion_stage=1)")
    print(f"├─ 后接PatchMerging: 16x16, dim=384")
    print(f"Stage 2: 16x16, dim=384 (融合点 fusion_stage=2)")
    print(f"├─ 后接PatchMerging: 8x8, dim=768")
    print(f"Stage 3: 8x8, dim=768 (融合点 fusion_stage=3)")
    print(f"└─ 无PatchMerging，直接到分类头")
    print(f"\n融合维度计算：embed_dim * 2^fusion_stage")
    print(f"- fusion_stage=0: 96 * 2^0 = 96")
    print(f"- fusion_stage=1: 96 * 2^1 = 192")
    print(f"- fusion_stage=2: 96 * 2^2 = 384")
    print(f"- fusion_stage=3: 96 * 2^3 = 768")
    print(f"\nWindow size = 16, 在Stage 3时会自动调整为8以适应特征图大小")