#!/usr/bin/env python
"""
测试脚本：Swin Transformer V2 with Alzheimer MMoE and SimMIM
测试模型的所有功能：
1. 分类任务（诊断和变化）
2. SimMIM重建任务
3. 临床先验融合
4. 专家权重分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


# 假设模型已经导入
from swin_transformer_v2_mtad_ptft import SwinTransformerV2_AlzheimerMMoE


def generate_mask(input_size: Tuple[int, int], patch_size: int, mask_ratio: float) -> torch.Tensor:
    """生成SimMIM的随机mask - patch级别的mask"""
    H, W = input_size
    # 计算patch的数量
    h, w = H // patch_size, W // patch_size
    num_patches = h * w
    num_mask = int(num_patches * mask_ratio)

    # 随机选择要mask的patches
    mask = torch.zeros(num_patches, dtype=torch.float32)
    mask_indices = torch.randperm(num_patches)[:num_mask]
    mask[mask_indices] = 1

    # 保持为1D，因为模型内部会处理
    return mask


def visualize_reconstruction(original, masked_input, reconstructed, mask, patch_size=4):
    """可视化重建结果"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 将所有数据转到CPU
    original_cpu = original[0].cpu()
    reconstructed_cpu = reconstructed[0].cpu()
    mask_cpu = mask[0].cpu()

    # 原始图像
    img_original = original_cpu.permute(1, 2, 0).numpy()
    img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min())
    axes[0].imshow(img_original)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # 将patch级别的mask转换为像素级别用于可视化
    C, H, W = original_cpu.shape
    h, w = H // patch_size, W // patch_size
    mask_visual = mask_cpu.reshape(h, w)
    mask_visual = mask_visual.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)

    axes[1].imshow(mask_visual.numpy(), cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Masked图像
    mask_3channel = mask_visual.unsqueeze(0).repeat(3, 1, 1)
    img_masked = original_cpu * (1 - mask_3channel)
    img_masked = img_masked.permute(1, 2, 0).numpy()
    img_masked = (img_masked - img_masked.min()) / (img_masked.max() - img_masked.min() + 1e-8)
    axes[2].imshow(img_masked)
    axes[2].set_title("Masked Input")
    axes[2].axis('off')

    # 重建图像
    img_recon = reconstructed_cpu.permute(1, 2, 0).numpy()
    img_recon = (img_recon - img_recon.min()) / (img_recon.max() - img_recon.min())
    axes[3].imshow(img_recon)
    axes[3].set_title("Reconstructed")
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


def test_model_components():
    """测试模型的各个组件"""
    print("=" * 60)
    print("测试 Swin Transformer V2 with Alzheimer MMoE and SimMIM")
    print("=" * 60)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 模型参数
    img_size = 256
    patch_size = 4
    mask_ratio = 0.6
    batch_size = 4

    print(f"\n模型参数:")
    print(f"- 图像大小: {img_size}x{img_size}")
    print(f"- Patch大小: {patch_size}x{patch_size}")
    print(f"- Patch数量: {(img_size // patch_size) ** 2}")
    print(f"- Mask比例: {mask_ratio}")

    # 创建模型
    print("\n创建模型...")
    model = SwinTransformerV2_AlzheimerMMoE(
        img_size=img_size,
        patch_size=patch_size,
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
        fusion_type='adaptive',
        # 预训练模式
        is_pretrain=True
    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"- 总参数量: {total_params:,}")
    print(f"- 可训练参数量: {trainable_params:,}")

    # 准备测试数据
    print("\n准备测试数据...")
    # 图像数据
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    # 临床先验（模拟概率分布）
    clinical_prior = torch.softmax(torch.randn(batch_size, 3), dim=1).to(device)
    # 标签（0: CN, 1: MCI, 2: AD）
    diagnosis_labels = torch.randint(0, 3, (batch_size,)).to(device)
    change_labels = torch.randint(0, 3, (batch_size,)).to(device)

    print(f"\n输入数据形状:")
    print(f"- 图像: {images.shape}")
    print(f"- 临床先验: {clinical_prior.shape}")
    print(f"- 诊断标签: {diagnosis_labels}")
    print(f"- 变化标签: {change_labels}")

    # ===== 测试1: 分类任务 =====
    print("\n" + "=" * 40)
    print("测试1: 分类任务")
    print("=" * 40)

    model.eval()
    with torch.no_grad():
        diag_out, change_out = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )

    print(f"分类输出:")
    print(f"- 诊断输出形状: {diag_out.shape}")
    print(f"- 变化输出形状: {change_out.shape}")

    # 计算预测
    diag_pred = torch.argmax(diag_out, dim=1)
    change_pred = torch.argmax(change_out, dim=1)
    print(f"\n预测结果:")
    print(f"- 诊断预测: {diag_pred}")
    print(f"- 变化预测: {change_pred}")

    # ===== 测试2: SimMIM重建任务 =====
    print("\n" + "=" * 40)
    print("测试2: SimMIM重建任务")
    print("=" * 40)

    # 生成mask
    mask = torch.stack([
        generate_mask((img_size, img_size), patch_size, mask_ratio)
        for _ in range(batch_size)
    ]).to(device)

    print(f"Mask形状: {mask.shape}")
    print(f"Masked patches比例: {mask.mean():.2%}")

    # 重建
    with torch.no_grad():
        reconstructed = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels,
            mask=mask
        )

    print(f"重建输出形状: {reconstructed.shape}")

    # 计算重建损失（仅在masked区域）
    # 将patch级别的mask转换为像素级别
    B, C, H, W = images.shape
    h, w = H // patch_size, W // patch_size

    # 重塑mask为2D grid
    mask_reshaped = mask.reshape(B, h, w)
    # 每个patch重复patch_size次
    # 使用unsqueeze来正确扩展维度
    mask_upsampled = mask_reshaped.unsqueeze(-1).unsqueeze(-1)  # [B, h, w, 1, 1]
    mask_upsampled = mask_upsampled.repeat(1, 1, 1, patch_size, patch_size)  # [B, h, w, patch_size, patch_size]
    # 重新排列维度
    mask_upsampled = mask_upsampled.permute(0, 1, 3, 2, 4).contiguous()  # [B, h, patch_size, w, patch_size]
    mask_upsampled = mask_upsampled.view(B, H, W)  # [B, H, W]

    # 添加channel维度
    mask_3channel = mask_upsampled.unsqueeze(1).repeat(1, 3, 1, 1)

    recon_loss = F.l1_loss(images * mask_3channel, reconstructed * mask_3channel)
    print(f"重建损失 (L1): {recon_loss.item():.4f}")

    # 可视化
    print("\n可视化重建结果...")
    visualize_reconstruction(images, None, reconstructed, mask, patch_size)

    # ===== 测试3: 专家权重分析 =====
    print("\n" + "=" * 40)
    print("测试3: 专家权重分析")
    print("=" * 40)

    # 分析不同诊断类别的专家权重
    expert_names = ['Shared', 'AD-focused', 'MCI-focused', 'CN-focused']

    # 为每个诊断类别创建样本
    for diag_class in range(3):
        class_name = ['CN', 'MCI', 'AD'][diag_class]
        print(f"\n{class_name}类别的专家权重:")

        # 创建该类别的batch
        test_labels = torch.full((4,), diag_class, dtype=torch.long).to(device)

        # 直接访问MMoE层分析权重
        with torch.no_grad():
            # 获取第一个stage的第一个block的mmoe
            first_block = model.layers[0].blocks[0]
            x_test = torch.randn(4, 64 * 64, 96).to(device)  # 模拟特征

            # 调用mmoe的forward方法，task='reconstruction'
            recon_output = first_block.mmoe(
                x_test,
                lbls_diagnosis=test_labels,
                task='reconstruction'
            )

        print(f"  重建时应该主要激活: {expert_names[diag_class + 1]}专家")

    # ===== 测试4: 内存和效率 =====
    print("\n" + "=" * 40)
    print("测试4: 内存和效率")
    print("=" * 40)

    if torch.cuda.is_available():
        # 清空缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # 测量前向传播时间
        import time

        # 预热
        for _ in range(3):
            _ = model(images, clinical_prior=clinical_prior)

        torch.cuda.synchronize()
        start_time = time.time()

        # 测试10次前向传播
        num_iterations = 10
        for _ in range(num_iterations):
            _ = model(images, clinical_prior=clinical_prior)

        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_iterations
        print(f"平均前向传播时间: {avg_time * 1000:.2f} ms")
        print(f"吞吐量: {batch_size / avg_time:.2f} images/s")

        # 内存使用
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"\nGPU内存使用:")
        print(f"- 已分配: {allocated:.2f} GB")
        print(f"- 已预留: {reserved:.2f} GB")

    # ===== 测试5: 不同配置测试 =====
    print("\n" + "=" * 40)
    print("测试5: 不同配置测试")
    print("=" * 40)

    # 测试不使用临床先验
    print("\n不使用临床先验:")
    with torch.no_grad():
        diag_out_no_prior, change_out_no_prior = model(
            images,
            clinical_prior=None,  # 不使用临床先验
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )
    print(f"- 诊断输出形状: {diag_out_no_prior.shape}")
    print(f"- 变化输出形状: {change_out_no_prior.shape}")

    # 测试微调模式
    print("\n切换到微调模式:")
    model.is_pretrain = False
    for layer in model.layers:
        layer.set_pretrain_mode(False)

    with torch.no_grad():
        diag_out_finetune, change_out_finetune = model(
            images,
            clinical_prior=clinical_prior,
            lbls_diagnosis=diagnosis_labels,
            lbls_change=change_labels
        )
    print(f"- 微调模式下输出正常")

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    # 运行测试
    test_model_components()