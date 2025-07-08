"""
Description: 
Author: JeffreyJ
Date: 2025/7/5
LastEditTime: 2025/7/5 14:36
Version: 1.0
"""
"""
Description: Single Task Trainer for Alzheimer's Disease Classification
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:01
Version: 2.0 - Single Task Degraded Version
"""
"""
阿尔兹海默症单任务分类训练器 - 退化版本 + SimMIM预训练
- 预训练阶段：使用SimMIM进行自监督重建任务
- 微调阶段：支持单个分类任务
  - Binary: CN vs AD (0, 1)
  - Three-class: CN, MCI, AD (0, 1, 2)
  - Custom: 任意类别数
- 使用单任务MoE架构，专家分配基于类别
- 使用wandb记录训练过程
- 包含早停机制
- 智能权重管理：预训练后自动移除decoder
"""
import logging
import os
import random
import sys
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import math

import logging
from datetime import datetime

# 假设您的数据加载器在这里
from data.data_ad_loader import build_loader_finetune


def load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=None):
    """
    加载预训练权重，可选择排除decoder

    Args:
        model: 要加载权重的模型
        pretrained_path: 预训练权重路径
        exclude_decoder: 是否排除decoder权重
        logger: 日志记录器

    Returns:
        dict: 包含加载信息的字典
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"Loading pretrained weights from: {pretrained_path}")

    # 加载checkpoint
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # 获取state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch_info = checkpoint.get('epoch', 'unknown')
        phase_info = checkpoint.get('phase', 'unknown')
        logger.info(f"Loading from epoch {epoch_info}, phase: {phase_info}")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        logger.info("Loading from 'model' key in checkpoint")
    else:
        state_dict = checkpoint
        logger.info("Loading state_dict directly")

    # 统计原始权重
    total_keys = len(state_dict)
    decoder_keys = [k for k in state_dict.keys() if k.startswith('decoder.')]

    logger.info(f"Original checkpoint contains {total_keys} parameters")
    logger.info(f"Found {len(decoder_keys)} decoder parameters")

    if exclude_decoder and decoder_keys:
        # 过滤掉decoder相关权重
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        logger.info(f"Excluded decoder parameters:")
        for key in decoder_keys:
            logger.info(f"  - {key}: {state_dict[key].shape}")

        state_dict = filtered_state_dict
        logger.info(f"Filtered state_dict contains {len(state_dict)} parameters")

    # 获取模型当前的参数
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())

    # 分析匹配情况
    matched_keys = model_keys & checkpoint_keys
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys

    logger.info(f"Parameter matching analysis:")
    logger.info(f"  - Matched parameters: {len(matched_keys)}")
    logger.info(f"  - Missing parameters: {len(missing_keys)}")
    logger.info(f"  - Unexpected parameters: {len(unexpected_keys)}")

    if missing_keys:
        logger.info("Missing parameters (will be randomly initialized):")
        for key in sorted(missing_keys):
            logger.info(f"  - {key}")

    if unexpected_keys:
        logger.info("Unexpected parameters (will be ignored):")
        for key in sorted(unexpected_keys):
            logger.info(f"  - {key}")

    # 加载权重，允许部分匹配
    load_result = model.load_state_dict(state_dict, strict=False)

    logger.info(f"Weight loading completed!")
    logger.info(f"  - Successfully loaded: {len(matched_keys)} parameters")
    logger.info(f"  - Randomly initialized: {len(missing_keys)} parameters")

    return {
        'total_keys': total_keys,
        'loaded_keys': len(matched_keys),
        'missing_keys': len(missing_keys),
        'unexpected_keys': len(unexpected_keys),
        'excluded_decoder': exclude_decoder and len(decoder_keys) > 0,
        'decoder_keys_count': len(decoder_keys)
    }


def remove_decoder_from_model(model, logger=None):
    """
    从模型中移除decoder组件并记录详细信息

    Args:
        model: 要修改的模型
        logger: 日志记录器

    Returns:
        dict: 包含移除信息的字典
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info("REMOVING DECODER FROM MODEL")
    logger.info("=" * 60)

    # 获取移除前的参数统计
    total_params_before = sum(p.numel() for p in model.parameters())
    decoder_params = 0
    decoder_components = []

    # 检查是否有decoder
    model_to_check = model.module if hasattr(model, 'module') else model

    if hasattr(model_to_check, 'decoder') and model_to_check.decoder is not None:
        # 统计decoder参数
        for name, param in model_to_check.decoder.named_parameters():
            decoder_params += param.numel()
            decoder_components.append((name, param.shape, param.numel()))

        logger.info(f"Decoder components to be removed:")
        for name, shape, numel in decoder_components:
            logger.info(f"  - {name}: {shape} ({numel:,} parameters)")

        logger.info(f"Total decoder parameters: {decoder_params:,}")

        # 移除decoder
        del model_to_check.decoder
        model_to_check.decoder = None

        logger.info("✓ Decoder successfully removed from model")

        # 如果是DataParallel，需要重新包装
        if hasattr(model, 'module'):
            logger.info("Re-wrapping model with DataParallel...")
            device_ids = list(range(torch.cuda.device_count()))
            model = nn.DataParallel(model_to_check, device_ids=device_ids)
            logger.info(f"✓ Model re-wrapped with DataParallel on devices: {device_ids}")
    else:
        logger.info("No decoder found in model or decoder already None")

    # 获取移除后的参数统计
    total_params_after = sum(p.numel() for p in model.parameters())
    memory_saved = decoder_params * 4 / (1024 ** 2)  # 假设float32，转换为MB

    logger.info(f"Parameter statistics:")
    logger.info(f"  - Before: {total_params_before:,} parameters")
    logger.info(f"  - After: {total_params_after:,} parameters")
    logger.info(f"  - Removed: {decoder_params:,} parameters")
    logger.info(f"  - Memory saved: ~{memory_saved:.2f} MB")

    logger.info("=" * 60)

    return {
        'removed': decoder_params > 0,
        'decoder_params': decoder_params,
        'params_before': total_params_before,
        'params_after': total_params_after,
        'memory_saved_mb': memory_saved,
        'components': decoder_components
    }


def log_model_components(model, phase="unknown", logger=None):
    """
    记录模型组件的详细信息

    Args:
        model: 要分析的模型
        phase: 当前阶段名称
        logger: 日志记录器
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'=' * 50}")
    logger.info(f"MODEL COMPONENTS ANALYSIS - {phase.upper()}")
    logger.info(f"{'=' * 50}")

    model_to_check = model.module if hasattr(model, 'module') else model

    # 统计各个组件的参数量
    components = {}

    for name, module in model_to_check.named_children():
        if module is not None:
            param_count = sum(p.numel() for p in module.parameters())
            components[name] = param_count

            # 特别标注重要组件
            if name in ['decoder', 'head', 'clinical_encoder', 'clinical_fusion']:
                status = "✓ Active" if param_count > 0 else "✗ None/Empty"
                logger.info(f"  {name:20}: {param_count:>10,} params {status}")
            else:
                logger.info(f"  {name:20}: {param_count:>10,} params")

    total_params = sum(components.values())
    logger.info(f"  {'=' * 40}")
    logger.info(f"  {'Total':20}: {total_params:>10,} params")

    # 检查特定组件状态
    special_components = ['decoder', 'head']
    logger.info(f"\nSpecial component status:")
    for comp in special_components:
        if hasattr(model_to_check, comp):
            attr = getattr(model_to_check, comp)
            if attr is None:
                logger.info(f"  - {comp}: None (removed/disabled)")
            else:
                param_count = sum(p.numel() for p in attr.parameters())
                logger.info(f"  - {comp}: Active ({param_count:,} params)")
        else:
            logger.info(f"  - {comp}: Not found")

    logger.info(f"{'=' * 50}")

    return components


def generate_mask(input_size: Tuple[int, int], patch_size: int, mask_ratio: float,
                  device: torch.device) -> torch.Tensor:
    """生成SimMIM的随机mask - patch级别的mask"""
    H, W = input_size
    # 计算patch的数量
    h, w = H // patch_size, W // patch_size
    num_patches = h * w
    num_mask = int(num_patches * mask_ratio)

    # 随机选择要mask的patches
    mask = torch.zeros(num_patches, dtype=torch.float32, device=device)
    mask_indices = torch.randperm(num_patches, device=device)[:num_mask]
    mask[mask_indices] = 1

    return mask


def norm_targets(targets, patch_size):
    """标准化目标 - from SimMIM"""
    assert patch_size % 2 == 1

    targets_ = targets
    targets_count = torch.ones_like(targets)

    targets_square = targets ** 2.

    targets_mean = F.avg_pool2d(targets, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                count_include_pad=False)
    targets_square_mean = F.avg_pool2d(targets_square, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                       count_include_pad=False)
    targets_count = F.avg_pool2d(targets_count, kernel_size=patch_size, stride=1, padding=patch_size // 2,
                                 count_include_pad=True) * (patch_size ** 2)

    targets_var = (targets_square_mean - targets_mean ** 2.) * (targets_count / (targets_count - 1))
    targets_var = torch.clamp(targets_var, min=0.)

    targets_ = (targets_ - targets_mean) / (targets_var + 1.e-6) ** 0.5

    return targets_


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        '''保存模型当验证损失下降时'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 保存时去除decoder（如果存在）
        model_to_save = model.module if hasattr(model, 'module') else model
        state_dict = model_to_save.state_dict()

        # 过滤decoder权重
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith('decoder.')
        }

        torch.save(filtered_state_dict, path)
        self.val_loss_min = val_loss


class SimMIMLoss(nn.Module):
    """SimMIM重建损失"""

    def __init__(self, patch_size=4, norm_target=True, norm_target_patch_size=47):
        super().__init__()
        self.patch_size = patch_size
        self.norm_target = norm_target
        self.norm_target_patch_size = norm_target_patch_size

    def forward(self, input_images, reconstructed, mask):
        """
        Args:
            input_images: 原始输入图像 [B, C, H, W]
            reconstructed: 重建图像 [B, C, H, W]
            mask: patch级别的mask [B, num_patches]
        """
        B, C, H, W = input_images.shape

        # 将patch级别的mask转换为像素级别
        h, w = H // self.patch_size, W // self.patch_size
        mask_reshaped = mask.reshape(B, h, w)

        # 扩展mask到像素级别
        mask_upsampled = mask_reshaped.unsqueeze(-1).unsqueeze(-1)
        mask_upsampled = mask_upsampled.repeat(1, 1, 1, self.patch_size, self.patch_size)
        mask_upsampled = mask_upsampled.permute(0, 1, 3, 2, 4).contiguous()
        mask_upsampled = mask_upsampled.view(B, H, W)
        mask_upsampled = mask_upsampled.unsqueeze(1).repeat(1, C, 1, 1)

        # 标准化目标（如果启用）
        targets = input_images
        if self.norm_target:
            targets = norm_targets(targets, self.norm_target_patch_size)

        # 计算重建损失（仅在masked区域）
        loss_recon = F.l1_loss(targets, reconstructed, reduction='none')
        loss = (loss_recon * mask_upsampled).sum() / (mask_upsampled.sum() + 1e-5) / C

        return loss


class SingleTaskLoss(nn.Module):
    """单任务损失函数"""

    def __init__(self, num_classes=2, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights
        )

    def forward(self, outputs: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算单任务损失
        Args:
            outputs: 模型输出 [B, num_classes]
            labels: 标签 [B] (0-based indexing)
        Returns:
            包含total_loss的字典
        """
        loss = self.criterion(outputs, labels)

        return {
            'total': loss,
            'classification': loss
        }


def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """计算评估指标 - 单任务版本 - FIXED"""
    # 获取预测结果
    pred = torch.argmax(outputs, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    # 计算准确率
    accuracy = accuracy_score(labels_np, pred)

    # 计算F1分数
    if num_classes == 2:
        # 二分类
        # 添加调试信息（训练初期）
        unique_preds = np.unique(pred)
        unique_labels = np.unique(labels_np)

        if np.random.random() < 0.01:  # 随机打印1%的批次
            print(f"\n[DEBUG] Unique predictions: {unique_preds}, Unique labels: {unique_labels}")
            print(f"[DEBUG] Pred distribution: {np.bincount(pred, minlength=2)}")
            print(f"[DEBUG] Label distribution: {np.bincount(labels_np, minlength=2)}")

        # 使用zero_division=0来避免警告
        f1 = f1_score(labels_np, pred, average='binary', zero_division=0)

        # 计算AUC (仅二分类)
        try:
            if len(unique_labels) > 1:  # 只有当有两个类别时才计算AUC
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                auc = roc_auc_score(labels_np, probs)
            else:
                auc = 0.0
        except:
            auc = 0.0

        # 计算敏感性和特异性
        try:
            cm = confusion_matrix(labels_np, pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                # 如果只有一个类别，创建一个伪矩阵
                tn = fp = fn = tp = 0
                if 0 in labels_np:
                    tn = len(labels_np)
                else:
                    tp = len(labels_np)
        except:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'balanced_accuracy': (sensitivity + specificity) / 2
        }
    else:
        # 多分类
        f1_macro = f1_score(labels_np, pred, average='macro', zero_division=0)
        f1_weighted = f1_score(labels_np, pred, average='weighted', zero_division=0)

        # 计算每类准确率
        cm = confusion_matrix(labels_np, pred, labels=list(range(num_classes)))
        per_class_acc = np.zeros(num_classes)

        for i in range(num_classes):
            if cm[i].sum() > 0:
                per_class_acc[i] = cm[i, i] / cm[i].sum()
            else:
                per_class_acc[i] = 0.0

        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'balanced_accuracy': np.mean(per_class_acc),
            'per_class_accuracy': per_class_acc.tolist()
        }

def log_expert_utilization(model, val_loader, device, epoch, num_classes):
    """记录专家利用率 - 用于分析单任务MoE的工作情况"""
    model.eval()

    with torch.no_grad():
        # 只取一个batch进行分析
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        clinical_priors = batch.get('prior', None)
        if clinical_priors is not None:
            clinical_priors = clinical_priors.to(device)

        # 获取专家利用率
        try:
            expert_utilization = model.get_expert_utilization(
                images,
                clinical_prior=clinical_priors,
                labels=labels
            )

            if expert_utilization:
                # 分析每层的专家权重
                for layer_idx, gate_weights in enumerate(expert_utilization):
                    if 'task_weights' in gate_weights:
                        task_weights = gate_weights['task_weights'].mean(dim=[0, 1])  # [num_experts]
                        expert_names = gate_weights.get('expert_names',
                                                        [f'Expert_{i}' for i in range(len(task_weights))])

                        # 记录到wandb
                        for i, name in enumerate(expert_names):
                            wandb.log({
                                f'expert_utilization/layer_{layer_idx}_{name}': task_weights[i].item(),
                                'epoch': epoch
                            })

                        # 记录专家多样性（熵）
                        entropy = -torch.sum(task_weights * torch.log(task_weights + 1e-8))
                        wandb.log({
                            f'expert_utilization/layer_{layer_idx}_entropy': entropy.item(),
                            'epoch': epoch
                        })

        except Exception as e:
            logging.warning(f"Failed to log expert utilization: {e}")


def train_one_epoch_pretrain(model, train_loader, criterion_simmim, optimizer, scheduler, device, epoch,
                             mask_ratio=0.6, patch_size=4, img_size=256, position=0):
    """预训练一个epoch - SimMIM重建任务 - FIXED版本"""
    model.train()

    # 确保模型处于预训练模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = True
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(True)

    total_loss = 0
    num_patches = (img_size // patch_size) ** 2

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Pretrain Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        clinical_priors = batch.get('prior', None)
        if clinical_priors is not None:
            clinical_priors = clinical_priors.to(device)

        batch_size = images.shape[0]

        # ===== CRITICAL FIX: 确保标签转换为0-based =====
        # 数据集使用1-based标签 (1,2,3)，模型需要0-based标签 (0,1,2)
        labels_zero_based = labels - 1

        # 安全检查：确保转换后的标签在有效范围内
        min_label = labels_zero_based.min().item()
        max_label = labels_zero_based.max().item()

        if min_label < 0:
            print(f"Warning: Found label < 1 in dataset, min original label: {labels.min().item()}")
            labels_zero_based = torch.clamp(labels_zero_based, min=0)

        # 可以添加日志来验证转换
        if idx == 0:  # 只在第一个batch打印
            print(
                f"Label conversion check - Original: {labels[:3].tolist()}, Converted: {labels_zero_based[:3].tolist()}")

        # 生成mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # 前向传播 - SimMIM重建，使用转换后的0-based标签
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            labels=labels_zero_based,  # 使用0-based标签
            mask=masks
        )

        # 计算重建损失
        loss = criterion_simmim(images, reconstructed, masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 记录损失
        total_loss += loss.item()

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}",
            'lr': f"{current_lr:.2e}"
        })

    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def train_one_epoch_finetune(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                             num_classes, use_timm_scheduler=False, position=0):
    """微调一个epoch - 单任务分类 - FIXED版本"""
    model.train()

    # 确保模型处于微调模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = False
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(False)

    total_loss = 0
    all_metrics = []

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Finetune Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        clinical_priors = batch.get('prior', None)
        if clinical_priors is not None:
            clinical_priors = clinical_priors.to(device)

        # ===== CRITICAL FIX: 确保标签转换为0-based =====
        # 数据集使用1-based标签 (1,2,3)，模型需要0-based标签 (0,1,2)
        labels_zero_based = labels - 1

        # 安全检查
        labels_zero_based = torch.clamp(labels_zero_based, min=0, max=num_classes - 1)

        # 验证日志
        if idx == 0:
            print(
                f"Finetune label check - Original: {labels[:3].tolist()}, Converted: {labels_zero_based[:3].tolist()}")

        # 前向传播 - 分类任务，使用转换后的0-based标签
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            labels=labels_zero_based  # 使用0-based标签
        )

        # 计算损失 - 注意：criterion期望0-based标签
        losses = criterion(outputs, labels_zero_based)  # 使用转换后的标签
        loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not use_timm_scheduler:
            scheduler.step()

        # 记录损失
        total_loss += loss.item()

        # 计算指标 - 为了计算指标，我们需要把标签转换回原始格式
        with torch.no_grad():
            # 注意：compute_metrics函数可能期望原始标签格式，所以我们传入原始标签
            metrics = compute_metrics(outputs, labels_zero_based, num_classes)
            all_metrics.append(metrics)

        # 更新进度条
        current_lr = optimizer.param_groups[0]['lr']
        if num_classes == 2:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'f1': f"{metrics['f1']:.4f}",
                'auc': f"{metrics['auc']:.4f}",
                'lr': f"{current_lr:.2e}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'f1': f"{metrics['f1_weighted']:.4f}",
                'lr': f"{current_lr:.2e}"
            })

    if use_timm_scheduler:
        scheduler.step(epoch)

    # 计算平均值
    avg_loss = total_loss / len(train_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'per_class_accuracy':
            # 特殊处理per_class_accuracy
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0).tolist()
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_metrics


@torch.no_grad()
def validate_pretrain(model, val_loader, criterion_simmim, device, epoch,
                      mask_ratio=0.6, patch_size=4, img_size=256, position=2):
    """预训练验证 - SimMIM重建任务 - FIXED版本"""
    model.eval()

    # 确保模型处于预训练模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = True
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(True)

    total_loss = 0

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Pretrain Val Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        clinical_priors = batch.get('prior', None)
        if clinical_priors is not None:
            clinical_priors = clinical_priors.to(device)

        batch_size = images.shape[0]

        # ===== CRITICAL FIX: 确保标签转换为0-based =====
        labels_zero_based = labels - 1
        labels_zero_based = torch.clamp(labels_zero_based, min=0)

        # 生成mask
        masks = torch.stack([
            generate_mask((img_size, img_size), patch_size, mask_ratio, device)
            for _ in range(batch_size)
        ])

        # 前向传播，使用转换后的0-based标签
        reconstructed = model(
            images,
            clinical_prior=clinical_priors,
            labels=labels_zero_based,  # 使用0-based标签
            mask=masks
        )

        # 计算损失
        loss = criterion_simmim(images, reconstructed, masks)
        total_loss += loss.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'mask_ratio': f"{mask_ratio:.2f}"
        })

    avg_loss = total_loss / len(val_loader)
    return avg_loss


@torch.no_grad()
def validate_finetune(model, val_loader, criterion, device, epoch, num_classes, position=2):
    """微调验证 - 单任务分类 - FIXED版本"""
    model.eval()

    # 确保模型处于微调模式
    model_to_check = model.module if hasattr(model, 'module') else model
    model_to_check.is_pretrain = False
    for layer in model_to_check.layers:
        layer.set_pretrain_mode(False)

    total_loss = 0
    all_metrics = []

    # For confusion matrix and classification report
    all_pred = []
    all_true = []

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Finetune Val Epoch {epoch}",
        position=position,
        leave=False
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        clinical_priors = batch.get('prior', None)
        if clinical_priors is not None:
            clinical_priors = clinical_priors.to(device)

        # ===== CRITICAL FIX: 确保标签转换为0-based =====
        labels_zero_based = labels - 1
        labels_zero_based = torch.clamp(labels_zero_based, min=0, max=num_classes - 1)

        # 前向传播，使用转换后的0-based标签
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            labels=labels_zero_based  # 使用0-based标签
        )

        # 计算损失
        losses = criterion(outputs, labels_zero_based)

        # 记录损失
        total_loss += losses['total'].item()

        # 计算指标
        metrics = compute_metrics(outputs, labels_zero_based, num_classes)
        all_metrics.append(metrics)

        # 收集预测结果 - 注意：这里我们需要转换回原始标签格式进行分析
        pred = torch.argmax(outputs, dim=1).cpu().numpy()
        pred_original = pred + 1  # 转换预测结果回1-based
        labels_original = labels.cpu().numpy()  # 使用原始1-based标签

        all_pred.extend(pred_original)
        all_true.extend(labels_original)

        # 更新进度条
        if num_classes == 2:
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'f1': f"{metrics['f1']:.4f}",
                'auc': f"{metrics['auc']:.4f}"
            })
        else:
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}",
                'f1': f"{metrics['f1_weighted']:.4f}"
            })

    # 计算平均值
    avg_loss = total_loss / len(val_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        if key == 'per_class_accuracy':
            avg_metrics[key] = np.mean([m[key] for m in all_metrics], axis=0).tolist()
        else:
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # 计算混淆矩阵 - 使用原始1-based标签
    cm = confusion_matrix(all_true, all_pred, labels=list(range(1, num_classes + 1)))

    return avg_loss, avg_metrics, cm, all_true, all_pred


def trainer_alzheimer_single_task(args, model, snapshot_path):
    """Alzheimer's disease single-task trainer with MoE - 支持SimMIM预训练 + 智能权重管理"""
    model_name = getattr(args, 'model_name', 'swin_single_task')
    if hasattr(args, 'MODEL') and hasattr(args.MODEL, 'NAME'):
        model_name = args.MODEL.NAME

    # 获取任务信息
    num_classes = getattr(args, 'num_classes', 2)
    task_type = getattr(args, 'task_type', 'binary')
    class_names = getattr(args, 'class_names', [f'Class_{i}' for i in range(num_classes)])

    # 添加时间戳和模型名称到快照路径
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    snapshot_path = os.path.join(
        os.path.dirname(snapshot_path),
        f"{model_name}_{task_type}_{num_classes}class_{timestamp}"
    )
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "training.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Initialize wandb
    wandb_name = getattr(args, 'wandb_run_name', f'single_task_{task_type}_{num_classes}class')
    wandb_dir = os.path.join(os.path.dirname(snapshot_path), 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=getattr(args, 'wandb_project', f'alzheimer-single-task-{task_type}'),
        name=wandb_name,
        config=vars(args) if hasattr(args, '__dict__') else args,
        dir=wandb_dir,
        mode='offline' if getattr(args, 'wandb_offline', False) else 'online'
    )

    # Log task information
    wandb.config.update({
        'task_type': task_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'model_type': 'single_task'
    })

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # ============== 权重加载逻辑 ==============
    start_epoch = getattr(args, 'start_epoch', 0)
    resume_path = getattr(args, 'resume', None)
    pretrained_path = getattr(args, 'pretrained', None)

    if resume_path:
        # 从中断点恢复训练
        logging.info("Resuming training from checkpoint...")
        load_info = load_pretrained_weights(model, resume_path, exclude_decoder=False, logger=logging.getLogger())

        # 从checkpoint中获取epoch信息
        checkpoint = torch.load(resume_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0) + 1
        logging.info(f"Resumed from epoch {start_epoch}")

    elif pretrained_path:
        # 加载预训练权重（排除decoder）
        logging.info("Loading pretrained weights (excluding decoder)...")
        load_info = load_pretrained_weights(model, pretrained_path, exclude_decoder=True, logger=logging.getLogger())

        # 记录权重加载信息到wandb
        wandb.log({
            'weight_loading/total_keys': load_info['total_keys'],
            'weight_loading/loaded_keys': load_info['loaded_keys'],
            'weight_loading/missing_keys': load_info['missing_keys'],
            'weight_loading/excluded_decoder': load_info['excluded_decoder']
        })

    # 记录初始模型组件
    log_model_components(model, "Initial", logging.getLogger())

    # Data loader
    logging.info("Loading data...")
    if hasattr(args, 'config'):
        config = args.config
    elif hasattr(args, 'DATA'):
        config = args
    else:
        config = type('Config', (), {})()
        config.DATA = args.DATA if hasattr(args, 'DATA') else args

    # Build data loaders
    dataset_train, dataset_val, train_loader, val_loader, mixup_fn = build_loader_finetune(config)
    logging.info(f"Train set size: {len(dataset_train)}, Val set size: {len(dataset_val)}")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs")

    # 损失函数
    # SimMIM损失（预训练）
    criterion_simmim = SimMIMLoss(
        patch_size=getattr(args, 'patch_size', 4),
        norm_target=getattr(args, 'norm_target', True),
        norm_target_patch_size=getattr(args, 'norm_target_patch_size', 47)
    )

    # 分类损失（微调）
    criterion_classification = SingleTaskLoss(
        num_classes=num_classes,
        label_smoothing=getattr(args, 'label_smoothing', 0.0)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    # 如果是resume，加载optimizer状态
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("Optimizer state loaded from checkpoint")

    # Scheduler
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    num_steps_per_epoch = len(train_loader)
    total_steps = num_steps_per_epoch * args.max_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # 如果是resume，加载scheduler状态
    if resume_path:
        checkpoint = torch.load(resume_path, map_location='cpu')
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logging.info("Scheduler state loaded from checkpoint")

    logging.info(f"Using cosine scheduler with warmup")
    logging.info(f"Warmup epochs: {warmup_epochs}, Base LR: {args.base_lr}")

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training parameters
    best_val_acc = 0
    pretrain_epochs = getattr(args, 'pretrain_epochs', args.max_epochs // 2)

    # SimMIM parameters
    mask_ratio = getattr(args, 'mask_ratio', 0.6)
    patch_size = getattr(args, 'patch_size', 4)
    img_size = getattr(args, 'img_size', 256)

    logging.info(f"Training plan for {task_type} task ({num_classes} classes):")
    logging.info(f"- Class names: {class_names}")
    logging.info(f"- Pretraining epochs: {pretrain_epochs} (SimMIM reconstruction)")
    logging.info(f"- Finetuning epochs: {args.max_epochs - pretrain_epochs} (Classification)")
    logging.info(f"- SimMIM mask ratio: {mask_ratio}")
    logging.info(f"- Starting from epoch: {start_epoch}")

    overall_pbar = tqdm(
        total=args.max_epochs - start_epoch,
        desc="Overall Training Progress",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}, {rate_fmt}]'
    )

    # 标记是否已经进行了decoder移除
    decoder_removed = False

    for epoch in range(start_epoch, args.max_epochs):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Epoch {epoch}/{args.max_epochs - 1}")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")

        is_pretrain = epoch < pretrain_epochs
        phase = "Pretrain (SimMIM)" if is_pretrain else f"Finetune ({task_type.capitalize()})"
        logging.info(f"Phase: {phase}")

        # ============== 阶段切换逻辑 ==============
        if not is_pretrain and not decoder_removed:
            # 从预训练切换到微调：移除decoder
            logging.info("\n" + "=" * 80)
            logging.info("SWITCHING FROM PRETRAINING TO FINETUNING")
            logging.info("=" * 80)

            # 移除decoder
            remove_info = remove_decoder_from_model(model, logging.getLogger())
            decoder_removed = True

            # 记录移除信息到wandb
            wandb.log({
                'model_modification/decoder_removed': remove_info['removed'],
                'model_modification/decoder_params_removed': remove_info['decoder_params'],
                'model_modification/memory_saved_mb': remove_info['memory_saved_mb'],
                'model_modification/params_after_removal': remove_info['params_after'],
                'epoch': epoch
            })

            # 重新创建optimizer（因为参数可能发生变化）
            logging.info("Recreating optimizer for remaining parameters...")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )

            # 重新创建scheduler
            remaining_steps = num_steps_per_epoch * (args.max_epochs - epoch)
            remaining_warmup = max(0, warmup_steps - epoch * num_steps_per_epoch)

            def new_lr_lambda(current_step):
                if current_step < remaining_warmup:
                    return float(current_step) / float(max(1, remaining_warmup))
                progress = float(current_step - remaining_warmup) / float(max(1, remaining_steps - remaining_warmup))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = LambdaLR(optimizer, new_lr_lambda)

            logging.info("✓ Optimizer and scheduler recreated for finetuning phase")

            # 记录切换后的模型组件
            log_model_components(model, "After Decoder Removal", logging.getLogger())

        if is_pretrain:
            # ===== 预训练阶段：SimMIM重建 =====
            train_loss = train_one_epoch_pretrain(
                model, train_loader, criterion_simmim, optimizer, scheduler, device, epoch,
                mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size, position=1
            )

            # Log training metrics
            wandb.log({
                'train/loss_simmim': train_loss,
                'train/lr': current_lr,
                'train/phase': 1,  # 1 for pretrain
                'train/mask_ratio': mask_ratio,
                'epoch': epoch
            })

            logging.info(f"Pretrain - SimMIM Loss: {train_loss:.4f}")

            # Validation (every eval_interval epochs)
            if (epoch + 1) % args.eval_interval == 0:
                val_loss = validate_pretrain(
                    model, val_loader, criterion_simmim, device, epoch,
                    mask_ratio=mask_ratio, patch_size=patch_size, img_size=img_size
                )

                wandb.log({
                    'val/loss_simmim': val_loss,
                    'val/phase': 1,  # 1 for pretrain
                    'epoch': epoch
                })

                logging.info(f"Pretrain Val - SimMIM Loss: {val_loss:.4f}")

                # Early stopping based on reconstruction loss
                early_stopping(val_loss, model, os.path.join(snapshot_path, 'pretrain_checkpoint.pth'))
                if early_stopping.early_stop:
                    logging.info("Early stopping triggered during pretraining!")
                    break

        else:
            # ===== 微调阶段：单任务分类 =====
            train_loss, train_metrics = train_one_epoch_finetune(
                model, train_loader, criterion_classification, optimizer, scheduler, device, epoch,
                num_classes, use_timm_scheduler=False, position=1
            )

            # Log training metrics
            wandb_metrics = {
                'train/loss': train_loss,
                'train/accuracy': train_metrics['accuracy'],
                'train/lr': current_lr,
                'train/phase': 0,  # 0 for finetune
                'epoch': epoch
            }

            if num_classes == 2:
                wandb_metrics.update({
                    'train/f1': train_metrics['f1'],
                    'train/auc': train_metrics['auc'],
                    'train/sensitivity': train_metrics['sensitivity'],
                    'train/specificity': train_metrics['specificity'],
                    'train/balanced_accuracy': train_metrics['balanced_accuracy']
                })
            else:
                wandb_metrics.update({
                    'train/f1_macro': train_metrics['f1_macro'],
                    'train/f1_weighted': train_metrics['f1_weighted'],
                    'train/balanced_accuracy': train_metrics['balanced_accuracy']
                })

            wandb.log(wandb_metrics)

            logging.info(f"Finetune - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}")

            # Validation (every eval_interval epochs)
            if (epoch + 1) % args.eval_interval == 0:
                val_results = validate_finetune(model, val_loader, criterion_classification, device, epoch, num_classes)
                val_loss, val_metrics, cm, all_true, all_pred = val_results

                # Log validation metrics
                wandb_val_metrics = {
                    'val/loss': val_loss,
                    'val/accuracy': val_metrics['accuracy'],
                    'val/phase': 0,  # 0 for finetune
                    'epoch': epoch
                }

                if num_classes == 2:
                    wandb_val_metrics.update({
                        'val/f1': val_metrics['f1'],
                        'val/auc': val_metrics['auc'],
                        'val/sensitivity': val_metrics['sensitivity'],
                        'val/specificity': val_metrics['specificity'],
                        'val/balanced_accuracy': val_metrics['balanced_accuracy']
                    })
                else:
                    wandb_val_metrics.update({
                        'val/f1_macro': val_metrics['f1_macro'],
                        'val/f1_weighted': val_metrics['f1_weighted'],
                        'val/balanced_accuracy': val_metrics['balanced_accuracy']
                    })

                wandb.log(wandb_val_metrics)

                # Log expert utilization (only during finetuning)
                log_expert_utilization(model, val_loader, device, epoch, num_classes)

                # Log confusion matrix
                fig_cm = plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap='Blues')
                plt.title(f'Confusion Matrix - {task_type.capitalize()} Classification')
                plt.colorbar()
                tick_marks = np.arange(num_classes)
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)
                plt.xlabel('Predicted')
                plt.ylabel('True')

                for i in range(num_classes):
                    for j in range(num_classes):
                        plt.text(j, i, str(cm[i, j]),
                                 ha="center", va="center", color="black")
                plt.tight_layout()

                wandb.log({
                    'val/confusion_matrix': wandb.Image(fig_cm),
                    'epoch': epoch
                })

                plt.close(fig_cm)

                logging.info(f"Finetune Val - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}")

                # Save best model (only during finetuning)
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    best_model_path = os.path.join(snapshot_path, 'best_model.pth')

                    model_to_save = model.module if hasattr(model, 'module') else model

                    # 保存时过滤decoder权重
                    state_dict = model_to_save.state_dict()
                    filtered_state_dict = {
                        k: v for k, v in state_dict.items()
                        if not k.startswith('decoder.')
                    }

                    torch.save(filtered_state_dict, best_model_path)
                    logging.info(f"Best model saved with acc: {best_val_acc:.4f}")

                    wandb.run.summary['best_val_acc'] = best_val_acc
                    wandb.run.summary['best_epoch'] = epoch

                # Early stopping based on classification accuracy
                early_stopping(-val_metrics['accuracy'], model, os.path.join(snapshot_path, 'finetune_checkpoint.pth'))
                if early_stopping.early_stop:
                    logging.info("Early stopping triggered during finetuning!")
                    break

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(snapshot_path, f'checkpoint_epoch_{epoch}.pth')
            model_to_save = model.module if hasattr(model, 'module') else model

            # 保存时过滤decoder权重
            state_dict = model_to_save.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if not k.startswith('decoder.')
            }

            torch.save({
                'epoch': epoch,
                'model_state_dict': filtered_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'is_pretrain': is_pretrain,
                'phase': phase,
                'decoder_removed': decoder_removed,
                'task_type': task_type,
                'num_classes': num_classes,
                'class_names': class_names
            }, checkpoint_path)

            logging.info(f"Checkpoint saved at epoch {epoch}")

        overall_pbar.update(1)

    # Save final model
    final_model_path = os.path.join(snapshot_path, 'final_model.pth')
    model_to_save = model.module if hasattr(model, 'module') else model

    # 保存时过滤decoder权重
    state_dict = model_to_save.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith('decoder.')
    }

    torch.save(filtered_state_dict, final_model_path)

    # 记录最终模型组件
    log_model_components(model, "Final", logging.getLogger())

    logging.info(f"\nTraining completed!")
    logging.info(f"Task: {task_type} ({num_classes} classes)")
    logging.info(f"Classes: {class_names}")
    if best_val_acc > 0:
        logging.info(f"Best validation accuracy: {best_val_acc:.4f}")

    # 记录最终结果到wandb
    wandb.run.summary.update({
        'final_accuracy': best_val_acc,
        'task_type': task_type,
        'num_classes': num_classes,
        'class_names': class_names,
        'total_epochs': args.max_epochs,
        'pretrain_epochs': pretrain_epochs
    })

    wandb.finish()
    overall_pbar.close()

    return "Training Finished!"


# 使用示例的参数类 - 单任务版本
class Args:
    def __init__(self, task_type='binary', num_classes=2):
        # 基础参数
        self.seed = 42
        self.max_epochs = 100
        self.eval_interval = 10
        self.save_interval = 20
        self.patience = 10

        # 任务特定参数
        self.task_type = task_type
        self.num_classes = num_classes

        if task_type == 'binary' or num_classes == 2:
            self.class_names = ['CN', 'AD']
        elif task_type == 'diagnosis' or num_classes == 3:
            self.class_names = ['CN', 'MCI', 'AD']
        else:
            self.class_names = [f'Class_{i}' for i in range(num_classes)]

        # 优化器参数
        self.base_lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4

        # 损失函数参数
        self.label_smoothing = 0.1

        # 训练阶段参数
        self.pretrain_epochs = 50 if task_type == 'binary' else 75

        # SimMIM参数
        self.mask_ratio = 0.6
        self.patch_size = 4
        self.img_size = 256
        self.norm_target = True
        self.norm_target_patch_size = 47

        # wandb参数
        self.wandb_project = f"alzheimer-single-task-{task_type}"
        self.exp_name = f"single-task-{task_type}-{num_classes}class"

        # 权重加载参数
        self.resume = None
        self.pretrained = None

        # 数据参数
        self.DATA = type('obj', (object,), {
            'DATASET': 'alzheimer',
            'DATA_PATH': 'path/to/data',
            'IMG_SIZE': 256,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'PIN_MEMORY': True
        })


if __name__ == "__main__":
    # 示例用法
    print("=" * 80)
    print("Single Task Trainer Examples")
    print("=" * 80)

    # 二分类示例
    print("\n1. Binary Classification (CN vs AD):")
    binary_args = Args(task_type='binary', num_classes=2)
    print(f"   Task: {binary_args.task_type}")
    print(f"   Classes: {binary_args.class_names}")
    print(f"   Pretrain epochs: {binary_args.pretrain_epochs}")
    print(f"   WandB project: {binary_args.wandb_project}")

    # 三分类示例
    print("\n2. Three-class Classification (CN/MCI/AD):")
    three_args = Args(task_type='diagnosis', num_classes=3)
    print(f"   Task: {three_args.task_type}")
    print(f"   Classes: {three_args.class_names}")
    print(f"   Pretrain epochs: {three_args.pretrain_epochs}")
    print(f"   WandB project: {three_args.wandb_project}")

    print("\n3. Usage:")
    print("   # from models.swin_transformer_v2_mtad_ptft_single_task import SwinTransformerV2_SingleTask")
    print("   # model = SwinTransformerV2_SingleTask(num_classes=2)")
    print("   # trainer_alzheimer_single_task(binary_args, model, './checkpoints/binary_exp')")

    print("\n" + "=" * 80)
    print("Single Task Trainer Ready!")
    print("=" * 80)