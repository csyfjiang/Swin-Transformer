"""
Description: 
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:01
Version: 1.0
"""
"""
阿尔兹海默症双任务分类训练器 - MMoE版本
- 支持两个分类任务：
  - Diagnosis (1=CN, 2=MCI, 3=Dementia)
  - Change Label (1=Stable, 2=Conversion, 3=Reversion)
- 使用MMoE架构，分别的门控网络
- 使用wandb记录训练过程
- 包含早停机制
"""
import logging
import os
import random
import sys
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import math

import logging
from datetime import datetime  # 添加这行

# 假设您的数据加载器在这里
from data.data_ad_loader import build_loader_finetune


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
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


class MultiTaskLoss(nn.Module):
    """多任务损失函数 - MMoE版本"""

    def __init__(self, weight_diagnosis=1.0, weight_change=1.0, label_smoothing=0.0):
        super().__init__()
        self.criterion_diagnosis = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_change = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weight_diagnosis = weight_diagnosis
        self.weight_change = weight_change

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                diagnosis_labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算两个任务的损失
        注意：标签是1,2,3，需要转换为0,1,2
        Returns:
            包含total_loss, diagnosis_loss, change_loss的字典
        """
        output_diagnosis, output_change = outputs

        # 将标签从1,2,3转换为0,1,2
        diagnosis_labels_zero_indexed = diagnosis_labels - 1
        change_labels_zero_indexed = change_labels - 1

        loss_diagnosis = self.criterion_diagnosis(output_diagnosis, diagnosis_labels_zero_indexed)
        loss_change = self.criterion_change(output_change, change_labels_zero_indexed)

        total_loss = self.weight_diagnosis * loss_diagnosis + self.weight_change * loss_change

        return {
            'total': total_loss,
            'diagnosis': loss_diagnosis,
            'change': loss_change
        }


def compute_metrics(outputs: Tuple[torch.Tensor, torch.Tensor],
                    diagnosis_labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, float]:
    """计算评估指标 - MMoE版本"""
    output_diagnosis, output_change = outputs

    # 获取预测结果（预测的是0,1,2，需要转回1,2,3）
    pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1
    pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

    diagnosis_labels_np = diagnosis_labels.cpu().numpy()
    change_labels_np = change_labels.cpu().numpy()

    # 计算准确率
    acc_diagnosis = accuracy_score(diagnosis_labels_np, pred_diagnosis)
    acc_change = accuracy_score(change_labels_np, pred_change)

    # 计算F1分数（使用1,2,3作为标签）
    f1_diagnosis = f1_score(diagnosis_labels_np, pred_diagnosis, labels=[1, 2, 3], average='weighted', zero_division=0)
    f1_change = f1_score(change_labels_np, pred_change, labels=[1, 2, 3], average='weighted', zero_division=0)

    return {
        'acc_diagnosis': acc_diagnosis,
        'acc_change': acc_change,
        'f1_diagnosis': f1_diagnosis,
        'f1_change': f1_change,
        'acc_avg': (acc_diagnosis + acc_change) / 2,
        'f1_avg': (f1_diagnosis + f1_change) / 2
    }

def log_expert_utilization(model, val_loader, device, epoch):
    """记录专家利用率 - 用于分析MMoE的工作情况（支持clinical prior）"""
    model.eval()
    expert_weights_list = []

    with torch.no_grad():
        # 只取一个batch进行分析
        batch = next(iter(val_loader))
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)  # 新增：获取clinical prior

        # 获取专家利用率
        try:
            expert_utilization = model.get_expert_utilization(
                images,
                clinical_prior=clinical_priors,  # 新增：传入clinical prior
                lbls_diagnosis=diagnosis_labels-1,
                lbls_change=change_labels-1
            )

            if expert_utilization:
                # 计算平均专家权重
                for layer_idx, gate_weights in enumerate(expert_utilization):
                    if 'diagnosis_weights' in gate_weights:
                        diagnosis_weights = gate_weights['diagnosis_weights'].mean(dim=[0, 1])  # [num_experts]
                        change_weights = gate_weights['change_weights'].mean(dim=[0, 1])  # [num_experts]

                        # 记录到wandb
                        expert_names = ['Shared', 'AD-focused', 'MCI-focused', 'CN-focused']
                        for i, name in enumerate(expert_names):
                            wandb.log({
                                f'expert_utilization/layer_{layer_idx}_diagnosis_{name}': diagnosis_weights[i].item(),
                                f'expert_utilization/layer_{layer_idx}_change_{name}': change_weights[i].item(),
                                'epoch': epoch
                            })
        except Exception as e:
            logging.warning(f"Failed to log expert utilization: {e}")


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch,
                                   use_timm_scheduler=False, position=0):
    """训练一个epoch - 支持step级别的scheduler更新"""
    model.train()

    total_loss = 0
    total_diagnosis_loss = 0
    total_change_loss = 0
    all_metrics = []

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Train Epoch {epoch}",
        position=position,  # 使用传入的位置参数
        leave=False  # 完成后不保留进度条
    )

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)

        # 前向传播
        outputs = model(
            images,
            clinical_prior=clinical_priors,
            lbls_diagnosis=diagnosis_labels - 1,
            lbls_change=change_labels - 1
        )

        # 计算损失
        losses = criterion(outputs, diagnosis_labels, change_labels)
        loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===== 新增：step级别的scheduler更新（用于warmup） =====
        if not use_timm_scheduler:
            scheduler.step()

        # 记录损失
        total_loss += loss.item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # 计算指标
        with torch.no_grad():
            metrics = compute_metrics(outputs, diagnosis_labels, change_labels)
            all_metrics.append(metrics)

        # 更新进度条，显示当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc_diag': f"{metrics['acc_diagnosis']:.4f}",  # 新增
            'acc_chg': f"{metrics['acc_change']:.4f}",  # 新增
            'acc_avg': f"{metrics['acc_avg']:.4f}",
            'lr': f"{current_lr:.2e}"
        })

    # ===== 如果使用timm scheduler，在epoch结束时更新 =====
    if use_timm_scheduler:
        scheduler.step(epoch)

    # 计算平均值
    avg_loss = total_loss / len(train_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(train_loader)
    avg_change_loss = total_change_loss / len(train_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, position=2):
    """Validation function - MMoE版本（支持clinical prior）"""
    model.eval()

    total_loss = 0
    total_diagnosis_loss = 0
    total_change_loss = 0
    all_metrics = []

    # For confusion matrix and classification report
    all_pred_diagnosis = []
    all_true_diagnosis = []
    all_pred_change = []
    all_true_change = []

    pbar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Val Epoch {epoch}",
        position=position,  # 使用传入的位置参数
        leave=False  # 完成后不保留进度条
    )


    for idx, batch in pbar:
        # Get data
        images = batch['image'].to(device)
        diagnosis_labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)
        clinical_priors = batch['prior'].to(device)  # 新增：获取clinical prior

        # Forward pass - 传递MMoE需要的标签和clinical prior
        outputs = model(
            images,
            clinical_prior=clinical_priors,  # 新增：传入clinical prior
            lbls_diagnosis=diagnosis_labels-1,  # 转换为0,1,2用于MoE路由
            lbls_change=change_labels-1
        )

        # Calculate loss
        losses = criterion(outputs, diagnosis_labels, change_labels)

        # Record losses
        total_loss += losses['total'].item()
        total_diagnosis_loss += losses['diagnosis'].item()
        total_change_loss += losses['change'].item()

        # Calculate metrics
        metrics = compute_metrics(outputs, diagnosis_labels, change_labels)
        all_metrics.append(metrics)

        # Collect predictions
        output_diagnosis, output_change = outputs
        pred_diagnosis = torch.argmax(output_diagnosis, dim=1).cpu().numpy() + 1  # Convert back to 1,2,3
        pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1  # Convert back to 1,2,3

        all_pred_diagnosis.extend(pred_diagnosis)
        all_true_diagnosis.extend(diagnosis_labels.cpu().numpy())
        all_pred_change.extend(pred_change)
        all_true_change.extend(change_labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'acc': f"{metrics['acc_avg']:.4f}"
        })

    # 后续代码保持不变...
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_diagnosis_loss = total_diagnosis_loss / len(val_loader)
    avg_change_loss = total_change_loss / len(val_loader)

    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Calculate confusion matrices
    cm_diagnosis = confusion_matrix(all_true_diagnosis, all_pred_diagnosis, labels=[1, 2, 3])
    cm_change = confusion_matrix(all_true_change, all_pred_change, labels=[1, 2, 3])

    # Return all results including prediction lists
    return (avg_loss, avg_diagnosis_loss, avg_change_loss, avg_metrics, cm_diagnosis, cm_change,
            all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change)


def trainer_alzheimer_mmoe(args, model, snapshot_path):
    """Alzheimer's disease dual-task trainer with MMoE main function"""
    model_name = getattr(args, 'model_name', 'swin_admoe')  # 默认名称
    if hasattr(args, 'MODEL') and hasattr(args.MODEL, 'NAME'):
        model_name = args.MODEL.NAME

    # 添加时间戳和模型名称到快照路径
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 构建新的快照路径: base_path/model_name_timestamp
    snapshot_path = os.path.join(
        os.path.dirname(snapshot_path),  # 获取父目录
        f"{model_name}_{timestamp}"  # 模型名称_时间戳
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
    wandb_name = getattr(args, 'wandb_run_name', getattr(args, 'exp_name', 'alzheimer_mmoe_run'))
    wandb_dir = os.path.join(os.path.dirname(snapshot_path), 'wandb')  # 将wandb目录放在上一级
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=getattr(args, 'wandb_project', 'alzheimer-mmoe-classification'),
        name=wandb_name,
        config=vars(args) if hasattr(args, '__dict__') else args,
        dir=wandb_dir,
        mode='offline' if getattr(args, 'wandb_offline', False) else 'online'
    )

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Data loader with automatic train/val split
    logging.info("Loading data...")

    # Get config object properly
    if hasattr(args, 'config'):
        config = args.config
    elif hasattr(args, 'DATA'):
        config = args  # args itself is the config
    else:
        # Create a simple namespace for config
        config = type('Config', (), {})()
        config.DATA = args.DATA if hasattr(args, 'DATA') else args

    # Get data path
    data_path = getattr(config.DATA, 'DATA_PATH', getattr(args, 'data_path', ''))

    if not data_path:
        raise ValueError("Data path not specified!")

    # Check if train/val subdirectories exist
    train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        logging.info("Train/val directories not found. Using automatic 80/20 split from single directory.")

        # Modify config to use single directory with auto-split
        if hasattr(config, 'defrost'):
            config.defrost()
            config.DATA.AUTO_SPLIT = True
            config.DATA.SPLIT_RATIO = 0.8  # 80% train, 20% val
            config.freeze()

    # Build data loaders
    dataset_train, dataset_val, train_loader, val_loader, mixup_fn = build_loader_finetune(config)
    logging.info(f"Train set size: {len(dataset_train)}, Val set size: {len(dataset_val)}")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs")

    # Loss function
    criterion = MultiTaskLoss(
        weight_diagnosis=getattr(args, 'weight_diagnosis', 1.0),
        weight_change=getattr(args, 'weight_change', 1.0),
        label_smoothing=getattr(args, 'label_smoothing', 0.0)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    # ===== 修改：完全使用PyTorch原生调度器 =====
    # Warmup设置
    warmup_epochs = getattr(args, 'warmup_epochs', 5)  # 默认5个epoch的warmup

    # 计算总的训练步数
    num_steps_per_epoch = len(train_loader)
    total_steps = num_steps_per_epoch * args.max_epochs
    warmup_steps = num_steps_per_epoch * warmup_epochs

    # 使用带warmup的cosine调度器
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        # Warmup阶段
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine annealing阶段
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    logging.info(f"Using cosine scheduler with warmup")
    logging.info(f"Warmup epochs: {warmup_epochs}, Base LR: {args.base_lr}, Min LR: {args.min_lr}")

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training loop
    best_val_acc = 0
    start_epoch = getattr(args, 'start_epoch', 0)

    # 记录是否处于预训练阶段
    pretrain_epochs = getattr(args, 'pretrain_epochs', args.max_epochs // 2)
    logging.info(f"Pretraining for {pretrain_epochs} epochs, then fine-tuning")

    overall_pbar = tqdm(
        total=args.max_epochs - start_epoch,
        desc="Overall Training Progress",
        position=0,
        leave=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} epochs [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for epoch in range(start_epoch, args.max_epochs):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Epoch {epoch}/{args.max_epochs - 1}")

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning rate: {current_lr:.6f}")

        # 设置训练模式（预训练 vs 微调）
        is_pretrain = epoch < pretrain_epochs
        if hasattr(model, 'set_pretrain_mode'):
            model.set_pretrain_mode(is_pretrain)
        elif hasattr(model, 'module') and hasattr(model.module, 'set_pretrain_mode'):
            model.module.set_pretrain_mode(is_pretrain)

        phase = "Pretrain" if is_pretrain else "Finetune"
        logging.info(f"Phase: {phase}")

        # Training - 注意这里传入 use_timm_scheduler=False
        train_loss, train_diagnosis_loss, train_change_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
            use_timm_scheduler=False, position=1  # 设置位置为1，避免与整体进度条冲突
        )

        # Log training metrics
        wandb.log({
            'train/loss': train_loss,
            'train/loss_diagnosis': train_diagnosis_loss,
            'train/loss_change': train_change_loss,
            'train/acc_diagnosis': train_metrics['acc_diagnosis'],
            'train/acc_change': train_metrics['acc_change'],
            'train/acc_avg': train_metrics['acc_avg'],
            'train/f1_diagnosis': train_metrics['f1_diagnosis'],
            'train/f1_change': train_metrics['f1_change'],
            'train/f1_avg': train_metrics['f1_avg'],
            'train/lr': current_lr,
            'train/phase': 1 if is_pretrain else 0,
            'epoch': epoch
        })

        logging.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['acc_avg']:.4f} ({phase})")

        # Validation (every eval_interval epochs)
        if (epoch + 1) % args.eval_interval == 0:
            val_results = validate(model, val_loader, criterion, device, epoch)

            if len(val_results) == 6:
                val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change = val_results
                all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = None, None, None, None
            else:
                # Extended return with prediction lists
                val_loss, val_diagnosis_loss, val_change_loss, val_metrics, cm_diagnosis, cm_change, \
                    all_true_diagnosis, all_pred_diagnosis, all_true_change, all_pred_change = val_results

            # Log validation metrics
            wandb.log({
                'val/loss': val_loss,
                'val/loss_diagnosis': val_diagnosis_loss,
                'val/loss_change': val_change_loss,
                'val/acc_diagnosis': val_metrics['acc_diagnosis'],
                'val/acc_change': val_metrics['acc_change'],
                'val/acc_avg': val_metrics['acc_avg'],
                'val/f1_diagnosis': val_metrics['f1_diagnosis'],
                'val/f1_change': val_metrics['f1_change'],
                'val/f1_avg': val_metrics['f1_avg'],
                'val/phase': 1 if is_pretrain else 0,  # 1 for pretrain, 0 for finetune
                'epoch': epoch
            })

            # Log expert utilization
            log_expert_utilization(model, val_loader, device, epoch)

            # Log confusion matrices
            diagnosis_names = ['CN', 'MCI', 'Dementia']
            change_names = ['Stable', 'Conversion', 'Reversion']

            # Create confusion matrix plots
            fig_cm_diagnosis = plt.figure(figsize=(8, 6))
            plt.imshow(cm_diagnosis, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix - Diagnosis')
            plt.colorbar()
            tick_marks = np.arange(3)
            plt.xticks(tick_marks, diagnosis_names, rotation=45)
            plt.yticks(tick_marks, diagnosis_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Add value labels
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, str(cm_diagnosis[i, j]),
                             ha="center", va="center", color="black")
            plt.tight_layout()

            fig_cm_change = plt.figure(figsize=(8, 6))
            plt.imshow(cm_change, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix - Change Label')
            plt.colorbar()
            plt.xticks(tick_marks, change_names, rotation=45)
            plt.yticks(tick_marks, change_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Add value labels
            for i in range(3):
                for j in range(3):
                    plt.text(j, i, str(cm_change[i, j]),
                             ha="center", va="center", color="black")
            plt.tight_layout()

            wandb.log({
                'val/confusion_matrix_diagnosis': wandb.Image(fig_cm_diagnosis),
                'val/confusion_matrix_change': wandb.Image(fig_cm_change),
                'epoch': epoch
            })

            plt.close(fig_cm_diagnosis)
            plt.close(fig_cm_change)

            # Log classification reports if available
            if all_true_diagnosis is not None and all_pred_diagnosis is not None:
                try:
                    diagnosis_report = classification_report(
                        all_true_diagnosis, all_pred_diagnosis,
                        labels=[1, 2, 3],
                        target_names=['CN', 'MCI', 'Dementia'],
                        output_dict=True
                    )

                    change_report = classification_report(
                        all_true_change, all_pred_change,
                        labels=[1, 2, 3],
                        target_names=['Stable', 'Conversion', 'Reversion'],
                        output_dict=True
                    )

                    # Log per-class metrics
                    for class_name, metrics in diagnosis_report.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                wandb.log({f'val/diagnosis_{class_name}_{metric_name}': value, 'epoch': epoch})

                    for class_name, metrics in change_report.items():
                        if isinstance(metrics, dict):
                            for metric_name, value in metrics.items():
                                wandb.log({f'val/change_{class_name}_{metric_name}': value, 'epoch': epoch})
                except Exception as e:
                    logging.warning(f"Failed to generate classification report: {e}")

            logging.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_metrics['acc_avg']:.4f} ({phase})")

            # Save best model
            if val_metrics['acc_avg'] > best_val_acc:
                best_val_acc = val_metrics['acc_avg']
                best_model_path = os.path.join(snapshot_path, 'best_model.pth')

                # Handle DataParallel
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), best_model_path)
                logging.info(f"Best model saved with acc: {best_val_acc:.4f}")

                # Record best metrics
                wandb.run.summary['best_val_acc'] = best_val_acc
                wandb.run.summary['best_epoch'] = epoch

            # Early stopping check
            early_stopping(val_loss, model, os.path.join(snapshot_path, 'checkpoint.pth'))
            if early_stopping.early_stop:
                logging.info("Early stopping triggered!")
                break

        # Update learning rate
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(snapshot_path, f'checkpoint_epoch_{epoch}.pth')
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'is_pretrain': is_pretrain
            }, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(snapshot_path, 'final_model.pth')
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), final_model_path)

    logging.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    wandb.finish()

    return "Training Finished!"


# 使用示例的参数类 - MMoE版本
class Args:
    def __init__(self):
        # 基础参数
        self.seed = 42
        self.max_epochs = 100
        self.eval_interval = 10  # 每10个epoch验证一次
        self.save_interval = 20
        self.patience = 5  # 早停耐心值

        # 优化器参数
        self.base_lr = 1e-4
        self.min_lr = 1e-6
        self.weight_decay = 1e-4

        # 损失函数参数
        self.weight_diagnosis = 1.0  # 诊断任务权重
        self.weight_change = 1.0     # 变化任务权重
        self.label_smoothing = 0.1

        # MMoE特定参数
        self.pretrain_epochs = 50    # 预训练轮数

        # wandb参数
        self.wandb_project = "alzheimer-mmoe-classification"
        self.exp_name = "dual-task-mmoe-classification"

        # 数据参数（从您的config）
        self.DATA = type('obj', (object,), {
            'DATASET': 'alzheimer',
            'DATA_PATH': 'path/to/data',
            'IMG_SIZE': 256,
            'BATCH_SIZE': 32,
            'NUM_WORKERS': 4,
            'PIN_MEMORY': True
        })

        # 模型输出类别数（都是3类）
        self.num_classes = 3


if __name__ == "__main__":
    # 示例用法
    args = Args()

    # 假设您的MMoE模型返回两个输出 (diagnosis_logits, change_logits)
    # model = SwinTransformerV2_AlzheimerMMoE(num_classes_diagnosis=3, num_classes_change=3)

    # snapshot_path = "./checkpoints/mmoe_exp1"
    # trainer_alzheimer_mmoe(args, model, snapshot_path)