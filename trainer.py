"""
Description: 
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:01
Version: 1.0
"""
"""
阿尔兹海默症双任务分类训练器
- 支持两个分类任务：
  - Diagnosis (1=CN, 2=MCI, 3=Dementia)
  - Change Label (1=Stable, 2=Conversion, 3=Reversion)
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
    """多任务损失函数"""

    def __init__(self, weight_label=1.0, weight_change=1.0, label_smoothing=0.0):
        super().__init__()
        self.criterion_label = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.criterion_change = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.weight_label = weight_label
        self.weight_change = weight_change

    def forward(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算两个任务的损失
        注意：标签是1,2,3，需要转换为0,1,2
        Returns:
            包含total_loss, label_loss, change_loss的字典
        """
        output_label, output_change = outputs

        # 将标签从1,2,3转换为0,1,2
        labels_zero_indexed = labels - 1
        change_labels_zero_indexed = change_labels - 1

        loss_label = self.criterion_label(output_label, labels_zero_indexed)
        loss_change = self.criterion_change(output_change, change_labels_zero_indexed)

        total_loss = self.weight_label * loss_label + self.weight_change * loss_change

        return {
            'total': total_loss,
            'label': loss_label,
            'change': loss_change
        }


def compute_metrics(outputs: Tuple[torch.Tensor, torch.Tensor],
                    labels: torch.Tensor, change_labels: torch.Tensor) -> Dict[str, float]:
    """计算评估指标"""
    output_label, output_change = outputs

    # 获取预测结果（预测的是0,1,2，需要转回1,2,3）
    pred_label = torch.argmax(output_label, dim=1).cpu().numpy() + 1
    pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1

    labels_np = labels.cpu().numpy()
    change_labels_np = change_labels.cpu().numpy()

    # 计算准确率
    acc_label = accuracy_score(labels_np, pred_label)
    acc_change = accuracy_score(change_labels_np, pred_change)

    # 计算F1分数（使用1,2,3作为标签）
    f1_label = f1_score(labels_np, pred_label, labels=[1, 2, 3], average='weighted', zero_division=0)
    f1_change = f1_score(change_labels_np, pred_change, labels=[1, 2, 3], average='weighted', zero_division=0)

    return {
        'acc_label': acc_label,
        'acc_change': acc_change,
        'f1_label': f1_label,
        'f1_change': f1_change,
        'acc_avg': (acc_label + acc_change) / 2,
        'f1_avg': (f1_label + f1_change) / 2
    }


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_label_loss = 0
    total_change_loss = 0
    all_metrics = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch}")

    for idx, batch in pbar:
        # 获取数据
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)

        # 前向传播
        outputs = model(images, lbls=labels-1)

        # 计算损失
        losses = criterion(outputs, labels, change_labels)
        loss = losses['total']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录损失
        total_loss += loss.item()
        total_label_loss += losses['label'].item()
        total_change_loss += losses['change'].item()

        # 计算指标
        with torch.no_grad():
            metrics = compute_metrics(outputs, labels, change_labels)
            all_metrics.append(metrics)

        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{metrics['acc_avg']:.4f}"
        })

    # 计算平均值
    avg_loss = total_loss / len(train_loader)
    avg_label_loss = total_label_loss / len(train_loader)
    avg_change_loss = total_change_loss / len(train_loader)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_loss, avg_label_loss, avg_change_loss, avg_metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch):
    """Validation function"""
    model.eval()

    total_loss = 0
    total_label_loss = 0
    total_change_loss = 0
    all_metrics = []

    # For confusion matrix and classification report
    all_pred_label = []
    all_true_label = []
    all_pred_change = []
    all_true_change = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Val Epoch {epoch}")

    for idx, batch in pbar:
        # Get data
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        change_labels = batch['change_label'].to(device)

        # Forward pass
        outputs = model(images, lbls=labels-1)

        # Calculate loss
        losses = criterion(outputs, labels, change_labels)

        # Record losses
        total_loss += losses['total'].item()
        total_label_loss += losses['label'].item()
        total_change_loss += losses['change'].item()

        # Calculate metrics
        metrics = compute_metrics(outputs, labels, change_labels)
        all_metrics.append(metrics)

        # Collect predictions
        output_label, output_change = outputs
        pred_label = torch.argmax(output_label, dim=1).cpu().numpy() + 1  # Convert back to 1,2,3
        pred_change = torch.argmax(output_change, dim=1).cpu().numpy() + 1  # Convert back to 1,2,3

        all_pred_label.extend(pred_label)
        all_true_label.extend(labels.cpu().numpy())
        all_pred_change.extend(pred_change)
        all_true_change.extend(change_labels.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'acc': f"{metrics['acc_avg']:.4f}"
        })

    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_label_loss = total_label_loss / len(val_loader)
    avg_change_loss = total_change_loss / len(val_loader)

    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Calculate confusion matrices
    cm_label = confusion_matrix(all_true_label, all_pred_label, labels=[1, 2, 3])
    cm_change = confusion_matrix(all_true_change, all_pred_change, labels=[1, 2, 3])

    # Return all results including prediction lists
    return (avg_loss, avg_label_loss, avg_change_loss, avg_metrics, cm_label, cm_change,
            all_true_label, all_pred_label, all_true_change, all_pred_change)


def trainer_alzheimer(args, model, snapshot_path):
    """Alzheimer's disease dual-task trainer main function"""

    # Setup logging
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
    wandb_name = getattr(args, 'wandb_run_name', getattr(args, 'exp_name', 'alzheimer_run'))
    wandb.init(
        project=getattr(args, 'wandb_project', 'alzheimer-classification'),
        name=wandb_name,
        config=vars(args) if hasattr(args, '__dict__') else args,
        dir=snapshot_path,
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
        weight_label=getattr(args, 'weight_diagnosis', getattr(args, 'weight_label', 1.0)),
        weight_change=getattr(args, 'weight_change', 1.0),
        label_smoothing=getattr(args, 'label_smoothing', 0.0)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=args.min_lr
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training loop
    best_val_acc = 0
    start_epoch = getattr(args, 'start_epoch', 0)

    for epoch in range(start_epoch, args.max_epochs):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"Epoch {epoch}/{args.max_epochs - 1}")
        logging.info(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Training
        train_loss, train_label_loss, train_change_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Log training metrics
        wandb.log({
            'train/loss': train_loss,
            'train/loss_label': train_label_loss,
            'train/loss_change': train_change_loss,
            'train/acc_label': train_metrics['acc_label'],
            'train/acc_change': train_metrics['acc_change'],
            'train/acc_avg': train_metrics['acc_avg'],
            'train/f1_label': train_metrics['f1_label'],
            'train/f1_change': train_metrics['f1_change'],
            'train/f1_avg': train_metrics['f1_avg'],
            'train/lr': scheduler.get_last_lr()[0],
            'epoch': epoch
        })

        logging.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['acc_avg']:.4f}")

        # Validation (every eval_interval epochs)
        if (epoch + 1) % args.eval_interval == 0:
            val_results = validate(model, val_loader, criterion, device, epoch)

            if len(val_results) == 6:
                val_loss, val_label_loss, val_change_loss, val_metrics, cm_label, cm_change = val_results
                all_true_label, all_pred_label, all_true_change, all_pred_change = None, None, None, None
            else:
                # Extended return with prediction lists
                val_loss, val_label_loss, val_change_loss, val_metrics, cm_label, cm_change, \
                    all_true_label, all_pred_label, all_true_change, all_pred_change = val_results

            # Log validation metrics
            wandb.log({
                'val/loss': val_loss,
                'val/loss_label': val_label_loss,
                'val/loss_change': val_change_loss,
                'val/acc_label': val_metrics['acc_label'],
                'val/acc_change': val_metrics['acc_change'],
                'val/acc_avg': val_metrics['acc_avg'],
                'val/f1_label': val_metrics['f1_label'],
                'val/f1_change': val_metrics['f1_change'],
                'val/f1_avg': val_metrics['f1_avg'],
                'epoch': epoch
            })

            # Log confusion matrices
            diagnosis_names = ['CN', 'MCI', 'Dementia']
            change_names = ['Stable', 'Conversion', 'Reversion']

            # Create confusion matrix plots
            fig_cm_diagnosis = plt.figure(figsize=(8, 6))
            plt.imshow(cm_label, interpolation='nearest', cmap='Blues')
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
                    plt.text(j, i, str(cm_label[i, j]),
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
            if all_true_label is not None and all_pred_label is not None:
                try:
                    diagnosis_report = classification_report(
                        all_true_label, all_pred_label,
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

            logging.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_metrics['acc_avg']:.4f}")

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
                'best_val_acc': best_val_acc
            }, checkpoint_path)

    # Save final model
    final_model_path = os.path.join(snapshot_path, 'final_model.pth')
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), final_model_path)

    logging.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
    wandb.finish()

    return "Training Finished!"

# 使用示例的参数类
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
        self.weight_label = 1.0  # label任务权重
        self.weight_change = 1.0  # change_label任务权重
        self.label_smoothing = 0.1

        # wandb参数
        self.wandb_project = "alzheimer-classification"
        self.exp_name = "dual-task-classification"

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

    # 假设您的模型返回两个输出 (label_logits, change_logits)
    # model = YourModel(num_classes_label=3, num_classes_change=3)

    # snapshot_path = "./checkpoints/exp1"
    # trainer_alzheimer(args, model, snapshot_path)