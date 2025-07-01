# 🚀 Alzheimer MMoE 训练恢复指南

本指南详细说明在不同训练阶段中断后如何正确恢复训练。

## 📋 目录
- [1. 预训练阶段中断恢复](#1-预训练阶段中断恢复)
- [2. 微调阶段中断恢复](#2-微调阶段中断恢复)
- [3. checkpoint文件结构](#3-checkpoint文件结构)
- [4. 常见问题与解决](#4-常见问题与解决)
- [5. 最佳实践建议](#5-最佳实践建议)

---

## 1. 预训练阶段中断恢复

### 🎯 场景描述
在**SimMIM预训练阶段**（epoch < `pretrain_epochs`）训练中断，需要从断点继续预训练。

### 💾 checkpoint文件位置
训练中断时会自动保存以下文件：
```
checkpoints/
├── swin_admoe_20250701_143022/
│   ├── checkpoint_epoch_25.pth          # 定期checkpoint
│   ├── pretrain_checkpoint.pth          # 早停保存的最佳预训练模型
│   └── training.log
```

### 🔧 恢复代码示例

#### 方法1: 从定期checkpoint恢复
```python
import torch
from trainer import trainer_alzheimer_mmoe
from models import SwinTransformerV2_AlzheimerMMoE

# 创建参数对象
args = Args()
args.seed = 42
args.max_epochs = 100
args.pretrain_epochs = 50  # 确保与原训练一致
args.eval_interval = 10
args.save_interval = 20

# 🔥 关键：设置resume路径
args.resume = "checkpoints/swin_admoe_20250701_143022/checkpoint_epoch_25.pth"

# 其他训练参数
args.base_lr = 1e-4
args.weight_decay = 1e-4
args.mask_ratio = 0.6
args.patience = 10

# 创建模型
model = SwinTransformerV2_AlzheimerMMoE(
    img_size=256,
    patch_size=4,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    num_classes_diagnosis=3,
    num_classes_change=3,
    use_clinical_prior=True,
    # ... 其他参数
)

# 🚀 恢复训练
trainer_alzheimer_mmoe(args, model, "checkpoints/resume_pretrain")
```

#### 方法2: 从早停checkpoint恢复
```python
args.resume = "checkpoints/swin_admoe_20250701_143022/pretrain_checkpoint.pth"
```

### 📊 预期日志输出
```
[14:30:22.123] Resuming training from checkpoint...
[14:30:22.456] Loading pretrained weights from: checkpoints/.../checkpoint_epoch_25.pth
[14:30:22.789] Loading from epoch 25, phase: Pretrain (SimMIM)
[14:30:23.012] Original checkpoint contains 142 parameters
[14:30:23.234] Found 8 decoder parameters
[14:30:23.456] Parameter matching analysis:
[14:30:23.678]   - Matched parameters: 142
[14:30:23.890]   - Missing parameters: 0
[14:30:24.012] Resumed from epoch 26
[14:30:24.234] Optimizer state loaded from checkpoint
[14:30:24.456] Scheduler state loaded from checkpoint

==================================================
MODEL COMPONENTS ANALYSIS - INITIAL
==================================================
  decoder             :  2,362,368 params ✓ Active
  head_diagnosis      :      2,307 params ✓ Active
  # ... 其他组件

Training plan:
- Pretraining epochs: 50 (SimMIM reconstruction)
- Finetuning epochs: 50 (Classification)
- Starting from epoch: 26
```

---

## 2. 微调阶段中断恢复

### 🎯 场景描述
在**分类微调阶段**（epoch >= `pretrain_epochs`）训练中断，需要从断点继续微调。

### 💾 checkpoint文件位置
```
checkpoints/
├── swin_admoe_20250701_143022/
│   ├── checkpoint_epoch_75.pth          # 定期checkpoint（微调阶段）
│   ├── finetune_checkpoint.pth          # 早停保存的最佳微调模型
│   ├── best_model.pth                   # 验证集最佳模型
│   └── final_model.pth                  # 训练完成的最终模型
```

### 🔧 恢复代码示例

#### 方法1: 从定期checkpoint恢复
```python
import torch
from trainer import trainer_alzheimer_mmoe
from models import SwinTransformerV2_AlzheimerMMoE

# 创建参数对象
args = Args()
args.seed = 42
args.max_epochs = 100
args.pretrain_epochs = 50  # 确保与原训练一致
args.eval_interval = 10
args.save_interval = 20

# 🔥 关键：设置resume路径（微调阶段的checkpoint）
args.resume = "checkpoints/swin_admoe_20250701_143022/checkpoint_epoch_75.pth"

# 任务权重
args.weight_diagnosis = 1.0
args.weight_change = 1.0
args.label_smoothing = 0.1

# 创建模型（注意：decoder会在切换时自动移除）
model = SwinTransformerV2_AlzheimerMMoE(
    img_size=256,
    patch_size=4,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    num_classes_diagnosis=3,
    num_classes_change=3,
    use_clinical_prior=True,
    # ... 其他参数
)

# 🚀 恢复训练
trainer_alzheimer_mmoe(args, model, "checkpoints/resume_finetune")
```

#### 方法2: 从最佳模型恢复
```python
# 如果想从验证集最佳模型继续训练
args.resume = "checkpoints/swin_admoe_20250701_143022/best_model.pth"
```

### 📊 预期日志输出
```
[16:45:12.123] Resuming training from checkpoint...
[16:45:12.456] Loading pretrained weights from: checkpoints/.../checkpoint_epoch_75.pth
[16:45:12.789] Loading from epoch 75, phase: Finetune (Classification)
[16:45:13.012] Original checkpoint contains 134 parameters
[16:45:13.234] Found 0 decoder parameters  # ← 注意：微调checkpoint已无decoder
[16:45:13.456] Parameter matching analysis:
[16:45:13.678]   - Matched parameters: 134
[16:45:13.890]   - Missing parameters: 0
[16:45:14.012] Resumed from epoch 76
[16:45:14.234] Optimizer state loaded from checkpoint
[16:45:14.456] Scheduler state loaded from checkpoint

==================================================
MODEL COMPONENTS ANALYSIS - INITIAL  
==================================================
  clinical_encoder    :    233,216 params ✓ Active
  clinical_fusion     :  1,772,546 params ✓ Active
  head_diagnosis      :      2,307 params ✓ Active
  head_change         :      2,307 params ✓ Active
  # 注意：没有decoder组件

Training plan:
- Pretraining epochs: 50 (SimMIM reconstruction)  
- Finetuning epochs: 50 (Classification)
- Starting from epoch: 76  # ← 直接进入微调阶段
```

---

## 3. checkpoint文件结构

### 🗂️ 完整checkpoint内容
```python
checkpoint = {
    'epoch': 75,                          # 当前epoch
    'model_state_dict': state_dict,       # 模型权重（已过滤decoder）
    'optimizer_state_dict': opt_state,    # 优化器状态  
    'scheduler_state_dict': sched_state,  # 学习率调度器状态
    'best_val_acc': 0.8234,              # 最佳验证准确率
    'is_pretrain': False,                 # 当前是否为预训练阶段
    'phase': 'Finetune (Classification)', # 当前训练阶段描述
    'decoder_removed': True               # decoder是否已移除
}
```

### 🔍 如何检查checkpoint信息
```python
import torch

# 加载并检查checkpoint
checkpoint = torch.load("checkpoint_epoch_75.pth", map_location='cpu')

print(f"Epoch: {checkpoint['epoch']}")
print(f"Phase: {checkpoint['phase']}")  
print(f"Best accuracy: {checkpoint['best_val_acc']:.4f}")
print(f"Decoder removed: {checkpoint.get('decoder_removed', False)}")
print(f"Model parameters: {len(checkpoint['model_state_dict'])}")

# 检查是否包含decoder权重
decoder_keys = [k for k in checkpoint['model_state_dict'].keys() 
                if k.startswith('decoder.')]
print(f"Decoder parameters: {len(decoder_keys)}")
```

---

## 4. 常见问题与解决

### ❌ 问题1: 权重不匹配
```
RuntimeError: Error(s) in loading state_dict for SwinTransformerV2_AlzheimerMMoE:
Missing key(s) in state_dict: "decoder.0.weight", "decoder.0.bias"
```

**🔧 解决方案:**
```python
# 检查checkpoint是否来自微调阶段
checkpoint = torch.load(resume_path, map_location='cpu')
if checkpoint.get('decoder_removed', False):
    print("✓ Loading from finetune checkpoint (no decoder)")
else:
    print("⚠️ Loading from pretrain checkpoint (has decoder)")
```

### ❌ 问题2: epoch数量不匹配
```
# 假设原训练：pretrain_epochs=50, max_epochs=100
# 但resume时设置：pretrain_epochs=40, max_epochs=90
```

**🔧 解决方案:**
```python
# 确保训练参数与原始训练一致
args.pretrain_epochs = 50  # 必须与原训练相同
args.max_epochs = 100      # 必须与原训练相同
```

### ❌ 问题3: 学习率调度器状态异常
```
Warning: scheduler state loaded but step count mismatch
```

**🔧 解决方案:**
```python
# 方法1: 重新创建scheduler（丢失历史状态）
# 在trainer中会自动处理

# 方法2: 检查checkpoint完整性
checkpoint = torch.load(resume_path)
if 'scheduler_state_dict' in checkpoint:
    print("✓ Scheduler state available")
else:
    print("⚠️ Scheduler state missing, will recreate")
```

---

## 5. 最佳实践建议

### 🎯 训练策略
1. **定期保存**: 设置合适的 `save_interval`（建议10-20 epochs）
2. **早停保护**: 启用 `patience` 避免过拟合
3. **备份重要checkpoint**: 手动备份关键阶段的权重

### 📁 文件管理
```bash
# 推荐的checkpoint目录结构
checkpoints/
├── experiment_name_timestamp/
│   ├── checkpoint_epoch_*.pth     # 定期checkpoint
│   ├── pretrain_checkpoint.pth    # 预训练最佳
│   ├── finetune_checkpoint.pth    # 微调最佳  
│   ├── best_model.pth             # 全局最佳
│   ├── final_model.pth            # 最终模型
│   ├── training.log               # 训练日志
│   └── config.yaml                # 配置备份
```

### 🔧 代码模板
```python
def create_resume_args(original_checkpoint_dir, resume_epoch=None):
    """创建恢复训练的参数"""
    args = Args()
    
    # 从原始训练日志或config恢复参数
    # args.pretrain_epochs = ...
    # args.max_epochs = ...
    
    if resume_epoch:
        args.resume = f"{original_checkpoint_dir}/checkpoint_epoch_{resume_epoch}.pth"
    else:
        # 自动找到最新的checkpoint
        import glob
        checkpoints = glob.glob(f"{original_checkpoint_dir}/checkpoint_epoch_*.pth")
        latest = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        args.resume = latest
        
    return args

# 使用示例
args = create_resume_args("checkpoints/swin_admoe_20250701_143022")
trainer_alzheimer_mmoe(args, model, "checkpoints/resumed_training")
```

### ⚡ 性能优化
```python
# 恢复训练时的优化建议
args.eval_interval = 5     # 更频繁的验证
args.save_interval = 10    # 更频繁的保存
args.patience = 15         # 适当增加耐心值

# 如果GPU内存紧张
args.batch_size = 16       # 减小batch size
args.accumulation_steps = 2 # 使用梯度累积
```

---

## 🎉 总结

| 场景 | checkpoint路径 | 关键设置 | 预期行为 |
|------|----------------|----------|----------|
| **预训练中断** | `checkpoint_epoch_X.pth`<br/>(X < pretrain_epochs) | `args.resume = path` | 继续SimMIM预训练 |
| **微调中断** | `checkpoint_epoch_X.pth`<br/>(X >= pretrain_epochs) | `args.resume = path` | 继续分类微调 |
| **最佳模型** | `best_model.pth` | `args.resume = path` | 从最佳点继续 |

记住：**resume会完整恢复训练状态**，包括epoch、optimizer、scheduler等，确保训练的连续性！