"""
Description: 
Author: JeffreyJ
Date: 2025/7/5
LastEditTime: 2025/7/5 14:37
Version: 1.0
"""
"""
Description: Single Task Training Script for Alzheimer's Disease Classification
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:22
Version: 2.0 - Single Task Support
"""
# !/usr/bin/env python
"""
Alzheimer's Disease Single-Task Classification Training Script with SimMIM Pretraining
- Pretrain Phase: SimMIM reconstruction task
- Finetune Phase: Single classification task
  - Binary: CN(0) vs AD(1)
  - Three-class: CN(0), MCI(1), AD(2)
  - Custom: Any number of classes
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime

# Import custom modules
from config_single_task import get_config, get_single_task_config
from models.build_single_task import build_model
from trainer_single_task import trainer_alzheimer_single_task
from logger import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
import logging

# 忽略特定的警告信息
warnings.filterwarnings("ignore", message=".*Fused window process.*")
warnings.filterwarnings("ignore", message=".*Tutel.*")
logging.getLogger("models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_args():
    """Parse command line arguments for single task training"""
    parser = argparse.ArgumentParser('Alzheimer Single-Task Classification Training with SimMIM Pretraining')

    # ============================================================================
    # Basic settings
    # ============================================================================
    parser.add_argument('--cfg', type=str,
                        default=r'configs/swin_admoe/swin_admoe_tiny_ptft_256_single_task.yaml',
                        metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs",
                        default=None, nargs='+')

    # ============================================================================
    # Data settings
    # ============================================================================
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

    # ============================================================================
    # Model settings
    # ============================================================================
    parser.add_argument('--pretrained', help='pretrained weight path')
    parser.add_argument('--resume', help='resume from checkpoint')

    # ============================================================================
    # Single Task specific settings
    # ============================================================================
    parser.add_argument('--single-task', action='store_true',
                        help='Enable single task mode')
    parser.add_argument('--task-type', type=str, default='binary',
                        choices=['binary', 'diagnosis', 'three_class', 'custom'],
                        help='Type of classification task')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (2 for binary, 3 for three-class, etc.)')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['CN', 'AD'],
                        help='Names of the classes')

    # Quick task setup options
    parser.add_argument('--binary-classification', action='store_true',
                        help='Quick setup for binary classification (CN vs AD)')
    parser.add_argument('--three-class', action='store_true',
                        help='Quick setup for three-class classification (CN vs MCI vs AD)')
    parser.add_argument('--cn-vs-ad', action='store_true',
                        help='Binary classification: CN vs AD')
    parser.add_argument('--cn-vs-mci', action='store_true',
                        help='Binary classification: CN vs MCI')
    parser.add_argument('--mci-vs-ad', action='store_true',
                        help='Binary classification: MCI vs AD')

    # ============================================================================
    # Training settings
    # ============================================================================
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')

    # Distributed training - Fix for LOCAL_RANK issue
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')
    else:
        # For PyTorch 2.x, make local_rank optional and default to 0
        parser.add_argument("--local_rank", type=int, default=0, help='local rank for distributed training')

    # ============================================================================
    # Optimization settings
    # ============================================================================
    parser.add_argument('--base-lr', type=float, help='base learning rate')
    parser.add_argument('--weight-decay', type=float, help='weight decay')
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")

    # ============================================================================
    # GPU settings
    # ============================================================================
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable-amp', action='store_true', help='Disable automatic mixed precision training')

    # ============================================================================
    # WandB settings
    # ============================================================================
    parser.add_argument('--wandb-project', type=str, default='alzheimer-single-task',
                        help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, help='wandb run name')
    parser.add_argument('--wandb-offline', action='store_true', help='disable wandb online sync')

    # ============================================================================
    # Loss and training settings
    # ============================================================================
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='label smoothing factor')
    parser.add_argument('--class-weights', type=float, nargs='+',
                        help='class weights for imbalanced dataset')

    # ============================================================================
    # SimMIM specific settings
    # ============================================================================
    parser.add_argument('--mask-ratio', type=float, default=0.6,
                        help='mask ratio for SimMIM pretraining')
    parser.add_argument('--norm-target', action='store_true', default=True,
                        help='normalize target for SimMIM')
    parser.add_argument('--norm-target-patch-size', type=int, default=47,
                        help='patch size for target normalization')

    # ============================================================================
    # Training phase control
    # ============================================================================
    parser.add_argument('--pretrain-epochs', type=int, help='number of pretraining epochs')
    parser.add_argument('--skip-pretrain', action='store_true',
                        help='skip pretraining phase and go directly to finetuning')
    parser.add_argument('--pretrain-only', action='store_true',
                        help='only run pretraining phase')
    parser.add_argument('--finetune-only', action='store_true',
                        help='only run finetuning phase (requires pretrained weights)')

    # ============================================================================
    # Clinical prior settings
    # ============================================================================
    parser.add_argument('--use-clinical-prior', action='store_true', default=False,
                        help='use clinical prior information')
    parser.add_argument('--disable-clinical-prior', action='store_true',
                        help='disable clinical prior for fair benchmark comparison')
    parser.add_argument('--fusion-stage', type=int, default=2,
                        help='which stage to fuse clinical prior')
    parser.add_argument('--fusion-type', type=str, default='adaptive',
                        choices=['adaptive', 'concat', 'add', 'hadamard'],
                        help='fusion strategy for clinical prior')

    # ============================================================================
    # Benchmark and evaluation settings
    # ============================================================================
    parser.add_argument('--benchmark-mode', action='store_true',
                        help='enable benchmark mode for fair comparison')
    parser.add_argument('--save-predictions', action='store_true',
                        help='save model predictions for analysis')
    parser.add_argument('--eval-interval', type=int, default=10,
                        help='evaluation interval')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='early stopping patience')

    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training"""
    # Check if distributed training is actually being used
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")

        # Only initialize distributed if world_size > 1
        if world_size > 1:
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
            dist.barrier()
        else:
            # Single GPU, no need for distributed
            rank = 0
            world_size = 1
    else:
        # Single GPU training
        rank = 0
        world_size = 1
        print("Single GPU training mode")

    return rank, world_size


def set_random_seed(seed, deterministic=True):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_task_config(args):
    """Setup task-specific configuration based on arguments"""
    # Quick task setup
    if args.binary_classification or args.cn_vs_ad:
        args.single_task = True
        args.task_type = 'binary'
        args.num_classes = 2
        args.class_names = ['CN', 'AD']
    elif args.cn_vs_mci:
        args.single_task = True
        args.task_type = 'binary'
        args.num_classes = 2
        args.class_names = ['CN', 'MCI']
    elif args.mci_vs_ad:
        args.single_task = True
        args.task_type = 'binary'
        args.num_classes = 2
        args.class_names = ['MCI', 'AD']
    elif args.three_class:
        args.single_task = True
        args.task_type = 'three_class'
        args.num_classes = 3
        args.class_names = ['CN', 'MCI', 'AD']

    # Auto-detect if single task should be enabled
    if args.task_type != 'binary' or args.num_classes != 2:
        args.single_task = True

    # Set default class names if not provided
    if not hasattr(args, 'class_names') or len(args.class_names) != args.num_classes:
        if args.num_classes == 2 and args.task_type == 'binary':
            args.class_names = ['CN', 'AD']
        elif args.num_classes == 3:
            args.class_names = ['CN', 'MCI', 'AD']
        else:
            args.class_names = [f'Class_{i}' for i in range(args.num_classes)]

    # Benchmark mode adjustments
    if args.benchmark_mode:
        args.disable_clinical_prior = True
        args.skip_pretrain = True
        args.label_smoothing = 0.0

    return args


def prepare_config(args):
    """Prepare configuration"""
    # Setup task configuration first
    args = setup_task_config(args)

    # Always load from config file
    config = get_config(args)

    # Override config with command line arguments
    config.defrost()

    # Basic overrides
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if args.weight_decay:
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.accumulation_steps > 1:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.disable_amp:
        config.AMP_ENABLE = False

    # Single task specific overrides
    if args.single_task:
        config.SINGLE_TASK.ENABLED = True
        config.MODEL.TYPE = 'swin_single_task'
        config.SINGLE_TASK.TASK_TYPE = args.task_type
        config.SINGLE_TASK.NUM_CLASSES = args.num_classes
        config.SINGLE_TASK.CLASS_NAMES = args.class_names
        config.MODEL.NUM_CLASSES = args.num_classes

        # Update model config
        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            config.MODEL.SWIN_SINGLE_TASK.TASK_TYPE = args.task_type
            config.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = args.num_classes
            config.MODEL.SWIN_SINGLE_TASK.CLASS_NAMES = args.class_names

    # SimMIM specific overrides
    if args.mask_ratio:
        config.MODEL.SIMMIM.MASK_RATIO = args.mask_ratio
    if args.norm_target is not None:
        config.MODEL.SIMMIM.NORM_TARGET.ENABLE = args.norm_target
    if args.norm_target_patch_size:
        config.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = args.norm_target_patch_size
    if args.pretrain_epochs:
        config.TRAIN.PRETRAIN_EPOCHS = args.pretrain_epochs
    if args.skip_pretrain:
        config.TRAIN.PRETRAIN_EPOCHS = 0
    if args.pretrain_only:
        config.TRAIN.EPOCHS = config.TRAIN.PRETRAIN_EPOCHS

    # Clinical prior overrides
    if args.use_clinical_prior:
        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            config.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR = True
    if args.disable_clinical_prior:
        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            config.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR = False
        elif hasattr(config.MODEL, 'SWIN_ADMOE'):
            config.MODEL.SWIN_ADMOE.USE_CLINICAL_PRIOR = False
    if args.fusion_stage:
        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            config.MODEL.SWIN_SINGLE_TASK.FUSION_STAGE = args.fusion_stage
    if args.fusion_type:
        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            config.MODEL.SWIN_SINGLE_TASK.FUSION_TYPE = args.fusion_type

    # Training settings
    if args.label_smoothing is not None:
        config.MODEL.LABEL_SMOOTHING = args.label_smoothing
    if args.eval_interval:
        config.EVAL.INTERVAL = args.eval_interval
    if args.early_stop_patience:
        config.EARLY_STOP.PATIENCE = args.early_stop_patience

    # Benchmark mode settings
    if args.benchmark_mode:
        config.SINGLE_TASK.BENCHMARK.ENABLED = True
        config.MODEL.LABEL_SMOOTHING = 0.0
        config.TRAIN.PRETRAIN_EPOCHS = 0
        config.AUG.MIXUP = 0.0
        config.AUG.CUTMIX = 0.0
        config.AUG.COLOR_JITTER = 0.0

    # Set output directory based on config
    config.OUTPUT = config.OUTPUT if config.OUTPUT else args.output

    # Set tag
    if not args.tag and hasattr(config, 'TAG'):
        args.tag = config.TAG
    elif not args.tag:
        if args.single_task:
            args.tag = f"{args.task_type}_{args.num_classes}class_{'_'.join(args.class_names)}"
        else:
            args.tag = "default"
    config.TAG = args.tag

    config.freeze()
    return config

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Setup distributed training
    rank, world_size = setup_distributed()

    # Prepare config
    config = prepare_config(args)

    # Set random seed
    seed = config.SEED + rank
    set_random_seed(seed, deterministic=args.deterministic)

    # Create output directory
    os.makedirs(config.OUTPUT, exist_ok=True)

    # Create logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=rank, name=f"{config.MODEL.NAME}")

    # Log config
    if rank == 0:
        path = os.path.join(config.OUTPUT, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")
        logger.info(config.dump())

    # Log task information
    if hasattr(config, 'SINGLE_TASK') and config.SINGLE_TASK.ENABLED:
        logger.info("=" * 80)
        logger.info("SINGLE TASK CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Task type: {config.SINGLE_TASK.TASK_TYPE}")
        logger.info(f"Number of classes: {config.SINGLE_TASK.NUM_CLASSES}")
        logger.info(f"Class names: {config.SINGLE_TASK.CLASS_NAMES}")
        logger.info(f"Model type: {config.MODEL.TYPE}")

        if hasattr(config.MODEL, 'SWIN_SINGLE_TASK'):
            logger.info(f"Clinical prior: {config.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR}")
            if config.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR:
                logger.info(f"Fusion stage: {config.MODEL.SWIN_SINGLE_TASK.FUSION_STAGE}")
                logger.info(f"Fusion type: {config.MODEL.SWIN_SINGLE_TASK.FUSION_TYPE}")

        if hasattr(config, 'SINGLE_TASK') and hasattr(config.SINGLE_TASK, 'BENCHMARK'):
            if config.SINGLE_TASK.BENCHMARK.ENABLED:
                logger.info("Benchmark mode: ENABLED")
                logger.info(f"Benchmark dataset: {config.SINGLE_TASK.BENCHMARK.DATASET}")
                logger.info(f"Benchmark metrics: {config.SINGLE_TASK.BENCHMARK.METRICS}")

        logger.info("=" * 80)

    # Build model
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    # Calculate model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_parameters:,}")

    # Log expert configuration for single task
    if hasattr(config, 'SINGLE_TASK') and config.SINGLE_TASK.ENABLED:
        if hasattr(model, 'layers') and len(model.layers) > 0:
            # Check first layer's first block for MoE info
            first_block = model.layers[0].blocks[0]
            if hasattr(first_block, 'mmoe'):
                logger.info(f"Expert configuration:")
                logger.info(f"  Number of experts: {first_block.mmoe.num_experts}")
                logger.info(f"  Expert names: {first_block.mmoe.expert_names}")

    # Load pretrained weights if specified
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False)

    # Move model to GPU
    model.cuda()

    # Create trainer arguments
    trainer_args = argparse.Namespace(
        # Basic settings
        seed=config.SEED,
        output_dir=config.OUTPUT,
        model_name=config.MODEL.NAME,
        tag=config.TAG,

        # Task settings
        task_type=config.SINGLE_TASK.TASK_TYPE if hasattr(config,
                                                          'SINGLE_TASK') and config.SINGLE_TASK.ENABLED else 'binary',
        num_classes=config.MODEL.NUM_CLASSES,
        class_names=config.SINGLE_TASK.CLASS_NAMES if hasattr(config,
                                                              'SINGLE_TASK') and config.SINGLE_TASK.ENABLED else ['CN',
                                                                                                                  'AD'],

        # Data settings
        data_path=config.DATA.DATA_PATH,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        img_size=config.DATA.IMG_SIZE,

        # Training settings
        max_epochs=config.TRAIN.EPOCHS,
        eval_interval=config.EVAL.INTERVAL,
        save_interval=config.SAVE_FREQ,

        # Optimization settings
        base_lr=config.TRAIN.BASE_LR,
        min_lr=config.TRAIN.MIN_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,

        # Early stopping
        patience=config.EARLY_STOP.PATIENCE,

        # WandB settings
        wandb_project=args.wandb_project or (
            config.WANDB.PROJECT if hasattr(config, 'WANDB') else f'alzheimer-single-task'),
        wandb_run_name=args.wandb_run_name or f"{config.MODEL.NAME}_{config.TAG}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        wandb_offline=args.wandb_offline,

        # Warmup设置
        warmup_epochs=getattr(config.TRAIN, 'WARMUP_EPOCHS', 5),
        warmup_lr=getattr(config.TRAIN, 'WARMUP_LR', 1e-6),

        # SimMIM预训练设置
        pretrain_epochs=getattr(config.TRAIN, 'PRETRAIN_EPOCHS', config.TRAIN.EPOCHS // 3),
        mask_ratio=getattr(config.MODEL.SIMMIM, 'MASK_RATIO', 0.6),
        patch_size=getattr(
            config.MODEL.SWIN_SINGLE_TASK if hasattr(config.MODEL, 'SWIN_SINGLE_TASK') else config.MODEL.SWIN_ADMOE,
            'PATCH_SIZE', 4),
        norm_target=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'ENABLE', True),
        norm_target_patch_size=getattr(config.MODEL.SIMMIM.NORM_TARGET, 'PATCH_SIZE', 47),

        # Config object
        config=config,

        # Distributed settings
        rank=rank,
        world_size=world_size,
        local_rank=args.local_rank,
    )

    # Add data-specific attributes
    trainer_args.DATA = config.DATA
    trainer_args.AUG = config.AUG
    trainer_args.MODEL = config.MODEL
    trainer_args.TRAIN = config.TRAIN

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        trainer_args.start_epoch = checkpoint['epoch'] + 1

        # 检查是否有训练阶段信息
        if 'phase' in checkpoint:
            logger.info(f"Resumed from {checkpoint['phase']} phase")
        if 'task_type' in checkpoint:
            logger.info(f"Resumed task type: {checkpoint['task_type']}")
        if 'num_classes' in checkpoint:
            logger.info(f"Resumed num classes: {checkpoint['num_classes']}")

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        trainer_args.start_epoch = 0

    # Evaluation only mode
    if args.eval:
        logger.info("Evaluation mode")
        # TODO: Implement evaluation function
        raise NotImplementedError("Evaluation mode not implemented yet")

    # Log training plan
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PLAN")
    logger.info("=" * 60)

    pretrain_epochs = trainer_args.pretrain_epochs
    total_epochs = trainer_args.max_epochs
    finetune_epochs = total_epochs - pretrain_epochs

    logger.info(f"Task: {trainer_args.task_type}")
    logger.info(f"Classes: {trainer_args.class_names}")
    logger.info(f"Number of classes: {trainer_args.num_classes}")

    if pretrain_epochs > 0:
        logger.info(f"\nPhase 1 - SimMIM Pretraining:")
        logger.info(f"  Epochs: 0 - {pretrain_epochs - 1} ({pretrain_epochs} epochs)")
        logger.info(f"  Task: Self-supervised reconstruction")
        logger.info(f"  Mask ratio: {trainer_args.mask_ratio}")
        logger.info(f"  Expert assignment: Based on class labels")

        logger.info(f"\nPhase 2 - Classification Finetuning:")
        logger.info(f"  Epochs: {pretrain_epochs} - {total_epochs - 1} ({finetune_epochs} epochs)")
        logger.info(f"  Task: {trainer_args.task_type} classification")
        logger.info(f"  Expert gating: Learned adaptive gating")
    else:
        logger.info(f"Single Phase - Classification Training:")
        logger.info(f"  Epochs: 0 - {total_epochs - 1} ({total_epochs} epochs)")
        logger.info(f"  Task: {trainer_args.task_type} classification")
        logger.info(f"  Note: Skipping SimMIM pretraining")

    logger.info("=" * 60)

    # Start training
    logger.info("Start training")
    if hasattr(config, 'SINGLE_TASK') and config.SINGLE_TASK.ENABLED:
        trainer_alzheimer_single_task(trainer_args, model, config.OUTPUT)
    else:
        # Fallback to dual task trainer if needed
        from trainer import trainer_alzheimer_mmoe
        trainer_alzheimer_mmoe(trainer_args, model, config.OUTPUT)


if __name__ == '__main__':
    main()