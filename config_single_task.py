"""
Description: 
Author: JeffreyJ
Date: 2025/7/5
LastEditTime: 2025/7/5 14:23
Version: 1.0
"""
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified for Single Task Alzheimer Classification
# --------------------------------------------------------

import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
_C.DATA.BATCH_SIZE_PRETRAIN = 256  # 预训练阶段batch size
_C.DATA.BATCH_SIZE_FINETUNE = 224  # 微调阶段batch size
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = 0
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Swin Transformer MoE parameters
_C.MODEL.SWIN_MOE = CN()
_C.MODEL.SWIN_MOE.PATCH_SIZE = 4
_C.MODEL.SWIN_MOE.IN_CHANS = 3
_C.MODEL.SWIN_MOE.EMBED_DIM = 96
_C.MODEL.SWIN_MOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MOE.WINDOW_SIZE = 7
_C.MODEL.SWIN_MOE.MLP_RATIO = 4.
_C.MODEL.SWIN_MOE.QKV_BIAS = True
_C.MODEL.SWIN_MOE.QK_SCALE = 0
_C.MODEL.SWIN_MOE.APE = False
_C.MODEL.SWIN_MOE.PATCH_NORM = True
_C.MODEL.SWIN_MOE.MLP_FC2_BIAS = True
_C.MODEL.SWIN_MOE.INIT_STD = 0.02
_C.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_MOE.MOE_BLOCKS = [[-1], [-1], [-1], [-1]]
_C.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS = 1
_C.MODEL.SWIN_MOE.TOP_VALUE = 1
_C.MODEL.SWIN_MOE.CAPACITY_FACTOR = 1.25
_C.MODEL.SWIN_MOE.COSINE_ROUTER = False
_C.MODEL.SWIN_MOE.NORMALIZE_GATE = False
_C.MODEL.SWIN_MOE.USE_BPR = True
_C.MODEL.SWIN_MOE.IS_GSHARD_LOSS = False
_C.MODEL.SWIN_MOE.GATE_NOISE = 1.0
_C.MODEL.SWIN_MOE.COSINE_ROUTER_DIM = 256
_C.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T = 0.5
_C.MODEL.SWIN_MOE.MOE_DROP = 0.0
_C.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT = 0.01

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# [SimMIM] Norm target during training
_C.MODEL.SIMMIM = CN()
# Mask ratio for SimMIM pretraining
_C.MODEL.SIMMIM.MASK_RATIO = 0.6
# Normalization target settings
_C.MODEL.SIMMIM.NORM_TARGET = CN()
_C.MODEL.SIMMIM.NORM_TARGET.ENABLE = False
_C.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = 47

# ============================================================================
# Single Task Swin Transformer with Alzheimer MoE parameters
# ============================================================================
_C.MODEL.SWIN_SINGLE_TASK = CN()
_C.MODEL.SWIN_SINGLE_TASK.PATCH_SIZE = 4
_C.MODEL.SWIN_SINGLE_TASK.IN_CHANS = 3
_C.MODEL.SWIN_SINGLE_TASK.EMBED_DIM = 96
_C.MODEL.SWIN_SINGLE_TASK.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_SINGLE_TASK.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_SINGLE_TASK.WINDOW_SIZE = 7
_C.MODEL.SWIN_SINGLE_TASK.MLP_RATIO = 4.
_C.MODEL.SWIN_SINGLE_TASK.QKV_BIAS = True
_C.MODEL.SWIN_SINGLE_TASK.APE = False
_C.MODEL.SWIN_SINGLE_TASK.PATCH_NORM = True
_C.MODEL.SWIN_SINGLE_TASK.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_SINGLE_TASK.SHIFT_MLP_RATIO = 1.0
_C.MODEL.SWIN_SINGLE_TASK.IS_PRETRAIN = True
_C.MODEL.SWIN_SINGLE_TASK.USE_SHIFTED_LAST_LAYER = False

# Single task specific parameters
_C.MODEL.SWIN_SINGLE_TASK.TASK_TYPE = 'diagnosis'  # 'diagnosis' or 'change' or 'custom'
_C.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = 3  # 2 for binary, 3 for CN/MCI/AD
_C.MODEL.SWIN_SINGLE_TASK.CLASS_NAMES = ['CN', 'MCI', 'AD']  # 类别名称，用于日志记录

# Clinical prior parameters
_C.MODEL.SWIN_SINGLE_TASK.USE_CLINICAL_PRIOR = True
_C.MODEL.SWIN_SINGLE_TASK.PRIOR_DIM = 3
_C.MODEL.SWIN_SINGLE_TASK.PRIOR_HIDDEN_DIM = 128
_C.MODEL.SWIN_SINGLE_TASK.FUSION_STAGE = 2
_C.MODEL.SWIN_SINGLE_TASK.FUSION_TYPE = 'adaptive'

# Expert configuration parameters
_C.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG = CN()
_C.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG.AUTO_CONFIG = True  # 是否自动根据类别数配置专家
_C.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG.SHARED_EXPERT_WEIGHT = 0.4  # 共享专家权重
_C.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG.SPECIFIC_EXPERT_WEIGHT = 0.6  # 特定专家权重

# ============================================================================
# 保留原有双任务配置以兼容
# ============================================================================
_C.MODEL.SWIN_ADMOE = CN()
_C.MODEL.SWIN_ADMOE.PATCH_SIZE = 4
_C.MODEL.SWIN_ADMOE.IN_CHANS = 3
_C.MODEL.SWIN_ADMOE.EMBED_DIM = 96
_C.MODEL.SWIN_ADMOE.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_ADMOE.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_ADMOE.WINDOW_SIZE = 8
_C.MODEL.SWIN_ADMOE.MLP_RATIO = 4.
_C.MODEL.SWIN_ADMOE.QKV_BIAS = True
_C.MODEL.SWIN_ADMOE.APE = False
_C.MODEL.SWIN_ADMOE.PATCH_NORM = True
_C.MODEL.SWIN_ADMOE.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWIN_ADMOE.SHIFT_MLP_RATIO = 1.0
_C.MODEL.SWIN_ADMOE.IS_PRETRAIN = True
_C.MODEL.SWIN_ADMOE.USE_SHIFTED_LAST_LAYER = False
# Dual-task specific parameters
_C.MODEL.SWIN_ADMOE.NUM_CLASSES_DIAGNOSIS = 3
_C.MODEL.SWIN_ADMOE.NUM_CLASSES_CHANGE = 3
# Clinical prior parameters
_C.MODEL.SWIN_ADMOE.USE_CLINICAL_PRIOR = True
_C.MODEL.SWIN_ADMOE.PRIOR_DIM = 3
_C.MODEL.SWIN_ADMOE.PRIOR_HIDDEN_DIM = 128
_C.MODEL.SWIN_ADMOE.FUSION_STAGE = 2
_C.MODEL.SWIN_ADMOE.FUSION_TYPE = 'adaptive'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Pretraining epochs for SimMIM
_C.TRAIN.PRETRAIN_EPOCHS = 100
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = (0.0, 1.0)
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Loss settings - 单任务简化版
# -----------------------------------------------------------------------------
_C.LOSS = CN()
# 单任务只需要一个权重
_C.LOSS.WEIGHT = 1.0
# 保留双任务权重以兼容
_C.LOSS.WEIGHT_DIAGNOSIS = 1.0
_C.LOSS.WEIGHT_CHANGE = 1.0

# -----------------------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.INTERVAL = 10
_C.EVAL.SAVE_BEST = True
_C.EVAL.METRICS = ['accuracy', 'f1_score', 'confusion_matrix']

# -----------------------------------------------------------------------------
# Early stopping settings
# -----------------------------------------------------------------------------
_C.EARLY_STOP = CN()
_C.EARLY_STOP.ENABLE = True
_C.EARLY_STOP.PATIENCE = 5
_C.EARLY_STOP.MONITOR = 'val_loss'

# -----------------------------------------------------------------------------
# WandB settings
# -----------------------------------------------------------------------------
_C.WANDB = CN()
_C.WANDB.PROJECT = 'alzheimer-classification'
_C.WANDB.NAME = 'experiment'
_C.WANDB.TAGS = []

# -----------------------------------------------------------------------------
# Phase-specific configurations - 单任务简化版
# -----------------------------------------------------------------------------
_C.PHASES = CN()
_C.PHASES.PRETRAIN = CN()
_C.PHASES.PRETRAIN.DESCRIPTION = "SimMIM self-supervised pretraining"
_C.PHASES.PRETRAIN.LOSS_TYPE = "reconstruction"
_C.PHASES.PRETRAIN.EXPERT_ASSIGNMENT = "label_based"
_C.PHASES.PRETRAIN.EVALUATION_METRIC = "reconstruction_loss"

_C.PHASES.FINETUNE = CN()
_C.PHASES.FINETUNE.DESCRIPTION = "Single task classification finetuning"
_C.PHASES.FINETUNE.LOSS_TYPE = "classification"
_C.PHASES.FINETUNE.EXPERT_ASSIGNMENT = "learned_gating"
_C.PHASES.FINETUNE.EVALUATION_METRIC = "classification_accuracy"

# -----------------------------------------------------------------------------
# Single Task Specific Configurations
# -----------------------------------------------------------------------------
_C.SINGLE_TASK = CN()
# Task configuration
_C.SINGLE_TASK.ENABLED = False  # 是否启用单任务模式
_C.SINGLE_TASK.TASK_TYPE = 'diagnosis'  # 'diagnosis', 'change', 'custom'
_C.SINGLE_TASK.NUM_CLASSES = 3  # 类别数
_C.SINGLE_TASK.CLASS_NAMES = ['CN', 'MCI', 'AD']  # 类别名称

# Binary classification quick configs
_C.SINGLE_TASK.BINARY = CN()
_C.SINGLE_TASK.BINARY.ENABLED = False
_C.SINGLE_TASK.BINARY.POSITIVE_CLASS = 'AD'  # 正类名称
_C.SINGLE_TASK.BINARY.NEGATIVE_CLASS = 'CN'  # 负类名称

# Three-class classification quick configs
_C.SINGLE_TASK.THREE_CLASS = CN()
_C.SINGLE_TASK.THREE_CLASS.ENABLED = False
_C.SINGLE_TASK.THREE_CLASS.CLASSES = ['CN', 'MCI', 'AD']

# Expert behavior analysis
_C.SINGLE_TASK.ANALYSIS = CN()
_C.SINGLE_TASK.ANALYSIS.LOG_EXPERT_WEIGHTS = True  # 是否记录专家权重
_C.SINGLE_TASK.ANALYSIS.EXPERT_ANALYSIS_INTERVAL = 25  # 专家分析间隔
_C.SINGLE_TASK.ANALYSIS.SAVE_ATTENTION_MAPS = False  # 是否保存注意力图

# Benchmark comparison settings
_C.SINGLE_TASK.BENCHMARK = CN()
_C.SINGLE_TASK.BENCHMARK.ENABLED = False  # 是否启用benchmark模式
_C.SINGLE_TASK.BENCHMARK.DATASET = 'ADNI'  # 'ADNI', 'OASIS', 'AIBL'
_C.SINGLE_TASK.BENCHMARK.SPLIT = 'standard'  # 'standard', 'custom'
_C.SINGLE_TASK.BENCHMARK.METRICS = ['accuracy', 'sensitivity', 'specificity', 'f1', 'auc']

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # SimMIM specific argument overrides
    if hasattr(args, 'mask_ratio') and args.mask_ratio is not None:
        config.MODEL.SIMMIM.MASK_RATIO = args.mask_ratio
    if hasattr(args, 'norm_target') and args.norm_target is not None:
        config.MODEL.SIMMIM.NORM_TARGET.ENABLE = args.norm_target
    if hasattr(args, 'norm_target_patch_size') and args.norm_target_patch_size is not None:
        config.MODEL.SIMMIM.NORM_TARGET.PATCH_SIZE = args.norm_target_patch_size
    if hasattr(args, 'pretrain_epochs') and args.pretrain_epochs is not None:
        config.TRAIN.PRETRAIN_EPOCHS = args.pretrain_epochs

    # Single task specific overrides
    if hasattr(args, 'single_task') and args.single_task:
        config.SINGLE_TASK.ENABLED = True

    if hasattr(args, 'task_type') and args.task_type is not None:
        config.SINGLE_TASK.TASK_TYPE = args.task_type
        config.MODEL.SWIN_SINGLE_TASK.TASK_TYPE = args.task_type

    if hasattr(args, 'num_classes') and args.num_classes is not None:
        config.MODEL.NUM_CLASSES = args.num_classes
        config.SINGLE_TASK.NUM_CLASSES = args.num_classes
        config.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = args.num_classes

    if hasattr(args, 'binary_classification') and args.binary_classification:
        config.SINGLE_TASK.BINARY.ENABLED = True
        config.SINGLE_TASK.ENABLED = True
        config.MODEL.NUM_CLASSES = 2
        config.SINGLE_TASK.NUM_CLASSES = 2
        config.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = 2
        config.SINGLE_TASK.CLASS_NAMES = ['CN', 'AD']  # or ['Negative', 'Positive']

    if hasattr(args, 'three_class') and args.three_class:
        config.SINGLE_TASK.THREE_CLASS.ENABLED = True
        config.SINGLE_TASK.ENABLED = True
        config.MODEL.NUM_CLASSES = 3
        config.SINGLE_TASK.NUM_CLASSES = 3
        config.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = 3
        config.SINGLE_TASK.CLASS_NAMES = ['CN', 'MCI', 'AD']

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        # For PyTorch 2.x, safely get LOCAL_RANK with fallback
        config.LOCAL_RANK = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', 0)))

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_single_task_config(task_type='diagnosis', num_classes=3, img_size=256, batch_size=32):
    """
    快速创建单任务配置的辅助函数

    Args:
        task_type: 'diagnosis', 'change', 'binary', 'three_class'
        num_classes: 类别数量
        img_size: 图像大小
        batch_size: 批次大小

    Returns:
        配置对象
    """
    config = _C.clone()
    config.defrost()

    # 启用单任务模式
    config.SINGLE_TASK.ENABLED = True
    config.MODEL.TYPE = 'swin_single_task'

    # 设置任务类型和类别数
    config.SINGLE_TASK.TASK_TYPE = task_type
    config.SINGLE_TASK.NUM_CLASSES = num_classes
    config.MODEL.NUM_CLASSES = num_classes
    config.MODEL.SWIN_SINGLE_TASK.NUM_CLASSES = num_classes
    config.MODEL.SWIN_SINGLE_TASK.TASK_TYPE = task_type

    # 设置图像大小和批次大小
    config.DATA.IMG_SIZE = img_size
    config.DATA.BATCH_SIZE = batch_size

    # 根据任务类型设置类别名称
    if task_type == 'binary' or num_classes == 2:
        config.SINGLE_TASK.BINARY.ENABLED = True
        config.SINGLE_TASK.CLASS_NAMES = ['CN', 'AD']
    elif task_type == 'three_class' or num_classes == 3:
        config.SINGLE_TASK.THREE_CLASS.ENABLED = True
        config.SINGLE_TASK.CLASS_NAMES = ['CN', 'MCI', 'AD']
    else:
        config.SINGLE_TASK.CLASS_NAMES = [f'Class_{i}' for i in range(num_classes)]

    # 设置专家配置
    config.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG.AUTO_CONFIG = True

    # 医学图像的保守增强设置
    config.AUG.COLOR_JITTER = 0.0
    config.AUG.MIXUP = 0.0
    config.AUG.CUTMIX = 0.0
    config.AUG.REPROB = 0.0

    # 设置输出目录和标签
    config.TAG = f'single_task_{task_type}_{num_classes}class'
    config.OUTPUT = f'./output/single_task/{task_type}'

    # 设置wandb项目名称
    config.WANDB.PROJECT = f'alzheimer-single-task-{task_type}'
    config.WANDB.NAME = f'swin_single_task_{task_type}_{num_classes}class'
    config.WANDB.TAGS = ['single_task', task_type, f'{num_classes}_class']

    config.freeze()
    return config


if __name__ == "__main__":
    """测试单任务配置"""
    print("=" * 80)
    print("Testing Single Task Configuration")
    print("=" * 80)

    # 测试二分类配置
    print("\n1. Binary Classification Config:")
    binary_config = get_single_task_config(task_type='binary', num_classes=2)
    print(f"   Task type: {binary_config.SINGLE_TASK.TASK_TYPE}")
    print(f"   Num classes: {binary_config.MODEL.NUM_CLASSES}")
    print(f"   Class names: {binary_config.SINGLE_TASK.CLASS_NAMES}")
    print(f"   Single task enabled: {binary_config.SINGLE_TASK.ENABLED}")
    print(f"   Binary enabled: {binary_config.SINGLE_TASK.BINARY.ENABLED}")
    print(f"   Expert auto config: {binary_config.MODEL.SWIN_SINGLE_TASK.EXPERT_CONFIG.AUTO_CONFIG}")

    # 测试三分类配置
    print("\n2. Three-class Classification Config:")
    three_config = get_single_task_config(task_type='three_class', num_classes=3)
    print(f"   Task type: {three_config.SINGLE_TASK.TASK_TYPE}")
    print(f"   Num classes: {three_config.MODEL.NUM_CLASSES}")
    print(f"   Class names: {three_config.SINGLE_TASK.CLASS_NAMES}")
    print(f"   Three class enabled: {three_config.SINGLE_TASK.THREE_CLASS.ENABLED}")

    # 测试自定义配置
    print("\n3. Custom Classification Config:")
    custom_config = get_single_task_config(task_type='diagnosis', num_classes=4)
    print(f"   Task type: {custom_config.SINGLE_TASK.TASK_TYPE}")
    print(f"   Num classes: {custom_config.MODEL.NUM_CLASSES}")
    print(f"   Class names: {custom_config.SINGLE_TASK.CLASS_NAMES}")

    print("\n4. Key Configuration Sections:")
    print(f"   Model type: {binary_config.MODEL.TYPE}")
    print(f"   Wandb project: {binary_config.WANDB.PROJECT}")
    print(f"   Output dir: {binary_config.OUTPUT}")
    print(f"   Tag: {binary_config.TAG}")

    print("\nConfiguration test completed successfully!")