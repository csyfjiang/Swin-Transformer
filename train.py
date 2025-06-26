"""
Description: 
Author: JeffreyJ
Date: 2025/6/25
LastEditTime: 2025/6/25 14:22
Version: 1.0
"""
# !/usr/bin/env python
"""
Alzheimer's Disease Dual-Task Classification Training Script
- Diagnosis Task: CN(1), MCI(2), Dementia(3)
- Change Task: Stable(1), Conversion(2), Reversion(3)
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
from config import get_config
from models import build_model
from trainer import trainer_alzheimer
from logger import create_logger

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('Alzheimer Dual-Task Classification Training')

    # Basic settings
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY VALUE' pairs",
                        default=None, nargs='+')

    # Data settings
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

    # Model settings
    parser.add_argument('--pretrained', help='pretrained weight path')
    parser.add_argument('--resume', help='resume from checkpoint')

    # Training settings
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


    # Optimization settings
    parser.add_argument('--base-lr', type=float, help='base learning rate')
    parser.add_argument('--weight-decay', type=float, help='weight decay')
    parser.add_argument('--accumulation-steps', type=int, default=1, help="gradient accumulation steps")

    # GPU settings
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable-amp', action='store_true', help='Disable automatic mixed precision training')

    # WandB settings
    parser.add_argument('--wandb-project', type=str, default='alzheimer-dual-task',
                        help='wandb project name')
    parser.add_argument('--wandb-run-name', type=str, help='wandb run name')
    parser.add_argument('--wandb-offline', action='store_true', help='disable wandb online sync')

    # Task weights
    parser.add_argument('--weight-diagnosis', type=float, default=1.0,
                        help='weight for diagnosis task loss')
    parser.add_argument('--weight-change', type=float, default=1.0,
                        help='weight for change task loss')

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


def prepare_config(args):
    """Prepare configuration"""
    config = get_config(args)

    # Override config with command line arguments
    if args.batch_size:
        config.defrost()
        config.DATA.BATCH_SIZE = args.batch_size
        config.freeze()

    if args.data_path:
        config.defrost()
        config.DATA.DATA_PATH = args.data_path
        config.freeze()

    if args.base_lr:
        config.defrost()
        config.TRAIN.BASE_LR = args.base_lr
        config.freeze()

    if args.weight_decay:
        config.defrost()
        config.TRAIN.WEIGHT_DECAY = args.weight_decay
        config.freeze()

    if args.accumulation_steps > 1:
        config.defrost()
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
        config.freeze()

    if args.use_checkpoint:
        config.defrost()
        config.TRAIN.USE_CHECKPOINT = True
        config.freeze()

    if args.disable_amp:
        config.defrost()
        config.TRAIN.AMP_ENABLE = False
        config.freeze()

    # Set output directory
    config.defrost()
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)
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

    # Build model
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    # Calculate model parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_parameters:,}")

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

        # Data settings
        data_path=config.DATA.DATA_PATH,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        img_size=config.DATA.IMG_SIZE,

        # Model settings
        num_classes=config.MODEL.NUM_CLASSES,

        # Training settings
        max_epochs=config.TRAIN.EPOCHS,
        eval_interval=config.EVAL.INTERVAL,
        save_interval=config.SAVE_FREQ,

        # Optimization settings
        base_lr=config.TRAIN.BASE_LR,
        min_lr=config.TRAIN.MIN_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
        label_smoothing=config.MODEL.LABEL_SMOOTHING,

        # Task weights
        weight_diagnosis=config.LOSS.WEIGHT_DIAGNOSIS,
        weight_change=config.LOSS.WEIGHT_CHANGE,

        # Early stopping
        patience=config.EARLY_STOP.PATIENCE,

        # WandB settings
        wandb_project=args.wandb_project or config.WANDB.PROJECT,
        wandb_run_name=args.wandb_run_name or f"{config.MODEL.NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        wandb_offline=args.wandb_offline,

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
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
    else:
        trainer_args.start_epoch = 0

    # Evaluation only mode
    if args.eval:
        logger.info("Evaluation mode")
        # TODO: Implement evaluation function
        raise NotImplementedError("Evaluation mode not implemented yet")

    # Start training
    logger.info("Start training")
    trainer_alzheimer(trainer_args, model, config.OUTPUT)


if __name__ == '__main__':
    main()