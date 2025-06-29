# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_transformer_moe import SwinTransformerMoE
from .swin_mlp import SwinMLP
from .swin_transformer_v2_mtad_ptft import SwinTransformerV2_AlzheimerMMoE
from .simmim import build_simmim


def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if is_pretrain:
        model = build_simmim(config)
        return model

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.SWINV2.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  ape=config.MODEL.SWINV2.APE,
                                  patch_norm=config.MODEL.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin_moe':
        model = SwinTransformerMoE(img_size=config.DATA.IMG_SIZE,
                                   patch_size=config.MODEL.SWIN_MOE.PATCH_SIZE,
                                   in_chans=config.MODEL.SWIN_MOE.IN_CHANS,
                                   num_classes=config.MODEL.NUM_CLASSES,
                                   embed_dim=config.MODEL.SWIN_MOE.EMBED_DIM,
                                   depths=config.MODEL.SWIN_MOE.DEPTHS,
                                   num_heads=config.MODEL.SWIN_MOE.NUM_HEADS,
                                   window_size=config.MODEL.SWIN_MOE.WINDOW_SIZE,
                                   mlp_ratio=config.MODEL.SWIN_MOE.MLP_RATIO,
                                   qkv_bias=config.MODEL.SWIN_MOE.QKV_BIAS,
                                   qk_scale=config.MODEL.SWIN_MOE.QK_SCALE,
                                   drop_rate=config.MODEL.DROP_RATE,
                                   drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                   ape=config.MODEL.SWIN_MOE.APE,
                                   patch_norm=config.MODEL.SWIN_MOE.PATCH_NORM,
                                   mlp_fc2_bias=config.MODEL.SWIN_MOE.MLP_FC2_BIAS,
                                   init_std=config.MODEL.SWIN_MOE.INIT_STD,
                                   use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                   pretrained_window_sizes=config.MODEL.SWIN_MOE.PRETRAINED_WINDOW_SIZES,
                                   moe_blocks=config.MODEL.SWIN_MOE.MOE_BLOCKS,
                                   num_local_experts=config.MODEL.SWIN_MOE.NUM_LOCAL_EXPERTS,
                                   top_value=config.MODEL.SWIN_MOE.TOP_VALUE,
                                   capacity_factor=config.MODEL.SWIN_MOE.CAPACITY_FACTOR,
                                   cosine_router=config.MODEL.SWIN_MOE.COSINE_ROUTER,
                                   normalize_gate=config.MODEL.SWIN_MOE.NORMALIZE_GATE,
                                   use_bpr=config.MODEL.SWIN_MOE.USE_BPR,
                                   is_gshard_loss=config.MODEL.SWIN_MOE.IS_GSHARD_LOSS,
                                   gate_noise=config.MODEL.SWIN_MOE.GATE_NOISE,
                                   cosine_router_dim=config.MODEL.SWIN_MOE.COSINE_ROUTER_DIM,
                                   cosine_router_init_t=config.MODEL.SWIN_MOE.COSINE_ROUTER_INIT_T,
                                   moe_drop=config.MODEL.SWIN_MOE.MOE_DROP,
                                   aux_loss_weight=config.MODEL.SWIN_MOE.AUX_LOSS_WEIGHT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_admoe':
        model = SwinTransformerV2_AlzheimerMMoE(
            img_size=config.DATA.IMG_SIZE,
            patch_size=getattr(config.MODEL.SWIN_ADMOE, 'PATCH_SIZE', 4),
            in_chans=getattr(config.MODEL.SWIN_ADMOE, 'IN_CHANS', 3),
            # 双任务类别数设置
            num_classes=config.MODEL.NUM_CLASSES,  # 保留兼容性
            num_classes_diagnosis=getattr(config.MODEL.SWIN_ADMOE, 'NUM_CLASSES_DIAGNOSIS', 3),
            num_classes_change=getattr(config.MODEL.SWIN_ADMOE, 'NUM_CLASSES_CHANGE', 3),
            embed_dim=getattr(config.MODEL.SWIN_ADMOE, 'EMBED_DIM', 96),
            depths=getattr(config.MODEL.SWIN_ADMOE, 'DEPTHS', [2, 2, 6, 2]),
            num_heads=getattr(config.MODEL.SWIN_ADMOE, 'NUM_HEADS', [3, 6, 12, 24]),
            window_size=getattr(config.MODEL.SWIN_ADMOE, 'WINDOW_SIZE', 7),
            mlp_ratio=getattr(config.MODEL.SWIN_ADMOE, 'MLP_RATIO', 4.),
            qkv_bias=getattr(config.MODEL.SWIN_ADMOE, 'QKV_BIAS', True),
            drop_rate=config.MODEL.DROP_RATE,
            attn_drop_rate=getattr(config.MODEL, 'ATTN_DROP_RATE', 0.),
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=layernorm,
            ape=getattr(config.MODEL.SWIN_ADMOE, 'APE', False),
            patch_norm=getattr(config.MODEL.SWIN_ADMOE, 'PATCH_NORM', True),
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            pretrained_window_sizes=getattr(config.MODEL.SWIN_ADMOE, 'PRETRAINED_WINDOW_SIZES', [0, 0, 0, 0]),
            shift_mlp_ratio=getattr(config.MODEL.SWIN_ADMOE, 'SHIFT_MLP_RATIO', 1.0),
            is_pretrain=getattr(config.MODEL.SWIN_ADMOE, 'IS_PRETRAIN', True),
            use_shifted_last_layer=getattr(config.MODEL.SWIN_ADMOE, 'USE_SHIFTED_LAST_LAYER', False),
            # ===== 新增MMoE参数 (仅基于原实现) =====
            num_experts = 4,  # 专家数量，原实现中有
            temperature_init = 1.0,  # 温度参数，原实现中forward时使用
            ## ===== 新增Clinical先验参数 =====
            use_clinical_prior=True,  # 是否使用临床先验信息 (True/False)
            prior_dim=3,  # 临床先验向量维度 (根据实际数据调整，如3维概率分布)
            prior_hidden_dim=128,  # MLP编码器的隐藏层维度 (建议: 64/128/256)
            fusion_stage=2,  # 在哪个stage后融合 (0/1/2/3，推荐1或2，越深语义越丰富)
            fusion_type='adaptive'  # 融合策略 ('adaptive'/'concat'/'add'/'hadamard')
            # - 'adaptive': 学习自适应权重（推荐）
            # - 'concat': 拼接后投影
            # - 'add': 加权相加
            # - 'hadamard': 逐元素乘积+残差
        )


    else:
        raise NotImplementedError(f"Unknown model: {model_type}")


    return model