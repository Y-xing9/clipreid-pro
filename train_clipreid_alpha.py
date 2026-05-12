
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid_alpha_v2 import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Alpha-CLIP-ReID v2 Training")
    parser.add_argument(
        "--config_file",
        default="configs/person/vit_clipreid_market.yml",
        help="path to config file",
        type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument("--local_rank", default=0, type=int)

    parser.add_argument(
        "--use_saliency",
        action='store_true',
        default=True,
        help="Use saliency-guided attention (default: True)"
    )
    parser.add_argument(
        "--sal_alpha",
        type=float,
        default=0.5,
        help="Override saliency alpha scaling (soft mask strength)."
    )
    parser.add_argument(
        "--quick_epochs",
        type=int,
        default=None,
        help="Optional: force both Stage1 and Stage2 epochs to this value for fast sweeps."
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Log incoming saliency alpha argument
    print(f"[cfg] requested sal_alpha={args.sal_alpha}")

    # Optional quick sweep: shrink epochs to speed up runs (best-effort; ignores missing keys)
    if args.quick_epochs is not None:
        cfg.defrost()
        try:
            cfg.SOLVER.STAGE1.MAX_EPOCHS = args.quick_epochs
        except Exception:
            pass
        try:
            cfg.SOLVER.STAGE2.MAX_EPOCHS = args.quick_epochs
        except Exception:
            pass
        cfg.freeze()
    else:
        cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    
    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("=" * 80)
    logger.info("Alpha-CLIP-ReID v2 Training")
    logger.info("Saliency-Guided Attention Mechanism")
    logger.info("=" * 80)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info("Use saliency: {}".format(args.use_saliency))
    logger.info(args)
    logger.info(f"Saliency alpha (requested): {args.sal_alpha}")

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 构建数据加载器
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # 构建 Alpha-v2 模型
    logger.info("=" * 80)
    logger.info("Building Alpha-CLIP-ReID v2 model...")
    logger.info("=" * 80)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num, use_saliency=args.use_saliency)

    # Apply saliency alpha override if provided
    if args.sal_alpha is not None:
        if hasattr(model, 'set_saliency_alpha'):
            model.set_saliency_alpha(args.sal_alpha)
        elif hasattr(model, 'saliency_alpha'):
            model.saliency_alpha = args.sal_alpha
        else:
            # Fallback: attach attribute for downstream use if model checks it dynamically
            model.saliency_alpha = args.sal_alpha
        logger.info(f"Override saliency alpha to {args.sal_alpha}")
    else:
        logger.info("Saliency alpha: using model default")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model Statistics:")
    logger.info(f"  - Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"  - Trainable parameters: {trainable_params / 1e6:.2f}M")
    logger.info(f"  - Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    # 损失函数
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # Stage 1: Prompt Learning
    logger.info("=" * 80)
    logger.info("Stage 1: Prompt Learning")
    logger.info("=" * 80)

    optimizer_1stage = make_optimizer_1stage(cfg, model)
    scheduler_1stage = create_scheduler(
        optimizer_1stage,
        num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
        lr_min=cfg.SOLVER.STAGE1.LR_MIN,
        warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
        warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
        noise_range=None
    )

    do_train_stage1(
        cfg,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank
    )

    # Stage 2: End-to-End Fine-tuning
    logger.info("=" * 80)
    logger.info("Stage 2: End-to-End Fine-tuning with Saliency Guidance")
    logger.info("=" * 80)


    
    if args.use_saliency and hasattr(model, 'saliency_detector'):
        logger.info("Setting higher learning rate for saliency detector...")

        # 分组参数
        saliency_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'saliency_detector' in name and param.requires_grad:
                saliency_params.append(param)
            elif param.requires_grad:
                other_params.append(param)

        # 创建参数组
        param_groups = [
            {'params': saliency_params, 'lr': cfg.SOLVER.STAGE2.BASE_LR * 10},
            {'params': other_params, 'lr': cfg.SOLVER.STAGE2.BASE_LR}
        ]

        logger.info(f"  - Saliency detector LR: {cfg.SOLVER.STAGE2.BASE_LR * 10:.6f}")
        logger.info(f"  - Other parameters LR: {cfg.SOLVER.STAGE2.BASE_LR:.6f}")

        # 使用自定义参数组的优化器
        optimizer_2stage = torch.optim.Adam(
            param_groups,
            weight_decay=cfg.SOLVER.STAGE2.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        if cfg.MODEL.IF_WITH_CENTER == 'yes':
            optimizer_center_2stage = torch.optim.SGD(
                center_criterion.parameters(),
                lr=cfg.SOLVER.STAGE2.CENTER_LR
            )
        else:
            
            optimizer_center_2stage = torch.optim.SGD(
                center_criterion.parameters(),
                lr=cfg.SOLVER.STAGE2.CENTER_LR if hasattr(cfg.SOLVER.STAGE2, 'CENTER_LR') else 0.5
            )
            logger.info("  - Center loss disabled, but created dummy optimizer for compatibility")
    else:
        # 使用原始优化器
        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)

    scheduler_2stage = WarmupMultiStepLR(
        optimizer_2stage,
        cfg.SOLVER.STAGE2.STEPS,
        cfg.SOLVER.STAGE2.GAMMA,
        cfg.SOLVER.STAGE2.WARMUP_FACTOR,
        cfg.SOLVER.STAGE2.WARMUP_ITERS,
        cfg.SOLVER.STAGE2.WARMUP_METHOD
    )

    do_train_stage2(
        cfg,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query,
        args.local_rank
    )

    logger.info("=" * 80)
    logger.info("Training Completed!")
    logger.info("=" * 80)

