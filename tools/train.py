# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import gc
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_train_with_center
from modeling import build_model
from layers import make_loss, make_loss_with_center
from solver import make_optimizer, make_optimizer_with_center, WarmupMultiStepLR

from utils.logger import setup_logger
from utils.wandb_utils import initialize_wandb


def train(cfg):
    # W&B initialization
    wandb_run = initialize_wandb(cfg)
    
    # prepare dataset
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, num_classes)

    try:
        if cfg.MODEL.IF_WITH_CENTER == 'no':
            print('Train without center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
            optimizer = make_optimizer(cfg, model)
            # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
            #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

            loss_func = make_loss(cfg, num_classes)     # modified by gu

            # Add for using self trained model
            if cfg.MODEL.PRETRAIN_CHOICE == 'self':
                start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
                print('Start epoch:', start_epoch)
                path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
                print('Path to the checkpoint of optimizer:', path_to_optimizer)
                model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
                optimizer.load_state_dict(torch.load(path_to_optimizer))
                scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
                start_epoch = 0
                scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            else:
                print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

            arguments = {}

            do_train(
                cfg,
                model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch,    # add for using self trained model
                wandb_run       # add for W&B logging
            )
        elif cfg.MODEL.IF_WITH_CENTER == 'yes':
            print('Train with center loss, the loss type is', cfg.MODEL.METRIC_LOSS_TYPE)
            loss_func, center_criterion = make_loss_with_center(cfg, num_classes)  # modified by gu
            optimizer, optimizer_center = make_optimizer_with_center(cfg, model, center_criterion)
            # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
            #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

            arguments = {}

            # Add for using self trained model
            if cfg.MODEL.PRETRAIN_CHOICE == 'self':
                start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
                print('Start epoch:', start_epoch)
                path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
                print('Path to the checkpoint of optimizer:', path_to_optimizer)
                path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
                print('Path to the checkpoint of center_param:', path_to_center_param)
                path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
                print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
                model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
                optimizer.load_state_dict(torch.load(path_to_optimizer))
                center_criterion.load_state_dict(torch.load(path_to_center_param))
                optimizer_center.load_state_dict(torch.load(path_to_optimizer_center))
                scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
            elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
                start_epoch = 0
                scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                              cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
            else:
                print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

            do_train_with_center(
                cfg,
                model,
                center_criterion,
                train_loader,
                val_loader,
                optimizer,
                optimizer_center,
                scheduler,      # modify for using self trained model
                loss_func,
                num_query,
                start_epoch,    # add for using self trained model
                wandb_run       # add for W&B logging
            )
        else:
            print("Unsupported value for cfg.MODEL.IF_WITH_CENTER {}, only support yes or no!\n".format(cfg.MODEL.IF_WITH_CENTER))
    
    finally:
        # GPU 메모리 정리 (subprocess에서 실행)
        print("\n" + "="*80)
        print("[train.py] Cleaning up GPU memory before exit...")
        print("="*80)
        try:
            if torch.cuda.is_available():
                # 모든 CUDA 디바이스의 캐시 비우기
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # 메모리 상태 출력
                print(f"[train.py] GPU memory status after cleanup:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # Python 가비지 컬렉션
            gc.collect()
            print("[train.py] Memory cleanup completed successfully")
            print("="*80 + "\n")
        except Exception as cleanup_error:
            print(f"[train.py] WARNING: GPU cleanup failed: {cleanup_error}")
            print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
