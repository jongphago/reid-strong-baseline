# encoding: utf-8
"""
W&B Integration Utilities
"""

import os
import wandb
from datetime import datetime


def check_wandb_logged_in():
    """W&B 로그인 상태 확인
    
    Returns:
        bool: 로그인 성공 여부
    """
    try:
        wandb_api_key = os.getenv("WANDB_API_KEY", None)
        if wandb_api_key is not None or os.path.exists(os.path.expanduser("~/.netrc")):
            return wandb.login(key=wandb_api_key)
    except wandb.errors.UsageError:
        print("Warning: W&B wasn't logged in.")
    return False


def initialize_wandb(cfg):
    """W&B 초기화
    
    Args:
        cfg: YACS 설정 객체
        
    Returns:
        wandb.run 또는 None (실패 시)
    """
    if not cfg.WANDB.ENABLE:
        return None
    
    # Check if W&B is disabled via environment variable (e.g., during sweep)
    if os.getenv("WANDB_MODE") == "disabled":
        print("W&B is disabled via WANDB_MODE environment variable (sweep mode)")
        return None
    
    if not check_wandb_logged_in():
        print("Warning: W&B login failed. Logging disabled.")
        return None
    
    try:
        # 실험 이름 생성
        time_string = datetime.now().strftime("%m%d_%H%M%S")
        run_name = cfg.WANDB.NAME if cfg.WANDB.NAME else f"train_{time_string}"
        
        # 설정을 딕셔너리로 변환
        config_dict = {
            'model': {
                'name': cfg.MODEL.NAME,
                'last_stride': cfg.MODEL.LAST_STRIDE,
                'neck': cfg.MODEL.NECK,
                'pretrain_choice': cfg.MODEL.PRETRAIN_CHOICE,
            },
            'solver': {
                'optimizer': cfg.SOLVER.OPTIMIZER_NAME,
                'base_lr': cfg.SOLVER.BASE_LR,
                'weight_decay': cfg.SOLVER.WEIGHT_DECAY,
                'max_epochs': cfg.SOLVER.MAX_EPOCHS,
                'ims_per_batch': cfg.SOLVER.IMS_PER_BATCH,
            },
            'dataset': {
                'names': cfg.DATASETS.NAMES,
                'root_dir': cfg.DATASETS.ROOT_DIR,
            }
        }
        
        # W&B 초기화
        run = wandb.init(
            project=cfg.WANDB.PROJECT,
            entity=cfg.WANDB.ENTITY if cfg.WANDB.ENTITY else None,
            name=run_name,
            config=config_dict,
            tags=list(cfg.WANDB.TAGS) if cfg.WANDB.TAGS else None,
            dir=cfg.OUTPUT_DIR if cfg.OUTPUT_DIR else ".",
            sync_tensorboard=cfg.WANDB.SYNC_TENSORBOARD,
            save_code=cfg.WANDB.SAVE_CODE
        )
        
        print(f"W&B initialized. Run: {run.name} ({run.id})")
        return run
        
    except Exception as e:
        print(f"Warning: W&B initialization failed: {e}")
        return None


def log_metrics(metrics_dict, step=None):
    """W&B에 메트릭 로깅
    
    Args:
        metrics_dict: 로깅할 메트릭 딕셔너리
        step: 스텝 번호 (선택)
    """
    if wandb.run is not None:
        wandb.log(metrics_dict, step=step)

