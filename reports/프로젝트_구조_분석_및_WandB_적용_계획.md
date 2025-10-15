# Reid-Strong-Baseline 프로젝트 구조 분석 및 W&B 적용 계획

## 목차

1. [서론](#서론)
2. [프로젝트 구조 분석](#프로젝트-구조-분석)
   - 2.1 [디렉토리 구조](#21-디렉토리-구조)
   - 2.2 [핵심 컴포넌트](#22-핵심-컴포넌트)
   - 2.3 [설정 시스템](#23-설정-시스템)
   - 2.4 [훈련 파이프라인](#24-훈련-파이프라인)
   - 2.5 [로깅 메커니즘](#25-로깅-메커니즘)
3. [TAO Pytorch Backend와의 비교](#tao-pytorch-backend와의-비교)
4. [W&B 적용 계획](#wb-적용-계획)
   - 4.1 [구현 단계](#41-구현-단계)
   - 4.2 [코드 수정 사항](#42-코드-수정-사항)
   - 4.3 [설정 추가](#43-설정-추가)
5. [구현 가이드](#구현-가이드)
6. [참조 목록](#참조-목록)

---

## 서론

본 보고서는 `reid-strong-baseline` 프로젝트의 구조를 분석하고, TAO Pytorch Backend에서 구현된 W&B 연동 방식을 적용하기 위한 계획을 수립합니다.

**분석 대상**:
- 프로젝트 위치: `~/projects/reid-strong-baseline`
- 기반 프레임워크: PyTorch Ignite
- 설정 관리: YACS (Yet Another Configuration System)

**목표**:
- TAO Pytorch Backend의 W&B 구현 패턴을 reid-strong-baseline에 적용
- 최소한의 코드 수정으로 W&B 로깅 기능 추가
- 기존 훈련 파이프라인 유지

---

## 프로젝트 구조 분석

### 2.1 디렉토리 구조

```
reid-strong-baseline/
├── config/
│   ├── __init__.py
│   └── defaults.py              # YACS 기반 기본 설정
├── configs/
│   ├── baseline.yml
│   ├── softmax_triplet.yml
│   ├── softmax_triplet_with_center.yml
│   └── softmax.yml
├── data/
│   ├── build.py                 # 데이터로더 빌드
│   ├── collate_batch.py
│   ├── datasets/                # Market1501, DukeMTMC 등
│   ├── samplers/                # Triplet sampler
│   └── transforms/              # 데이터 증강
├── engine/
│   ├── inference.py
│   └── trainer.py               # PyTorch Ignite 훈련 엔진
├── layers/
│   ├── center_loss.py
│   └── triplet_loss.py
├── modeling/
│   ├── baseline.py              # Baseline 모델
│   └── backbones/               # ResNet, SENet, IBN-Net
├── solver/
│   ├── build.py                 # 옵티마이저 빌드
│   └── lr_scheduler.py          # 학습률 스케줄러
├── tools/
│   ├── train.py                 # 훈련 엔트리포인트
│   ├── test.py                  # 테스트 엔트리포인트
│   └── convert_to_onnx.py       # ONNX 변환
├── utils/
│   ├── logger.py                # 로깅 유틸리티
│   ├── reid_metric.py           # CMC, mAP 메트릭
│   └── re_ranking.py            # Re-ranking
└── pyproject.toml               # 의존성 관리
```

### 2.2 핵심 컴포넌트

#### 2.2.1 설정 시스템 (YACS)

Reid-strong-baseline은 YACS를 사용하여 설정을 관리합니다:

```python
# config/defaults.py에서 설정 정의
_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NAME = 'resnet50'
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 3e-4
# ... 기타 설정
```

**특징**:
- 계층적 설정 구조
- YAML 파일과 CLI 인자를 통한 오버라이드
- `cfg.merge_from_file()`, `cfg.merge_from_list()` 지원

#### 2.2.2 훈련 엔진 (PyTorch Ignite)

PyTorch Ignite를 사용한 이벤트 기반 훈련:

```python
# engine/trainer.py
trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
evaluator = create_supervised_evaluator(model, metrics, device)

# 이벤트 핸들러 등록
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    logger.info("Epoch[{}] Loss: {:.3f}".format(...))

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(engine):
    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info("mAP: {:.1%}".format(mAP))
```

**주요 이벤트**:
- `Events.STARTED`: 훈련 시작
- `Events.EPOCH_STARTED`: 에포크 시작 (LR 스케줄링)
- `Events.ITERATION_COMPLETED`: 스텝 완료 (로깅)
- `Events.EPOCH_COMPLETED`: 에포크 완료 (검증, 체크포인트)

#### 2.2.3 로깅 메커니즘

현재 Python 기본 logging 모듈 사용:

```python
# utils/logger.py
def setup_logger(name, save_dir, distributed_rank):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러
    ch = logging.StreamHandler(stream=sys.stdout)
    
    # 파일 핸들러
    fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
    
    return logger
```

**현재 로깅 내용**:
- 훈련 손실, 정확도, 학습률
- 에포크 시간, 처리 속도
- 검증 메트릭 (CMC, mAP)

#### 2.2.4 체크포인트 관리

Ignite의 ModelCheckpoint 사용:

```python
checkpointer = ModelCheckpoint(
    output_dir, 
    cfg.MODEL.NAME, 
    checkpoint_period, 
    n_saved=10, 
    require_empty=False
)

trainer.add_event_handler(
    Events.EPOCH_COMPLETED, 
    checkpointer, 
    {'model': model, 'optimizer': optimizer}
)
```

### 2.3 설정 시스템

#### 2.3.1 기본 설정 파일

`config/defaults.py`에서 모든 설정 정의:

```python
# 모델 설정
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'resnet50'
_C.MODEL.LAST_STRIDE = 1
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.NECK = 'bnneck'
_C.MODEL.IF_WITH_CENTER = 'no'
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.IF_LABELSMOOTH = 'on'

# 데이터셋 설정
_C.DATASETS.NAMES = ('market1501')
_C.DATASETS.ROOT_DIR = ('./data')

# 훈련 설정
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.IMS_PER_BATCH = 64

# 출력 설정
_C.OUTPUT_DIR = ""
```

#### 2.3.2 실험 설정 파일 예시

`configs/baseline.yml`:

```yaml
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: '/path/to/resnet50-19c8e357.pth'
  LAST_STRIDE: 2
  NECK: 'no'

SOLVER:
  OPTIMIZER_NAME: 'Adam'
  MAX_EPOCHS: 120
  BASE_LR: 0.00035
  STEPS: [40, 70]
  GAMMA: 0.1

OUTPUT_DIR: "/path/to/output"
```

### 2.4 훈련 파이프라인

#### 2.4.1 훈련 엔트리포인트

`tools/train.py`의 전체 흐름:

```python
def train(cfg):
    # 1. 데이터 준비
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    
    # 2. 모델 빌드
    model = build_model(cfg, num_classes)
    
    # 3. 손실 함수 및 옵티마이저
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(...)
    loss_func = make_loss(cfg, num_classes)
    
    # 4. 훈련 실행
    do_train(cfg, model, train_loader, val_loader, 
             optimizer, scheduler, loss_func, num_query, start_epoch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # 설정 로드 및 병합
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    # 로거 설정
    logger = setup_logger("reid_baseline", output_dir, 0)
    
    # 훈련 시작
    train(cfg)
```

#### 2.4.2 훈련 루프

`engine/trainer.py`의 `do_train()` 함수:

```python
def do_train(cfg, model, train_loader, val_loader, 
             optimizer, scheduler, loss_fn, num_query, start_epoch):
    
    # Ignite 엔진 생성
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device)
    evaluator = create_supervised_evaluator(model, metrics, device)
    
    # 메트릭 추적
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    
    # 이벤트 핸들러
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Loss: {:.3f}, Acc: {:.3f}, Lr: {:.2e}"
                       .format(engine.state.epoch, 
                               engine.state.metrics['avg_loss'],
                               engine.state.metrics['avg_acc'],
                               scheduler.get_lr()[0]))
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r-1]))
    
    # 훈련 실행
    trainer.run(train_loader, max_epochs=epochs)
```

### 2.5 로깅 메커니즘

#### 2.5.1 현재 로깅 구조

```python
# utils/logger.py
logger = logging.getLogger("reid_baseline")
logger.addHandler(StreamHandler)  # 콘솔
logger.addHandler(FileHandler)    # log.txt 파일
```

**로깅되는 정보**:

**훈련 중**:
- Epoch, Iteration 번호
- Loss, Accuracy
- Learning Rate
- 처리 속도 (samples/s)

**검증 중**:
- mAP (Mean Average Precision)
- CMC Rank-1, 5, 10

#### 2.5.2 로깅 위치

- **콘솔 출력**: `sys.stdout`
- **파일 출력**: `{OUTPUT_DIR}/log.txt`

---

## TAO Pytorch Backend와의 비교

### 주요 차이점

| 항목 | reid-strong-baseline | TAO Pytorch Backend |
|------|---------------------|---------------------|
| **훈련 프레임워크** | PyTorch Ignite | PyTorch Lightning |
| **설정 관리** | YACS (CN) | Dataclass + Hydra |
| **로깅** | Python logging | PyTorch Lightning Logger |
| **이벤트 시스템** | Ignite Events | Lightning Hooks |
| **W&B 통합** | ❌ 없음 | ✅ WandbLogger 사용 |
| **체크포인트** | Ignite ModelCheckpoint | Lightning Checkpoint |
| **메트릭 추적** | Ignite Metrics | Lightning Metrics |

### 유사점

| 항목 | 공통점 |
|------|--------|
| **모델 구조** | Baseline with BNNeck |
| **손실 함수** | Cross Entropy + Triplet + Center Loss |
| **데이터셋** | Market1501, DukeMTMC-reID |
| **백본** | ResNet, SENet, IBN-Net |
| **메트릭** | CMC, mAP, Re-ranking |
| **증강** | Random Erasing, Random Flip |

### 설정 구조 비교

**reid-strong-baseline (YACS)**:
```python
_C = CN()
_C.MODEL = CN()
_C.MODEL.NAME = 'resnet50'
_C.SOLVER = CN()
_C.SOLVER.BASE_LR = 3e-4
```

**TAO Pytorch Backend (Dataclass)**:
```python
@dataclass
class ModelConfig:
    name: str = "resnet50"

@dataclass
class SolverConfig:
    base_lr: float = 3e-4
```

### 로깅 메커니즘 비교

**reid-strong-baseline (Ignite Events)**:
```python
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    logger.info("Loss: {:.3f}".format(loss))
```

**TAO Pytorch Backend (Lightning Hooks)**:
```python
def training_step(self, batch, batch_idx):
    loss = self.loss_func(...)
    self.log("train_loss", loss, on_step=True, on_epoch=True)
    return loss
```

---

## W&B 적용 계획

### 4.1 구현 단계

#### Phase 1: 기본 W&B 통합 (필수)

**목표**: 훈련 및 검증 메트릭을 W&B에 자동 로깅

**작업 내용**:
1. 설정에 W&B 섹션 추가
2. W&B 초기화 함수 구현
3. Ignite 이벤트에서 W&B 로깅 연결
4. 기본 메트릭 로깅 (loss, acc, lr, mAP, CMC)

**예상 소요 시간**: 2-3시간

#### Phase 2: 고급 기능 (선택)

**목표**: Sweep, 이미지 로깅 등 고급 기능 추가

**작업 내용**:
1. W&B Sweep 스크립트 작성
2. 샘플 이미지 로깅
3. 모델 아티팩트 저장
4. 커스텀 차트 및 테이블

**예상 소요 시간**: 4-6시간

### 4.2 코드 수정 사항

#### 4.2.1 설정 추가

**파일**: `config/defaults.py`

```python
# 기존 설정 끝에 추가
# ---------------------------------------------------------------------------- #
# W&B Config
# ---------------------------------------------------------------------------- #
_C.WANDB = CN()
_C.WANDB.ENABLE = True
_C.WANDB.PROJECT = "reid-strong-baseline"
_C.WANDB.ENTITY = ""  # 팀/사용자 이름 (선택)
_C.WANDB.NAME = ""  # 실험 이름 (선택, 자동 생성)
_C.WANDB.TAGS = []  # 태그 리스트
_C.WANDB.SYNC_TENSORBOARD = False
_C.WANDB.SAVE_CODE = False
```

#### 4.2.2 W&B 유틸리티 함수

**새 파일**: `utils/wandb_utils.py`

```python
"""W&B 통합 유틸리티"""
import os
import wandb
from datetime import datetime

def check_wandb_logged_in():
    """W&B 로그인 상태 확인"""
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
    
    if not check_wandb_logged_in():
        print("Warning: W&B login failed. Logging disabled.")
        return None
    
    try:
        # 실험 이름 생성
        time_string = datetime.now().strftime("%m%d_%H%M%S")
        run_name = cfg.WANDB.NAME if cfg.WANDB.NAME else f"train_{time_string}"
        
        # W&B 초기화
        run = wandb.init(
            project=cfg.WANDB.PROJECT,
            entity=cfg.WANDB.ENTITY if cfg.WANDB.ENTITY else None,
            name=run_name,
            config=cfg,
            tags=cfg.WANDB.TAGS if cfg.WANDB.TAGS else None,
            dir=cfg.OUTPUT_DIR,
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
```

#### 4.2.3 훈련 엔트리포인트 수정

**파일**: `tools/train.py`

```python
# 기존 import에 추가
from utils.wandb_utils import initialize_wandb, log_metrics

def train(cfg):
    # W&B 초기화 (기존 코드 앞에 추가)
    wandb_run = initialize_wandb(cfg)
    
    # 기존 데이터 로더, 모델 빌드 코드
    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model(cfg, num_classes)
    
    # ... 기존 코드 유지 ...
    
    if cfg.MODEL.IF_WITH_CENTER == 'no':
        do_train(
            cfg,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            loss_func,
            num_query,
            start_epoch,
            wandb_run  # W&B run 전달
        )
    # ... 나머지 코드
```

#### 4.2.4 훈련 엔진 수정

**파일**: `engine/trainer.py`

```python
# 기존 import에 추가
from utils.wandb_utils import log_metrics

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        wandb_run=None  # 추가
):
    # ... 기존 코드 유지 ...
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        
        if ITER % log_period == 0:
            # 기존 로깅
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                       .format(engine.state.epoch, ITER, len(train_loader),
                               engine.state.metrics['avg_loss'], 
                               engine.state.metrics['avg_acc'],
                               scheduler.get_lr()[0]))
            
            # W&B 로깅 추가
            if wandb_run is not None:
                log_metrics({
                    'train/loss': engine.state.metrics['avg_loss'],
                    'train/acc': engine.state.metrics['avg_acc'],
                    'train/lr': scheduler.get_lr()[0],
                    'epoch': engine.state.epoch,
                }, step=ITER + (engine.state.epoch - 1) * len(train_loader))
        
        if len(train_loader) == ITER:
            ITER = 0
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            
            # 기존 로깅
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            
            # W&B 로깅 추가
            if wandb_run is not None:
                log_metrics({
                    'val/mAP': mAP,
                    'val/cmc_rank_1': cmc[0],
                    'val/cmc_rank_5': cmc[4],
                    'val/cmc_rank_10': cmc[9],
                    'epoch': engine.state.epoch,
                }, step=engine.state.epoch)
    
    # ... 나머지 코드 유지 ...
```

#### 4.2.5 Center Loss 포함 훈련 수정

**파일**: `engine/trainer.py`의 `do_train_with_center()` 함수도 동일한 방식으로 수정

```python
def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        wandb_run=None  # 추가
):
    # do_train()과 동일한 W&B 로깅 추가
    # ...
```

### 4.3 설정 추가

#### 4.3.1 실험 설정 파일 업데이트

**파일**: `configs/baseline.yml` (또는 새 파일 생성)

```yaml
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: '/path/to/resnet50-19c8e357.pth'
  LAST_STRIDE: 1
  NECK: 'bnneck'

SOLVER:
  OPTIMIZER_NAME: 'Adam'
  MAX_EPOCHS: 120
  BASE_LR: 0.00035
  STEPS: [40, 70]

# W&B 설정 추가
WANDB:
  ENABLE: True
  PROJECT: "reid-strong-baseline"
  ENTITY: "your-username"  # 선택
  NAME: "market1501_resnet50_baseline"
  TAGS: ["baseline", "market1501", "resnet50"]
  SYNC_TENSORBOARD: False
  SAVE_CODE: False

OUTPUT_DIR: "./log/market1501/baseline_with_wandb"
```

#### 4.3.2 환경 변수 설정

**옵션 1: 쉘 스크립트**

```bash
#!/bin/bash
# train_with_wandb.sh

export WANDB_API_KEY="your_api_key_here"

python tools/train.py \
    --config_file='configs/baseline.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')" \
    OUTPUT_DIR "('./log/market1501/wandb_test')"
```

**옵션 2: .env 파일**

```bash
# .env
WANDB_API_KEY=your_api_key_here
```

```python
# tools/train.py에 추가
from dotenv import load_dotenv
load_dotenv()  # main() 함수 시작 부분
```

---

## 구현 가이드

### Step 1: 의존성 추가

**파일**: `pyproject.toml`

```toml
[project]
dependencies = [
    "pytorch-ignite==0.1.2",
    "torch>=1.11.0,<2.0.0",
    "torchvision>=0.12.0,<0.15.0",
    "yacs>=0.1.8",
    "wandb>=0.18.0",  # 추가
    "python-dotenv>=0.19.0",  # 선택 (환경 변수 관리)
]
```

설치:
```bash
cd ~/projects/reid-strong-baseline
uv sync
# 또는
pip install wandb python-dotenv
```

### Step 2: 설정 파일 수정

1. `config/defaults.py` 끝에 W&B 섹션 추가
2. `configs/baseline.yml`에 W&B 설정 추가

### Step 3: W&B 유틸리티 작성

1. `utils/wandb_utils.py` 생성
2. `check_wandb_logged_in()` 구현
3. `initialize_wandb()` 구현
4. `log_metrics()` 구현

### Step 4: 훈련 코드 수정

1. `tools/train.py` 수정
   - W&B 초기화 추가
   - `wandb_run` 전달

2. `engine/trainer.py` 수정
   - `do_train()` 함수에 `wandb_run` 파라미터 추가
   - `log_training_loss` 핸들러에 W&B 로깅 추가
   - `log_validation_results` 핸들러에 W&B 로깅 추가
   - `do_train_with_center()` 함수도 동일하게 수정

### Step 5: 환경 설정

```bash
# W&B 로그인
wandb login

# 또는 환경 변수 설정
export WANDB_API_KEY="your_api_key_here"
```

### Step 6: 테스트 실행

```bash
# 짧은 테스트 (5 에포크)
python tools/train.py \
    --config_file='configs/baseline.yml' \
    SOLVER.MAX_EPOCHS 5 \
    SOLVER.EVAL_PERIOD 5 \
    WANDB.ENABLE True \
    WANDB.PROJECT "reid-test" \
    OUTPUT_DIR "./log/wandb_test"
```

### Step 7: 전체 훈련 실행

```bash
# 전체 훈련
python tools/train.py \
    --config_file='configs/baseline.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')" \
    WANDB.ENABLE True \
    WANDB.PROJECT "reid-strong-baseline" \
    WANDB.NAME "market1501_resnet50_baseline" \
    OUTPUT_DIR "./log/market1501/baseline_with_wandb"
```

### 추가 기능 (선택)

#### Sweep 기능 구현

**새 파일**: `tools/sweep.py`

```python
"""W&B Sweep for hyperparameter tuning"""
import argparse
import os
import wandb
import yaml
import subprocess
import sys

def train_sweep():
    """Sweep agent가 호출하는 훈련 함수"""
    run = wandb.init()
    
    # Sweep config에서 하이퍼파라미터 가져오기
    lr = wandb.config.lr
    weight_decay = wandb.config.weight_decay
    batch_size = wandb.config.batch_size
    
    # 훈련 명령 구성
    cmd = [
        sys.executable,
        "tools/train.py",
        "--config_file=configs/baseline.yml",
        f"SOLVER.BASE_LR {lr}",
        f"SOLVER.WEIGHT_DECAY {weight_decay}",
        f"SOLVER.IMS_PER_BATCH {batch_size}",
        f"OUTPUT_DIR ./log/sweep/{run.id}",
        f"WANDB.NAME {run.name}"
    ]
    
    # 훈련 실행
    subprocess.run(cmd, check=True)
    wandb.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    
    # Sweep 설정 로드
    with open(args.sweep_config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Sweep 생성
    sweep_id = wandb.sweep(sweep_config, project="reid-strong-baseline")
    
    # Agent 실행
    wandb.agent(sweep_id, function=train_sweep, count=args.count)

if __name__ == "__main__":
    main()
```

**Sweep 설정 파일**: `configs/sweep_config.yaml`

```yaml
name: reid_hyperparameter_search
method: bayes
metric:
  name: val/mAP
  goal: maximize
parameters:
  lr:
    values: [0.00025, 0.00035, 0.0005, 0.0007]
  weight_decay:
    values: [0.0001, 0.0005, 0.001]
  batch_size:
    values: [32, 64, 96]
count: 20
```

**Sweep 실행**:

```bash
python tools/sweep.py --sweep-config configs/sweep_config.yaml --count 20
```

#### 이미지 로깅

**파일**: `engine/trainer.py`에 추가

```python
import wandb
import torchvision

@trainer.on(Events.EPOCH_COMPLETED)
def log_sample_images(engine):
    """검증 데이터 샘플 이미지 로깅"""
    if wandb_run is not None and engine.state.epoch % eval_period == 0:
        # 검증 데이터에서 샘플 가져오기
        dataiter = iter(val_loader)
        images, pids, camids = next(dataiter)
        
        # 이미지 그리드 생성
        grid = torchvision.utils.make_grid(images[:16], nrow=4, normalize=True)
        
        # W&B에 로깅
        wandb_run.log({
            "val/sample_images": wandb.Image(grid),
            "epoch": engine.state.epoch
        })
```

### 체크리스트

- [ ] `pyproject.toml`에 wandb 의존성 추가
- [ ] `config/defaults.py`에 W&B 설정 추가
- [ ] `utils/wandb_utils.py` 생성
- [ ] `tools/train.py` 수정 (W&B 초기화)
- [ ] `engine/trainer.py` 수정 (로깅 추가)
  - [ ] `do_train()` 함수
  - [ ] `do_train_with_center()` 함수
- [ ] `configs/baseline.yml`에 W&B 설정 추가
- [ ] WANDB_API_KEY 환경 변수 설정
- [ ] 짧은 테스트 실행 (5 에포크)
- [ ] 전체 훈련 실행
- [ ] (선택) Sweep 기능 구현
- [ ] (선택) 이미지 로깅 추가

---

## 참조 목록

### Reid-Strong-Baseline 파일

- `/home/jongphago/projects/reid-strong-baseline/config/defaults.py`
  - 전체 설정 정의

- `/home/jongphago/projects/reid-strong-baseline/tools/train.py`
  - 훈련 엔트리포인트
  - `train()` 함수: 25-117행
  - `main()` 함수: 119-158행

- `/home/jongphago/projects/reid-strong-baseline/engine/trainer.py`
  - `create_supervised_trainer()`: 20-54행
  - `do_train()`: 134-208행
  - `do_train_with_center()`: 211-290행
  - 훈련 로깅: 176-187행
  - 검증 로깅: 198-206행

- `/home/jongphago/projects/reid-strong-baseline/utils/logger.py`
  - `setup_logger()`: 12-30행

- `/home/jongphago/projects/reid-strong-baseline/configs/baseline.yml`
  - 설정 파일 예시

- `/home/jongphago/projects/reid-strong-baseline/pyproject.toml`
  - 의존성 정의

### TAO Pytorch Backend 참조

이전 보고서 참조: `/home/jongphago/projects/tao_pytorch_backend/W&B_구현_분석_보고서.md`

주요 참조 파일:
- `nvidia_tao_pytorch/core/common_config.py:126-136` - WandBConfig
- `nvidia_tao_pytorch/core/mlops/wandb.py` - W&B 유틸리티
- `nvidia_tao_pytorch/core/initialize_experiments.py:76-89` - 초기화

### 외부 문서

- [PyTorch Ignite Documentation](https://pytorch.org/ignite/)
- [YACS Documentation](https://github.com/rbgirshick/yacs)
- [Weights & Biases Python Library](https://docs.wandb.ai/ref/python)
- [W&B with PyTorch Ignite](https://docs.wandb.ai/guides/integrations/ignite)

---

**보고서 생성일**: 2025-10-15  
**타겟 프로젝트**: `/home/jongphago/projects/reid-strong-baseline`  
**참조 프로젝트**: `/home/jongphago/projects/tao_pytorch_backend`

