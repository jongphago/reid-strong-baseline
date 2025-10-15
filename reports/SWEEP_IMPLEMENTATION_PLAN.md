# W&B Sweep 구현 계획

## 목차

1. [개요](#개요)
2. [TAO Pytorch Backend Sweep 분석](#tao-pytorch-backend-sweep-분석)
3. [reid-strong-baseline 적용 계획](#reid-strong-baseline-적용-계획)
4. [구현 세부사항](#구현-세부사항)
5. [테스트 계획](#테스트-계획)

---

## 개요

reid-strong-baseline 프로젝트에 W&B Sweep 기능을 추가하여 하이퍼파라미터 자동 튜닝을 지원합니다.

**목표**:
- Grid, Random, Bayesian search 지원
- 주요 하이퍼파라미터 튜닝
- Early termination 지원
- 간단한 CLI 인터페이스

---

## TAO Pytorch Backend Sweep 분석

### 핵심 구조

1. **Sweep 스크립트**: `scripts/sweep.py`
   - argparse로 CLI 인자 파싱
   - YAML 설정 파일 로드
   - `wandb.sweep()` 생성
   - `wandb.agent()` 실행

2. **train_sweep() 함수**
   - `wandb.init()` 초기화
   - `wandb.config`에서 파라미터 가져오기
   - `subprocess`로 train.py 실행
   - 각 run마다 독립된 output 디렉토리

3. **Sweep 설정 파일**
   ```yaml
   name: sweep_name
   method: random  # grid, random, bayes
   metric:
     name: mAP
     goal: maximize
   parameters:
     lr: [1e-4, 3e-4, 1e-3]
     weight_decay: [1e-5, 5e-5, 1e-4]
   count: 10
   early_terminate:
     type: hyperband
     max_iter: 12
   ```

### 주요 기능

- CLI 인자와 YAML 설정 병합
- 기존 sweep 재개 (`--sweep_id`)
- 독립적인 결과 디렉토리
- 환경 변수 전달 (WANDB_RUN_ID, WANDB_SWEEP_ID)

---

## reid-strong-baseline 적용 계획

### 파일 구조

```
reid-strong-baseline/
├── tools/
│   ├── train.py          # 기존
│   └── sweep.py          # 신규
├── configs/
│   ├── baseline.yml      # 기존
│   └── sweep_test.yaml   # 신규
└── sweeps/               # 신규 디렉토리
    ├── sweep_lr.yaml
    └── sweep_full.yaml
```

### 구현 방식

**Option 1: TAO 스타일 (subprocess)**
- 장점: 완전한 프로세스 격리
- 단점: 오버헤드 증가

**Option 2: 직접 호출**
- 장점: 빠르고 효율적
- 단점: 메모리 누수 가능성

**선택**: Option 1 (subprocess) - 안정성 우선

---

## 구현 세부사항

### 1. tools/sweep.py

#### 주요 함수

**main()**
```python
- argparse로 CLI 인자 파싱
  - --config_file: 기본 설정 파일
  - --sweep-config: sweep 설정 YAML
  - --count: 실행 횟수
  - --method: grid/random/bayes
  - --sweep_id: 재개용 ID

- YAML에서 sweep 설정 로드
- wandb.sweep() 생성
- wandb.agent() 실행
```

**train_sweep()**
```python
- wandb.init() 초기화
- wandb.config에서 파라미터 가져오기
- 독립 디렉토리 생성
- subprocess로 tools/train.py 실행
  - 하이퍼파라미터를 CLI 인자로 전달
- wandb.finish()
```

#### CLI 인터페이스

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_test.yaml \
    --count 5
```

### 2. Sweep 설정 파일

#### sweeps/sweep_test.yaml (테스트용)

```yaml
name: reid_test_sweep
method: random
metric:
  name: val/mAP
  goal: maximize
parameters:
  base_lr: [0.0001, 0.00035, 0.0005]
  weight_decay: [0.0001, 0.0005, 0.001]
count: 3
results_dir_base: ./log/sweep_test
```

#### sweeps/sweep_lr.yaml (학습률 탐색)

```yaml
name: reid_lr_sweep
method: grid
metric:
  name: val/mAP
  goal: maximize
parameters:
  base_lr: [0.0001, 0.00025, 0.00035, 0.0005, 0.0007]
count: 5
results_dir_base: ./log/sweep_lr
```

#### sweeps/sweep_full.yaml (전체 탐색)

```yaml
name: reid_full_sweep
method: bayes
metric:
  name: val/mAP
  goal: maximize
parameters:
  base_lr: [0.0001, 0.00035, 0.0005, 0.0007]
  weight_decay: [0.0001, 0.0005, 0.001]
  warmup_factor: [0.01, 0.05, 0.1]
  margin: [0.2, 0.3, 0.4]
count: 20
results_dir_base: ./log/sweep_full
early_terminate:
  type: hyperband
  max_iter: 5
  min_iter: 2
  s: 2
  eta: 3
```

### 3. YACS 파라미터 매핑

| W&B Config | YACS Config | 설명 |
|-----------|-------------|------|
| base_lr | SOLVER.BASE_LR | 기본 학습률 |
| weight_decay | SOLVER.WEIGHT_DECAY | 가중치 감쇠 |
| warmup_factor | SOLVER.WARMUP_FACTOR | 워밍업 계수 |
| warmup_iters | SOLVER.WARMUP_ITERS | 워밍업 반복 |
| margin | SOLVER.MARGIN | 트리플릿 마진 |
| ims_per_batch | SOLVER.IMS_PER_BATCH | 배치 크기 |
| max_epochs | SOLVER.MAX_EPOCHS | 최대 에포크 |

### 4. 독립 디렉토리 구조

```
log/
└── sweep_test/
    └── sweep/
        └── {sweep_id}/
            ├── run-1/
            │   ├── log.txt
            │   └── resnet50_model_*.pth
            ├── run-2/
            └── run-3/
```

---

## 테스트 계획

### Phase 1: 빠른 테스트 (3 runs, 2 epochs)

**설정**: `sweeps/sweep_test.yaml`
```yaml
parameters:
  base_lr: [0.0001, 0.00035, 0.0005]
count: 3
max_epochs: 2  # 빠른 테스트
eval_period: 2
```

**실행**:
```bash
python tools/sweep.py \
    --config_file configs/test_wandb.yml \
    --sweep-config sweeps/sweep_test.yaml
```

**예상 시간**: 약 20-30분 (2 epochs × 3 runs)

### Phase 2: 학습률 탐색 (5 runs, 5 epochs)

**설정**: `sweeps/sweep_lr.yaml`

**실행**:
```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```

**예상 시간**: 약 2-3시간

### Phase 3: 전체 탐색 (20 runs, 120 epochs)

**설정**: `sweeps/sweep_full.yaml`

**실행**:
```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_full.yaml
```

**예상 시간**: 수 시간 ~ 수 일

---

## 참고

### TAO Pytorch Backend 파일

- `nvidia_tao_pytorch/cv/re_identification/scripts/sweep.py`
- `nvidia_tao_pytorch/cv/re_identification/sweeps/reid_market1501_resnet_sweep.yaml`

### W&B 문서

- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)
- [Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)
- [Early Termination](https://docs.wandb.ai/guides/sweeps/early-termination)

