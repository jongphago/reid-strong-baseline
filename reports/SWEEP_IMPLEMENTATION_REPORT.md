# W&B Sweep 구현 완료 보고서

## 개요

reid-strong-baseline 프로젝트에 W&B Sweep 기능이 성공적으로 구현되었습니다.

**구현 날짜**: 2025-10-15  
**구현 범위**: 하이퍼파라미터 자동 튜닝 (Grid/Random/Bayesian search)  
**테스트 상태**: 코드 구현 완료, 사용자 테스트 대기

---

## 구현 내용

### 1. Sweep 스크립트

**파일**: `tools/sweep.py` (신규 생성, 311 lines)

#### 주요 기능

- **CLI 인터페이스**: argparse 기반
- **YAML 설정 로드**: sweep_config 파일에서 파라미터 로드
- **W&B Sweep 생성**: `wandb.sweep()` 호출
- **독립 실행**: subprocess로 train.py 실행
- **환경 변수 전달**: WANDB_RUN_ID, WANDB_SWEEP_ID

#### 핵심 함수

1. **load_sweep_config(sweep_config_path)**
   - YAML 파일에서 sweep 설정 로드
   - 에러 처리 포함

2. **load_wandb_project_entity(config_path)**
   - 설정 파일에서 W&B project/entity 추출
   - 환경 변수 우선 사용

3. **main()**
   - CLI 인자 파싱
   - Sweep 설정 구성
   - wandb.agent() 실행

4. **train_sweep()**
   - wandb.init() 초기화
   - wandb.config에서 파라미터 가져오기
   - subprocess로 train.py 실행
   - 독립 디렉토리 생성

### 2. Sweep 설정 파일

4개의 YAML 설정 파일 생성:

#### sweeps/sweep_test.yaml (테스트용)

```yaml
name: reid_test_sweep
method: random
parameters:
  base_lr: [0.0001, 0.00035, 0.0005]
  weight_decay: [0.0001, 0.0005]
  max_epochs: [2]
count: 3
```

**용도**: 빠른 기능 테스트 (3 runs, 2 epochs)  
**예상 시간**: 20-30분

#### sweeps/sweep_lr.yaml (학습률 탐색)

```yaml
name: reid_lr_sweep
method: grid
parameters:
  base_lr: [0.0001, 0.00025, 0.00035, 0.0005, 0.0007]
count: 5
```

**용도**: 최적 학습률 찾기  
**예상 시간**: 2-3시간 (5 epochs 기준)

#### sweeps/sweep_optimizer.yaml (옵티마이저 튜닝)

```yaml
name: reid_optimizer_sweep
method: random
parameters:
  base_lr: [...]
  weight_decay: [...]
  warmup_factor: [...]
  warmup_iters: [...]
count: 15
```

**용도**: 옵티마이저 하이퍼파라미터 최적화  
**예상 시간**: 수 시간

#### sweeps/sweep_full.yaml (전체 탐색)

```yaml
name: reid_full_sweep
method: bayes
parameters:
  base_lr: [...]
  weight_decay: [...]
  warmup_factor: [...]
  warmup_iters: [...]
  margin: [...]
  ims_per_batch: [...]
count: 30
early_terminate:
  type: hyperband
  max_iter: 10
  min_iter: 3
```

**용도**: 전체 하이퍼파라미터 최적화 + Early termination  
**예상 시간**: 수 일

### 3. 테스트 스크립트

**파일**: `test_sweep.sh` (신규 생성, 실행 가능)

- 환경 변수 체크
- 데이터셋 존재 확인
- 빠른 sweep 테스트 실행

### 4. 문서

#### docs/SWEEP_IMPLEMENTATION_PLAN.md

- TAO Pytorch Backend 분석
- reid-strong-baseline 적용 계획
- 구현 세부사항
- 테스트 계획

---

## 지원 하이퍼파라미터

| 파라미터 | YACS 경로 | 설명 |
|---------|-----------|------|
| base_lr | SOLVER.BASE_LR | 기본 학습률 |
| weight_decay | SOLVER.WEIGHT_DECAY | 가중치 감쇠 |
| warmup_factor | SOLVER.WARMUP_FACTOR | 워밍업 시작 배율 |
| warmup_iters | SOLVER.WARMUP_ITERS | 워밍업 반복 횟수 |
| margin | SOLVER.MARGIN | 트리플릿 손실 마진 |
| ims_per_batch | SOLVER.IMS_PER_BATCH | 배치 크기 |
| max_epochs | SOLVER.MAX_EPOCHS | 최대 에포크 |

---

## 디렉토리 구조

```
reid-strong-baseline/
├── tools/
│   ├── train.py          # 기존
│   └── sweep.py          # 신규 (311 lines)
├── sweeps/               # 신규 디렉토리
│   ├── sweep_test.yaml
│   ├── sweep_lr.yaml
│   ├── sweep_optimizer.yaml
│   └── sweep_full.yaml
├── docs/                 # 신규 디렉토리
│   ├── SWEEP_IMPLEMENTATION_PLAN.md
│   └── SWEEP_IMPLEMENTATION_REPORT.md (본 문서)
└── test_sweep.sh         # 신규 (테스트 스크립트)
```

### 실행 시 생성되는 디렉토리

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

## 사용 방법

### 기본 사용

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_test.yaml
```

### 옵션 지정

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml \
    --count 10 \
    --method bayes
```

### CLI로 파라미터 지정

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --lr_values "0.0001,0.00035,0.0005" \
    --wd_values "0.0001,0.0005,0.001" \
    --count 5
```

### 기존 Sweep 재개

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id {SWEEP_ID}
```

---

## 테스트 방법

### 1. 환경 설정

```bash
# W&B 로그인
wandb login
# 또는
export WANDB_API_KEY="your_api_key_here"
```

### 2. 빠른 테스트 실행

```bash
cd /home/jongphago/projects/reid-strong-baseline

# 테스트 스크립트 실행
./test_sweep.sh

# 또는 직접 실행
python tools/sweep.py \
    --config_file configs/test_wandb.yml \
    --sweep-config sweeps/sweep_test.yaml
```

### 3. W&B 대시보드 확인

- 터미널 출력에서 sweep URL 확인
- https://wandb.ai에서 프로젝트 확인
- Sweep parallel coordinates plot 확인
- 최적 하이퍼파라미터 식별

---

## 기술 세부사항

### Subprocess 실행 방식

각 sweep run은 독립된 프로세스로 실행됩니다:

```python
def train_sweep():
    run = wandb.init()
    
    # 하이퍼파라미터 가져오기
    base_lr = wandb.config.get("base_lr")
    
    # 독립 디렉토리 생성
    run_results_dir = os.path.join(base_dir, sweep_id, run_name)
    
    # subprocess로 train.py 실행
    cmd = ["python", "tools/train.py", 
           "--config_file=...",
           f"SOLVER.BASE_LR {base_lr}",
           f"OUTPUT_DIR {run_results_dir}"]
    
    subprocess.run(cmd, env=env)
    wandb.finish()
```

### 환경 변수 전달

```python
env = os.environ.copy()
env["WANDB_RUN_ID"] = run.id
env["WANDB_SWEEP_ID"] = sweep_id
env["WANDB_RESUME"] = "allow"
```

### Early Termination (Hyperband)

```yaml
early_terminate:
  type: hyperband
  max_iter: 10      # 최대 반복 (에포크)
  min_iter: 3       # 최소 반복
  s: 2              # 제거 비율
  eta: 3            # 축소 계수
```

---

## TAO Pytorch Backend와의 차이점

| 항목 | TAO | reid-strong-baseline |
|------|-----|----------------------|
| **설정 관리** | Hydra (dataclass) | YACS (CN) |
| **파라미터 전달** | Hydra override | YACS CLI override |
| **엔트리포인트** | re_identification.py | tools/train.py |
| **파라미터 형식** | `train.optim.base_lr=` | `SOLVER.BASE_LR ` |

### YACS CLI Override 형식

TAO (Hydra):
```bash
train.optim.base_lr=0.001
```

reid-strong-baseline (YACS):
```bash
SOLVER.BASE_LR 0.001
```

---

## 검증 체크리스트

- [x] tools/sweep.py 스크립트 작성
- [x] Sweep 설정 파일 4개 작성
- [x] 테스트 스크립트 작성
- [x] 문서 작성
- [x] YACS 파라미터 매핑
- [x] Subprocess 실행 구현
- [x] 환경 변수 전달
- [x] 독립 디렉토리 생성
- [x] Early termination 지원
- [ ] 사용자 테스트 실행
- [ ] W&B 대시보드 확인

---

## 예상 워크플로우

### Phase 1: 빠른 테스트 (필수)

```bash
./test_sweep.sh
```

**목적**: 기능 확인  
**시간**: 20-30분  
**결과**: W&B에 3개 run 기록

### Phase 2: 학습률 탐색

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```

**목적**: 최적 학습률 찾기  
**시간**: 2-3시간  
**결과**: 최적 학습률 식별

### Phase 3: 옵티마이저 튜닝

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_optimizer.yaml
```

**목적**: 옵티마이저 하이퍼파라미터 최적화  
**시간**: 수 시간  
**결과**: 최적 옵티마이저 설정

### Phase 4: 전체 탐색 (선택)

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_full.yaml
```

**목적**: 전체 하이퍼파라미터 최적화  
**시간**: 수 일  
**결과**: 최적 설정 조합

---

## 문제 해결

### Sweep이 시작되지 않음

```
ERROR: Config file not found: configs/baseline.yml
```

**해결**: 절대 경로 또는 올바른 상대 경로 사용

### W&B 로그인 실패

```
Warning: W&B wasn't logged in.
```

**해결**:
```bash
wandb login
# 또는
export WANDB_API_KEY="your_key"
```

### 메모리 부족

배치 크기를 줄이거나 max_epochs를 조정:

```yaml
parameters:
  ims_per_batch: [16, 32]  # 32 대신 16 사용
  max_epochs: [50]          # 120 대신 50
```

---

## 참고 문서

- [W&B Sweeps 공식 문서](https://docs.wandb.ai/guides/sweeps)
- [Hyperband Early Termination](https://docs.wandb.ai/guides/sweeps/early-termination)
- [Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)
- TAO Pytorch Backend: `nvidia_tao_pytorch/cv/re_identification/scripts/sweep.py`

---

## 요약

### 구현 완료 사항

✅ Sweep 스크립트 (`tools/sweep.py`)  
✅ 4개의 Sweep 설정 파일  
✅ 테스트 스크립트 (`test_sweep.sh`)  
✅ 구현 계획 문서  
✅ 구현 완료 보고서 (본 문서)  

### 다음 단계

1. `./test_sweep.sh` 실행
2. W&B 대시보드 확인
3. 최적 하이퍼파라미터 식별
4. 최적 설정으로 전체 훈련

---

**구현 완료**: 2025-10-15  
**테스트 상태**: 사용자 실행 대기  
**예상 테스트 시간**: 20-30분 (빠른 테스트)

