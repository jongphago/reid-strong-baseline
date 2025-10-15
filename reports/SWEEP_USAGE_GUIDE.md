# W&B Sweep 사용 가이드

## 목차

1. [시작하기](#시작하기)
2. [빠른 테스트](#빠른-테스트)
3. [사용 예시](#사용-예시)
4. [Sweep 설정 커스터마이징](#sweep-설정-커스터마이징)
5. [결과 분석](#결과-분석)
6. [문제 해결](#문제-해결)

---

## 시작하기

### 필수 준비사항

1. **W&B 계정**
   - https://wandb.ai에서 가입
   - API 키 확인

2. **데이터셋**
   - Market1501 데이터셋을 `./data/market1501`에 준비

3. **환경 설정**
   ```bash
   wandb login
   # 또는
   export WANDB_API_KEY="your_api_key_here"
   ```

---

## 빠른 테스트

가장 빠르게 Sweep을 테스트하는 방법:

```bash
cd /home/jongphago/projects/reid-strong-baseline

# 한 줄로 실행
./test_sweep.sh
```

이 스크립트는:
- 환경 변수 확인
- 데이터셋 존재 확인
- 3개의 run으로 빠른 sweep 실행 (2 epochs)
- 약 20-30분 소요

---

## 사용 예시

### 1. 기본 사용법

YAML 설정 파일을 사용한 가장 간단한 실행:

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_test.yaml
```

### 2. 학습률 최적화

최적의 학습률을 찾기 위한 Grid search:

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```

**설정 내용** (sweeps/sweep_lr.yaml):
- Method: Grid search (모든 조합 테스트)
- Parameters: 5개의 학습률 값
- Runs: 5개
- 예상 시간: 2-3시간 (5 epochs 기준)

### 3. 옵티마이저 하이퍼파라미터 튜닝

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_optimizer.yaml
```

**설정 내용** (sweeps/sweep_optimizer.yaml):
- Method: Random search
- Parameters: learning rate, weight decay, warmup 관련
- Runs: 15개
- 예상 시간: 수 시간

### 4. 전체 하이퍼파라미터 최적화

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_full.yaml
```

**설정 내용** (sweeps/sweep_full.yaml):
- Method: Bayesian optimization
- Parameters: 6개 하이퍼파라미터
- Runs: 30개
- Early termination: Hyperband
- 예상 시간: 수 일

### 5. CLI로 실행 횟수 조정

```bash
# 3번만 실행
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_full.yaml \
    --count 3
```

### 6. CLI로 search 방법 변경

```bash
# Random search로 변경
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml \
    --method random \
    --count 3
```

### 7. YAML 없이 CLI만으로 실행

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --lr_values "0.0001,0.00035,0.0005" \
    --wd_values "0.0001,0.0005" \
    --count 5
```

### 8. 기존 Sweep 재개

Sweep이 중단된 경우 재개:

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id abc123xyz  # 실제 sweep ID
```

Sweep ID는 W&B 대시보드 URL에서 확인:
```
https://wandb.ai/username/project/sweeps/abc123xyz
                                              ^^^^^^^ 
                                              여기가 sweep_id
```

---

## Sweep 설정 커스터마이징

### 새로운 Sweep 설정 파일 만들기

`sweeps/my_custom_sweep.yaml`:

```yaml
# Sweep 이름
name: my_custom_sweep

# Search 방법: grid, random, bayes
method: random

# 최적화할 메트릭
metric:
  name: val/mAP        # 메트릭 이름 (W&B에 로깅된 이름)
  goal: maximize       # maximize 또는 minimize

# 튜닝할 하이퍼파라미터
parameters:
  # 학습률
  base_lr: [0.0001, 0.00025, 0.00035, 0.0005]
  
  # 가중치 감쇠
  weight_decay: [0.0001, 0.0005, 0.001]
  
  # 워밍업
  warmup_factor: [0.01, 0.05, 0.1]
  warmup_iters: [0, 10, 20]
  
  # 트리플릿 마진
  margin: [0.2, 0.3, 0.4]
  
  # 배치 크기
  ims_per_batch: [32, 64]
  
  # 에포크 수 (테스트용)
  max_epochs: [5]

# 실행 횟수
count: 10

# 결과 저장 디렉토리
results_dir_base: ./log/my_custom_sweep

# Early termination (선택)
early_terminate:
  type: hyperband
  max_iter: 10    # 최대 10 에포크
  min_iter: 3     # 최소 3 에포크
  s: 2
  eta: 3
```

### Search 방법 선택 가이드

#### Grid Search
```yaml
method: grid
```
- **언제 사용**: 파라미터가 적고 모든 조합을 테스트하고 싶을 때
- **장점**: 완전한 탐색
- **단점**: 파라미터가 많으면 시간이 오래 걸림
- **예시**: 학습률만 튜닝 (5개 값)

#### Random Search
```yaml
method: random
count: 20
```
- **언제 사용**: 파라미터가 많고 빠르게 좋은 설정을 찾고 싶을 때
- **장점**: 빠르고 효율적
- **단점**: 최적값을 놓칠 수 있음
- **예시**: 4-5개 파라미터 튜닝

#### Bayesian Optimization
```yaml
method: bayes
count: 30
```
- **언제 사용**: 파라미터가 많고 최적값을 찾고 싶을 때
- **장점**: 효율적이고 지능적인 탐색
- **단점**: 초기에는 Random과 비슷
- **예시**: 전체 하이퍼파라미터 최적화

### Early Termination 설정

성능이 낮은 run을 조기 종료하여 시간 절약:

```yaml
early_terminate:
  type: hyperband
  max_iter: 10    # 최대 반복 (에포크)
  min_iter: 3     # 최소 반복 (모든 run은 최소 3 에포크 실행)
  s: 2            # 제거 비율
  eta: 3          # 축소 계수
```

**작동 방식**:
- 모든 run은 최소 3 에포크 실행
- 3 에포크 후 성능이 낮은 run은 종료
- 성능이 좋은 run만 계속 실행

---

## 결과 분석

### W&B 대시보드 접속

Sweep 실행 후 터미널에 출력된 URL 클릭:

```
View results at: https://wandb.ai/username/reid-strong-baseline/sweeps/abc123xyz
```

또는 https://wandb.ai → 프로젝트 → Sweeps 탭

### 주요 뷰

#### 1. Parallel Coordinates Plot

최적 파라미터 조합을 시각적으로 확인:

- 각 선이 하나의 run을 나타냄
- 색상이 성능을 나타냄 (빨강=좋음, 파랑=나쁨)
- 좋은 성능의 선들이 지나가는 값이 최적값

#### 2. Importance Plot

어떤 파라미터가 성능에 가장 영향을 미치는지 확인:

- 막대가 길수록 중요한 파라미터
- 가장 중요한 파라미터부터 튜닝

#### 3. Parameter Importance

각 파라미터 값에 따른 성능 분포:

- 어떤 값이 좋은 성능을 내는지 확인
- 최적 범위 식별

#### 4. Runs Table

모든 run의 결과를 테이블로 확인:

- 정렬: mAP 기준 정렬
- 필터: 특정 조건의 run만 보기
- 비교: 여러 run 선택하여 비교

### 최적 하이퍼파라미터 식별

1. **Runs Table에서 정렬**
   - `val/mAP` 컬럼 클릭하여 내림차순 정렬
   - 가장 높은 mAP를 가진 run 확인

2. **파라미터 확인**
   - 최고 성능 run의 파라미터 값 확인
   - 해당 값들을 메모

3. **최종 훈련**
   ```bash
   python tools/train.py \
       --config_file configs/baseline.yml \
       SOLVER.BASE_LR 0.00035 \
       SOLVER.WEIGHT_DECAY 0.0005 \
       SOLVER.WARMUP_FACTOR 0.05 \
       OUTPUT_DIR ./log/best_config
   ```

---

## 고급 사용

### 특정 GPU 지정

```bash
CUDA_VISIBLE_DEVICES=0 python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_test.yaml
```

### 병렬 실행 (여러 GPU)

**Terminal 1** (GPU 0):
```bash
CUDA_VISIBLE_DEVICES=0 python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id abc123xyz \
    --count 3
```

**Terminal 2** (GPU 1):
```bash
CUDA_VISIBLE_DEVICES=1 python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id abc123xyz \
    --count 3
```

동일한 `sweep_id`를 사용하면 여러 agent가 동시에 작업합니다.

### 조건부 파라미터

특정 조합만 테스트하고 싶을 때:

```yaml
parameters:
  base_lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.001
  
  weight_decay:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.001
```

---

## 문제 해결

### 문제 1: Sweep이 시작되지 않음

**증상**:
```
ERROR: Config file not found: configs/baseline.yml
```

**해결**:
```bash
# 현재 디렉토리 확인
pwd

# reid-strong-baseline 디렉토리에서 실행해야 함
cd /home/jongphago/projects/reid-strong-baseline

# 다시 실행
python tools/sweep.py --config_file configs/baseline.yml ...
```

### 문제 2: W&B 로그인 실패

**증상**:
```
Warning: W&B wasn't logged in.
```

**해결**:
```bash
# 방법 1: 로그인
wandb login

# 방법 2: API 키 설정
export WANDB_API_KEY="your_api_key_here"

# API 키 확인
echo $WANDB_API_KEY
```

### 문제 3: 메모리 부족 (OOM)

**증상**:
```
CUDA out of memory
```

**해결**:

sweeps/my_sweep.yaml 수정:
```yaml
parameters:
  ims_per_batch: [16, 32]  # 64 → 32 또는 16
```

또는 기존 설정 파일 수정:
```bash
python tools/sweep.py \
    --config_file configs/test_wandb.yml \  # 이미 작은 배치 사용
    --sweep-config sweeps/sweep_test.yaml
```

### 문제 4: Validation 결과가 기록되지 않음

**증상**: W&B에 `val/mAP`가 보이지 않음

**원인**: EVAL_PERIOD가 MAX_EPOCHS보다 큼

**해결**:
```yaml
# sweeps/sweep_test.yaml
parameters:
  max_epochs: [5]
```

그리고 configs/test_wandb.yml 확인:
```yaml
SOLVER:
  MAX_EPOCHS: 5
  EVAL_PERIOD: 5  # ≤ MAX_EPOCHS 여야 함
```

### 문제 5: Sweep이 중단됨

**해결**: 동일한 sweep_id로 재개
```bash
# W&B 대시보드에서 sweep_id 확인
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id abc123xyz
```

---

## 실전 워크플로우

### Step 1: 빠른 기능 테스트 (필수)

```bash
./test_sweep.sh
```

**목적**: Sweep 기능 확인  
**시간**: 20-30분  
**체크**: W&B 대시보드에 3개 run 확인

### Step 2: 학습률 찾기

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```

**목적**: 최적 학습률 식별  
**시간**: 2-3시간  
**결과**: 예) base_lr = 0.00035

### Step 3: 세부 튜닝

sweeps/my_finetune.yaml 생성:
```yaml
name: finetune_sweep
method: grid
parameters:
  base_lr: [0.00025, 0.00035, 0.0005]  # Step 2 결과 주변
  weight_decay: [0.0001, 0.0005, 0.001]
  warmup_factor: [0.01, 0.05]
count: 18
```

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/my_finetune.yaml
```

### Step 4: 최종 훈련

```bash
# 최적 하이퍼파라미터로 전체 에포크 훈련
python tools/train.py \
    --config_file configs/baseline.yml \
    SOLVER.BASE_LR 0.00035 \
    SOLVER.WEIGHT_DECAY 0.0005 \
    SOLVER.WARMUP_FACTOR 0.05 \
    SOLVER.MAX_EPOCHS 120 \
    OUTPUT_DIR ./log/final_best
```

---

## 팁과 요령

### 1. 시작은 작게

처음에는 적은 파라미터, 적은 에포크로 시작:

```yaml
parameters:
  base_lr: [0.0001, 0.00035, 0.001]  # 3개만
  max_epochs: [5]                     # 5 에포크만
count: 3
```

### 2. 점진적 확장

성능에 영향이 큰 파라미터부터 튜닝:

1. 학습률
2. Weight decay
3. Warmup
4. Margin
5. Batch size

### 3. Early Termination 활용

시간이 오래 걸리는 경우:

```yaml
early_terminate:
  type: hyperband
  max_iter: 10
  min_iter: 3
```

### 4. 로그 확인

각 run의 로그는 독립 디렉토리에 저장:

```bash
# 특정 run의 로그 확인
tail -f log/sweep_test/sweep/{sweep_id}/{run_name}/log.txt
```

### 5. 중간 결과 활용

Sweep 진행 중에도 W&B 대시보드에서 중간 결과 확인 가능:
- 현재까지 최고 성능
- 파라미터 중요도
- 조기 중단 가능

---

## 참고 자료

- [W&B Sweeps 공식 문서](https://docs.wandb.ai/guides/sweeps)
- [Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)
- [Hyperband](https://docs.wandb.ai/guides/sweeps/early-termination)
- [구현 보고서](./SWEEP_IMPLEMENTATION_REPORT.md)
- [구현 계획](./SWEEP_IMPLEMENTATION_PLAN.md)

---

**마지막 업데이트**: 2025-10-15  
**질문/피드백**: GitHub Issues 또는 W&B Community

