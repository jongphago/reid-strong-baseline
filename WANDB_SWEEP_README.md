# W&B Sweep 구현 완료

reid-strong-baseline 프로젝트에 W&B Sweep 하이퍼파라미터 자동 튜닝 기능이 추가되었습니다.

## 📋 구현 완료 사항

### ✅ 핵심 파일

1. **Sweep 스크립트**
   - `tools/sweep.py` (311 lines) - 메인 sweep 실행 스크립트

2. **Sweep 설정 파일** (4개)
   - `sweeps/sweep_test.yaml` - 빠른 테스트 (3 runs, 2 epochs)
   - `sweeps/sweep_lr.yaml` - 학습률 탐색 (Grid)
   - `sweeps/sweep_optimizer.yaml` - 옵티마이저 튜닝 (Random)
   - `sweeps/sweep_full.yaml` - 전체 최적화 (Bayesian + Hyperband)

3. **테스트 스크립트**
   - `test_sweep.sh` - 빠른 기능 테스트

4. **문서** (3개)
   - `docs/SWEEP_IMPLEMENTATION_PLAN.md` - 구현 계획
   - `docs/SWEEP_IMPLEMENTATION_REPORT.md` - 구현 보고서
   - `docs/SWEEP_USAGE_GUIDE.md` - 사용 가이드

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# W&B 로그인
wandb login
```

### 2. 빠른 테스트 (20-30분)

```bash
cd /home/jongphago/projects/reid-strong-baseline
./test_sweep.sh
```

### 3. 학습률 최적화 (2-3시간)

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```

## 📊 주요 기능

### Sweep Methods
- **Grid Search**: 모든 조합 테스트
- **Random Search**: 랜덤 샘플링
- **Bayesian Optimization**: 지능적 탐색

### 지원 하이퍼파라미터
- `base_lr` - 학습률
- `weight_decay` - 가중치 감쇠
- `warmup_factor` - 워밍업 배율
- `warmup_iters` - 워밍업 반복
- `margin` - 트리플릿 마진
- `ims_per_batch` - 배치 크기
- `max_epochs` - 에포크 수

### Early Termination
- Hyperband 알고리즘 지원
- 성능 낮은 run 조기 종료
- 시간 절약

## 📁 파일 구조

```
reid-strong-baseline/
├── tools/
│   ├── train.py
│   └── sweep.py          ← 신규
├── sweeps/               ← 신규
│   ├── sweep_test.yaml
│   ├── sweep_lr.yaml
│   ├── sweep_optimizer.yaml
│   └── sweep_full.yaml
├── docs/                 ← 신규
│   ├── SWEEP_IMPLEMENTATION_PLAN.md
│   ├── SWEEP_IMPLEMENTATION_REPORT.md
│   └── SWEEP_USAGE_GUIDE.md
├── test_sweep.sh         ← 신규
└── WANDB_SWEEP_README.md ← 본 문서
```

## 📖 사용 예시

### 기본 사용

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_test.yaml
```

### 실행 횟수 지정

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml \
    --count 3
```

### CLI로 파라미터 지정

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --lr_values "0.0001,0.00035,0.0005" \
    --wd_values "0.0001,0.0005" \
    --count 5
```

### 기존 Sweep 재개

```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep_id abc123xyz
```

## 🔬 Sweep 설정 예시

### sweep_test.yaml (테스트)
```yaml
name: reid_test_sweep
method: random
metric:
  name: val/mAP
  goal: maximize
parameters:
  base_lr: [0.0001, 0.00035, 0.0005]
  weight_decay: [0.0001, 0.0005]
  max_epochs: [2]
count: 3
```

### sweep_full.yaml (전체 최적화)
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
count: 30
early_terminate:
  type: hyperband
  max_iter: 10
  min_iter: 3
```

## 📈 결과 확인

### W&B 대시보드

Sweep 실행 후 터미널에 출력된 URL로 접속:
```
https://wandb.ai/username/reid-strong-baseline/sweeps/{sweep_id}
```

### 주요 뷰
1. **Parallel Coordinates**: 최적 파라미터 조합 시각화
2. **Importance**: 파라미터 중요도
3. **Runs Table**: 모든 실행 결과 비교

### 최적 설정 찾기

1. Runs Table에서 `val/mAP` 기준 정렬
2. 최고 성능 run의 파라미터 확인
3. 해당 설정으로 최종 훈련:

```bash
python tools/train.py \
    --config_file configs/baseline.yml \
    SOLVER.BASE_LR 0.00035 \
    SOLVER.WEIGHT_DECAY 0.0005 \
    OUTPUT_DIR ./log/best_config
```

## 🛠 문제 해결

### W&B 로그인 실패
```bash
wandb login
# 또는
export WANDB_API_KEY="your_key"
```

### 메모리 부족
```yaml
# 배치 크기 줄이기
parameters:
  ims_per_batch: [16, 32]  # 대신 64
```

### Validation 결과 없음
```yaml
# EVAL_PERIOD ≤ MAX_EPOCHS 확인
SOLVER:
  MAX_EPOCHS: 5
  EVAL_PERIOD: 5
```

## 📚 상세 문서

- **사용 가이드**: [docs/SWEEP_USAGE_GUIDE.md](docs/SWEEP_USAGE_GUIDE.md)
  - 상세한 사용법
  - 고급 기능
  - 실전 워크플로우

- **구현 보고서**: [docs/SWEEP_IMPLEMENTATION_REPORT.md](docs/SWEEP_IMPLEMENTATION_REPORT.md)
  - 구현 세부사항
  - 기술 세부사항
  - TAO와의 차이점

- **구현 계획**: [docs/SWEEP_IMPLEMENTATION_PLAN.md](docs/SWEEP_IMPLEMENTATION_PLAN.md)
  - TAO 분석
  - 설계 결정
  - 테스트 계획

## 🎯 권장 워크플로우

### Phase 1: 빠른 테스트 (필수)
```bash
./test_sweep.sh
```
**시간**: 20-30분  
**목적**: 기능 확인

### Phase 2: 학습률 탐색
```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_lr.yaml
```
**시간**: 2-3시간  
**목적**: 최적 학습률 찾기

### Phase 3: 세부 튜닝
```bash
python tools/sweep.py \
    --config_file configs/baseline.yml \
    --sweep-config sweeps/sweep_optimizer.yaml
```
**시간**: 수 시간  
**목적**: 옵티마이저 하이퍼파라미터 최적화

### Phase 4: 최종 훈련
```bash
python tools/train.py \
    --config_file configs/baseline.yml \
    SOLVER.BASE_LR {best_lr} \
    SOLVER.WEIGHT_DECAY {best_wd} \
    SOLVER.MAX_EPOCHS 120
```

## ✨ 주요 특징

- ✅ **TAO Pytorch Backend 기반**: 검증된 구현 패턴
- ✅ **YACS 통합**: 기존 설정 시스템과 완벽 호환
- ✅ **독립 실행**: Subprocess로 안정적 실행
- ✅ **Early Termination**: 시간 절약
- ✅ **완전한 문서화**: 계획서, 보고서, 사용 가이드

## 📝 참고

### TAO Pytorch Backend 참조
- `nvidia_tao_pytorch/cv/re_identification/scripts/sweep.py`
- `nvidia_tao_pytorch/cv/re_identification/sweeps/*.yaml`

### W&B 문서
- [W&B Sweeps](https://docs.wandb.ai/guides/sweeps)
- [Sweep Configuration](https://docs.wandb.ai/guides/sweeps/configuration)
- [Early Termination](https://docs.wandb.ai/guides/sweeps/early-termination)

---

**구현 완료**: 2025-10-15  
**테스트 상태**: 사용자 실행 대기  
**예상 테스트 시간**: 20-30분 (빠른 테스트)

궁금한 점이나 문제가 있으면 [SWEEP_USAGE_GUIDE.md](docs/SWEEP_USAGE_GUIDE.md)의 문제 해결 섹션을 참조하세요.

