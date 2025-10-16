# 목차

- [목차](#목차)
- [개요](#개요)
- [문제 상황](#문제-상황)
- [문제 원인 분석](#문제-원인-분석)
- [해결 과정](#해결-과정)
- [해결 방법](#해결-방법)
- [검증 결과](#검증-결과)
- [결론](#결론)
- [참조](#참조)

# 개요

본 보고서는 `reid-strong-baseline` 프로젝트에 W&B Sweep 기능을 구현하는 과정에서 발생한 메트릭 기록 문제의 원인 분석 및 해결 과정을 기술합니다.

## 핵심 요약

- **문제**: W&B Sweep 실행 시 학습 및 평가 메트릭이 W&B 웹앱에 기록되지 않음
- **원인**: 부모-자식 프로세스 간 W&B run 초기화 충돌
- **해결**: 자식 프로세스에서 W&B 비활성화, 부모 프로세스가 stdout 파싱하여 메트릭 기록
- **부수적 문제**: YACS 설정 파라미터 전달 형식 오류

# 문제 상황

## 초기 문제: YACS 파라미터 전달 오류

Sweep 실행 시 다음과 같은 오류 발생:

```
AssertionError: Non-existent key: OUTPUT_DIR ./log/sweep_test/sweep/qh61ykhs/dutiful-sweep-2
```

YACS의 `merge_from_list` 메서드가 키와 값을 별도의 리스트 항목으로 받아야 하는데, 공백으로 구분된 하나의 문자열로 전달되어 발생한 오류입니다.

## 주요 문제: 메트릭 미기록

YACS 오류 수정 후에도 다음과 같은 문제 발생:

- Sweep이 정상적으로 시작되고 훈련이 완료됨
- 로컬 로그 파일에는 loss, accuracy, mAP, CMC ranks 등 모든 메트릭이 기록됨
- W&B 웹앱에는 메트릭이 전혀 기록되지 않음
- `wandb-summary.json` 파일에 `_runtime`만 존재하고 메트릭 데이터 없음

# 문제 원인 분석

## YACS 파라미터 전달 오류

잘못된 방식:
```python
cmd_parts.append(f"OUTPUT_DIR {run_results_dir}")  # 하나의 문자열
cmd_parts.append(f"SOLVER.BASE_LR {base_lr}")      # 하나의 문자열
```

YACS는 `merge_from_list`에서 리스트를 `[key1, value1, key2, value2, ...]` 형식으로 받아야 합니다.

## W&B Sweep 메트릭 미기록 원인

### 문제의 근본 원인

1. **부모 프로세스** (`sweep.py`)에서 `wandb.init()` 호출 → run 생성
2. **자식 프로세스** (`train.py`)에서 다시 `wandb.init()` 호출 시도
3. W&B가 sweep 모드에서 run_id 재사용을 무시: `WARNING Ignoring run_id 'xxx' when running a sweep`
4. 자식 프로세스의 W&B 로깅이 부모 run과 연결되지 않음
5. 결과적으로 메트릭이 W&B에 기록되지 않음

### 원인 분석 로그

```
wandb: WARNING Ignoring project 'reid-strong-baseline-test' when running a sweep.
wandb: WARNING Ignoring run_id 'nuselqfe' when running a sweep.
```

W&B agent가 sweep 모드에서 run을 자동으로 관리하려 하지만, 자식 프로세스에서 별도로 `wandb.init()`을 호출하면서 충돌이 발생했습니다.

# 해결 과정

## 단계 1: YACS 파라미터 전달 수정

키와 값을 별도의 리스트 항목으로 분리:

```python
# 수정 전
cmd_parts.append(f"OUTPUT_DIR {run_results_dir}")

# 수정 후
cmd_parts.append("OUTPUT_DIR")
cmd_parts.append(run_results_dir)
```

모든 하이퍼파라미터 오버라이드에 동일한 방식 적용.

## 단계 2: W&B 초기화 로직 검토

여러 접근 방법을 검토:

### 시도 1: 자식 프로세스에서 기존 run 재사용
```python
# utils/wandb_utils.py
if os.getenv("WANDB_RUN_ID"):
    run = wandb.init(id=os.getenv("WANDB_RUN_ID"), resume="allow")
```

**결과**: W&B가 sweep 모드에서 run_id를 무시하여 실패

### 시도 2: 부모 프로세스만 W&B 사용 (최종 채택)

자식 프로세스에서 W&B를 완전히 비활성화하고, 부모 프로세스가 자식의 출력을 파싱하여 메트릭을 기록하는 방식으로 변경.

# 해결 방법

## 파일 수정 내역

### 1. sweep.py - 환경 변수 설정

자식 프로세스에서 W&B 비활성화:

```python
# Set environment variables
env = os.environ.copy()
# Disable W&B in child process to avoid conflicts
env["WANDB_MODE"] = "disabled"
```

### 2. sweep.py - stdout 파싱 및 메트릭 로깅

자식 프로세스의 출력을 실시간으로 파싱하여 메트릭 추출:

```python
# Run training and capture output for logging
import re
process = subprocess.Popen(
    cmd_parts,
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

# Parse output and log metrics to W&B
step = 0
for line in iter(process.stdout.readline, ''):
    print(line, end='')  # Echo output
    
    # Parse training metrics
    train_match = re.search(
        r'Epoch\[(\d+)\].*Loss:\s*([\d.]+).*Acc:\s*([\d.]+).*Base Lr:\s*([\d.e-]+)', 
        line
    )
    if train_match:
        epoch = int(train_match.group(1))
        loss = float(train_match.group(2))
        acc = float(train_match.group(3))
        lr = float(train_match.group(4))
        wandb.log({
            'train/loss': loss, 
            'train/acc': acc, 
            'train/lr': lr, 
            'epoch': epoch
        }, step=step)
        step += 1
    
    # Parse validation metrics
    map_match = re.search(r'mAP:\s*([\d.]+)%', line)
    if map_match:
        mAP = float(map_match.group(1))
        wandb.log({'val/mAP': mAP}, step=step)
    
    # Parse CMC metrics
    cmc_match = re.search(r'CMC curve, Rank-(\d+)\s*:([\d.]+)%', line)
    if cmc_match:
        rank = int(cmc_match.group(1))
        value = float(cmc_match.group(2))
        wandb.log({f'val/cmc_rank_{rank}': value}, step=step)
```

### 3. wandb_utils.py - 환경 변수 체크

Sweep 모드에서 자식 프로세스의 W&B 초기화 방지:

```python
def initialize_wandb(cfg):
    if not cfg.WANDB.ENABLE:
        return None
    
    # Check if W&B is disabled via environment variable (e.g., during sweep)
    if os.getenv("WANDB_MODE") == "disabled":
        print("W&B is disabled via WANDB_MODE environment variable (sweep mode)")
        return None
    
    # ... 기존 초기화 로직
```

### 4. test_sweep.sh - Python 경로 수정

가상 환경의 Python을 명시적으로 사용:

```bash
.venv/bin/python tools/sweep.py \
    --config_file configs/test_wandb.yml \
    --sweep-config sweeps/sweep_test.yaml
```

## 아키텍처

```
┌─────────────────────────────────────────────┐
│ sweep.py (부모 프로세스)                     │
│ - wandb.init() 호출                          │
│ - sweep run 생성                             │
│ - WANDB_MODE=disabled 설정                   │
│ - 자식 프로세스 실행                          │
│ - stdout 파싱 → 메트릭 추출 → wandb.log()   │
│ - wandb.finish()                             │
└─────────────┬───────────────────────────────┘
              │
              │ subprocess.Popen()
              │ env={"WANDB_MODE": "disabled"}
              │
              ▼
┌─────────────────────────────────────────────┐
│ train.py (자식 프로세스)                     │
│ - WANDB_MODE=disabled 감지                   │
│ - W&B 초기화 스킵                            │
│ - 정상적으로 학습 진행                        │
│ - stdout으로 로그 출력                       │
└─────────────────────────────────────────────┘
```

# 검증 결과

## 테스트 실행

```bash
cd /home/jongphago/projects/reid-strong-baseline
.venv/bin/python tools/sweep.py \
    --config_file configs/test_wandb.yml \
    --sweep-config sweeps/sweep_test.yaml \
    --count 1
```

## 로그 확인

자식 프로세스에서 W&B 비활성화 확인:
```
W&B is disabled via WANDB_MODE environment variable (sweep mode)
```

훈련 진행 정상:
```
2025-10-16 02:56:55,388 reid_baseline.train INFO: Epoch[1] Iteration[20/183] Loss: 10.012, Acc: 0.000, Base Lr: 1.09e-05
...
2025-10-16 03:01:42,406 reid_baseline.train INFO: mAP: 4.4%
2025-10-16 03:01:42,406 reid_baseline.train INFO: CMC curve, Rank-1  :11.9%
```

## W&B 메트릭 기록 확인

`wandb-summary.json` 파일 내용:

```json
{
    "train/loss": 7.486,
    "train/acc": 0.204,
    "train/lr": 2.08e-05,
    "epoch": 2,
    "val/mAP": 10.6,
    "val/cmc_rank_1": 23.9,
    "val/cmc_rank_5": 43.5,
    "val/cmc_rank_10": 53.5,
    "_step": 18,
    "_runtime": 624.451325206
}
```

## 기록된 메트릭

- **훈련 메트릭**: loss, accuracy, learning rate, epoch
- **검증 메트릭**: mAP, CMC rank-1, rank-5, rank-10
- **메타 정보**: step, runtime

모든 메트릭이 정상적으로 W&B에 기록되었으며, 웹앱에서 시각화 확인 가능합니다.

# 결론

## 해결된 문제

1. YACS 파라미터 전달 형식 오류 수정
2. W&B Sweep 모드에서 메트릭 미기록 문제 해결
3. 부모-자식 프로세스 간 W&B run 충돌 해소

## 핵심 해결 방안

- 자식 프로세스에서 W&B를 완전히 비활성화 (`WANDB_MODE=disabled`)
- 부모 프로세스가 자식의 stdout을 실시간 파싱
- 정규식을 사용하여 메트릭 추출
- 부모 프로세스에서 `wandb.log()`로 메트릭 기록

## 장점

1. **안정성**: 프로세스 간 W&B 충돌 완전 제거
2. **유연성**: 자식 프로세스의 코드 수정 최소화
3. **확장성**: 새로운 메트릭 추가 시 정규식만 추가하면 됨
4. **호환성**: 기존 단일 훈련 모드와 sweep 모드 모두 지원

## 향후 개선 사항

1. 정규식 패턴을 설정 파일로 분리하여 유지보수성 향상
2. 파싱 실패 시 fallback 메커니즘 추가
3. 멀티 프로세스 환경에서의 동시 sweep 지원

# 참조

## 수정된 파일

### tools/sweep.py
- 행 228-232: OUTPUT_DIR 파라미터 전달 방식 수정
- 행 235-255: 하이퍼파라미터 오버라이드 방식 수정 (키/값 분리)
- 행 258-261: 환경 변수 설정 (WANDB_MODE=disabled)
- 행 268-306: stdout 파싱 및 메트릭 로깅 로직 추가

### utils/wandb_utils.py
- 행 38-41: WANDB_MODE 환경 변수 체크 추가

### test_sweep.sh
- 행 27: Python 경로 수정 (python → .venv/bin/python)

## W&B 링크

- Sweep 페이지: `https://wandb.ai/jongphago/reid-strong-baseline-test/sweeps/43qp59vg`
- Run 페이지: `https://wandb.ai/jongphago/reid-strong-baseline-test/runs/6bpwakjh`

## 참고 문서

- YACS Documentation: `merge_from_list` API
- W&B Sweeps Documentation: Agent 사용법
- Python subprocess 모듈: `Popen` 및 stdout 파이프 처리
- Python re 모듈: 정규식 패턴 매칭

