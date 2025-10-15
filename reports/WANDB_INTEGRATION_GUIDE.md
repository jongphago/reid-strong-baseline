# W&B Integration Guide

이 가이드는 reid-strong-baseline 프로젝트에 추가된 Weights & Biases (W&B) 통합에 대한 사용 방법을 설명합니다.

## 설치

W&B 패키지가 이미 `pyproject.toml`에 추가되어 있습니다. 의존성을 설치하세요:

```bash
cd /home/jongphago/projects/reid-strong-baseline
uv sync
# 또는
pip install wandb
```

## 환경 설정

### 방법 1: W&B 로그인 (권장)

```bash
wandb login
```

브라우저가 열리면 W&B 계정으로 로그인하세요.

### 방법 2: API 키 환경 변수

```bash
export WANDB_API_KEY="your_api_key_here"
```

API 키는 https://wandb.ai/authorize 에서 확인할 수 있습니다.

## 빠른 테스트

W&B 통합이 제대로 작동하는지 확인하기 위한 빠른 테스트:

```bash
cd /home/jongphago/projects/reid-strong-baseline

# WANDB_API_KEY 설정 (필요한 경우)
export WANDB_API_KEY="your_api_key_here"

# 테스트 실행 (5 에포크)
./test_wandb.sh
```

테스트 완료 후 W&B 대시보드(https://wandb.ai)에서 결과를 확인하세요.

## 사용 방법

### 기본 훈련

W&B 로깅이 활성화된 상태로 훈련:

```bash
python tools/train.py \
    --config_file='configs/baseline.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')" \
    WANDB.ENABLE True \
    WANDB.PROJECT "my-reid-project" \
    WANDB.NAME "experiment_1" \
    OUTPUT_DIR "./log/experiment_1"
```

### W&B 비활성화

W&B 없이 훈련하려면:

```bash
python tools/train.py \
    --config_file='configs/baseline.yml' \
    WANDB.ENABLE False
```

또는 설정 파일에서 직접 수정:

```yaml
WANDB:
  ENABLE: False
```

## 설정 옵션

YAML 설정 파일 또는 CLI 인자로 다음 옵션을 설정할 수 있습니다:

```yaml
WANDB:
  ENABLE: True                           # W&B 로깅 활성화/비활성화
  PROJECT: "reid-strong-baseline"        # W&B 프로젝트 이름
  ENTITY: ""                             # W&B 팀/사용자 이름 (선택)
  NAME: ""                               # 실험 이름 (비어있으면 자동 생성)
  TAGS: ["tag1", "tag2"]                 # 실험 태그
  SYNC_TENSORBOARD: False                # TensorBoard 동기화
  SAVE_CODE: False                       # 코드 저장
```

## 로깅되는 메트릭

### 훈련 메트릭 (매 LOG_PERIOD 스텝마다)

- `train/loss`: 훈련 손실
- `train/acc`: 훈련 정확도
- `train/lr`: 현재 학습률
- `epoch`: 현재 에포크

### 검증 메트릭 (매 EVAL_PERIOD 에포크마다)

- `val/mAP`: Mean Average Precision
- `val/cmc_rank_1`: CMC Rank-1 정확도
- `val/cmc_rank_5`: CMC Rank-5 정확도
- `val/cmc_rank_10`: CMC Rank-10 정확도
- `epoch`: 현재 에포크

## 실험 예시

### Market1501 데이터셋

```bash
python tools/train.py \
    --config_file='configs/softmax_triplet.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('market1501')" \
    WANDB.ENABLE True \
    WANDB.PROJECT "reid-market1501" \
    WANDB.NAME "resnet50_baseline" \
    WANDB.TAGS "['baseline', 'market1501', 'resnet50']" \
    OUTPUT_DIR "./log/market1501/baseline"
```

### DukeMTMC-reID 데이터셋 (Center Loss 포함)

```bash
python tools/train.py \
    --config_file='configs/softmax_triplet_with_center.yml' \
    MODEL.DEVICE_ID "('0')" \
    DATASETS.NAMES "('dukemtmc')" \
    WANDB.ENABLE True \
    WANDB.PROJECT "reid-duke" \
    WANDB.NAME "resnet50_with_center" \
    WANDB.TAGS "['center_loss', 'dukemtmc', 'resnet50']" \
    OUTPUT_DIR "./log/dukemtmc/with_center"
```

## W&B 대시보드 확인

훈련 시작 후 터미널에 출력되는 W&B 링크를 클릭하거나, https://wandb.ai 에서 프로젝트를 찾아 다음 정보를 확인할 수 있습니다:

- 실시간 훈련/검증 메트릭 그래프
- 하이퍼파라미터 비교
- 시스템 리소스 사용량
- 실험 간 비교

## 문제 해결

### W&B 로그인 실패

```
Warning: W&B wasn't logged in.
Warning: W&B login failed. Logging disabled.
```

해결 방법:
1. `wandb login` 실행
2. 또는 `export WANDB_API_KEY="your_key"` 설정

### 네트워크 오류

인터넷 연결이 불안정하면 W&B 로깅이 실패할 수 있습니다. 이 경우 훈련은 계속 진행되며 로컬 로그만 저장됩니다.

### W&B 없이 훈련

W&B를 사용하지 않으려면 설정에서 `WANDB.ENABLE: False`로 설정하세요.

## 구현 세부사항

W&B 통합은 다음 파일에서 구현되었습니다:

- `config/defaults.py`: W&B 설정 추가
- `utils/wandb_utils.py`: W&B 유틸리티 함수
- `tools/train.py`: W&B 초기화
- `engine/trainer.py`: 메트릭 로깅

## 추가 기능 (향후 계획)

- W&B Sweep을 통한 하이퍼파라미터 튜닝
- 샘플 이미지 로깅
- 모델 아티팩트 저장
- 커스텀 차트 및 테이블

## 참고 문서

- [W&B 공식 문서](https://docs.wandb.ai/)
- [PyTorch Ignite와 W&B](https://docs.wandb.ai/guides/integrations/ignite)
- [프로젝트 구조 분석 보고서](./프로젝트_구조_분석_및_WandB_적용_계획.md)

