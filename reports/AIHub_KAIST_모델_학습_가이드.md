# 목차

- [목차](#목차)
- [개요](#개요)
- [사전 준비](#사전-준비)
- [설정 파일 구성](#설정-파일-구성)
- [학습 실행](#학습-실행)
- [학습 모니터링](#학습-모니터링)
- [고급 설정](#고급-설정)
- [문제 해결](#문제-해결)
- [참조](#참조)

# 개요

## 목적

본 문서는 AIHub KAIST Person Re-identification 데이터셋을 사용하여 Reid Strong Baseline 프로젝트에서 모델을 학습하는 전체 과정을 설명합니다.

## 학습 파이프라인 요약

Reid Strong Baseline의 학습 파이프라인은 다음과 같은 단계로 구성됩니다:

1. 설정 파일 로드 및 병합
2. 데이터셋 로딩 및 전처리
3. 모델 구성 및 초기화
4. 손실 함수 및 옵티마이저 설정
5. 학습 루프 실행
6. 주기적 평가 및 체크포인트 저장
7. W&B를 통한 실시간 모니터링

## 지원하는 학습 모드

프로젝트는 두 가지 학습 모드를 지원합니다:

1. 표준 모드: Softmax + Triplet Loss
2. Center Loss 모드: Softmax + Triplet + Center Loss

AIHub KAIST 데이터셋은 두 모드 모두에서 사용 가능합니다.

# 사전 준비

## 데이터셋 준비

### 디렉토리 구조 확인

다음 디렉토리가 올바르게 구성되어 있는지 확인합니다:

```
data/aihub_kaist/
├── bounding_box_train_1/    # 20,119 이미지 (188 PIDs)
├── bounding_box_train_2/    # 18,476 이미지 (175 PIDs)
├── bounding_box_train_3/    # 18,495 이미지 (175 PIDs)
├── query/                   # 1,065 이미지 (145 PIDs)
└── bounding_box_test/       # 15,385 이미지 (145 PIDs)
```

### Query 폴더 준비

Query 폴더가 압축되어 있다면 다음 명령으로 압축을 해제합니다:

```bash
cd data/aihub_kaist
unzip query.zip
```

## 환경 설정

### Python 패키지 확인

필수 패키지가 설치되어 있는지 확인합니다:

```bash
python3 -c "import torch; import torchvision; import yacs; print('OK')"
```

### GPU 확인

CUDA가 사용 가능한지 확인합니다:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### W&B 인증

Weights & Biases를 사용할 경우 로그인이 필요합니다:

```bash
wandb login
```

## 사전 학습 모델 준비

ImageNet 사전 학습 ResNet-50 모델이 필요합니다. 기본 경로는 다음과 같습니다:

```
/home/jongphago/.torch/models/resnet50.pth
```

다른 경로를 사용할 경우 설정 파일에서 `MODEL.PRETRAIN_PATH`를 수정합니다.

# 설정 파일 구성

## 기본 설정 파일

`configs/aihub_kaist.yml` 파일은 AIHub KAIST 데이터셋에 최적화된 기본 설정을 제공합니다.

### 모델 설정

```yaml
MODEL:
  PRETRAIN_CHOICE: 'imagenet'
  PRETRAIN_PATH: '/home/jongphago/.torch/models/resnet50.pth'
  METRIC_LOSS_TYPE: 'triplet'
  IF_LABELSMOOTH: 'on'
  IF_WITH_CENTER: 'no'
```

주요 옵션:
- `PRETRAIN_CHOICE`: 사전 학습 가중치 선택 (imagenet/self)
- `METRIC_LOSS_TYPE`: 메트릭 학습 손실 함수 (triplet/triplet_center)
- `IF_LABELSMOOTH`: 레이블 스무딩 활성화 여부
- `IF_WITH_CENTER`: Center Loss 사용 여부

### 입력 설정

```yaml
INPUT:
  SIZE_TRAIN: [256, 128]
  SIZE_TEST: [256, 128]
  PROB: 0.5
  RE_PROB: 0.5
  PADDING: 10
```

주요 옵션:
- `SIZE_TRAIN`: 훈련 이미지 크기 [높이, 너비]
- `SIZE_TEST`: 테스트 이미지 크기
- `PROB`: 수평 플립 확률
- `RE_PROB`: Random Erasing 확률
- `PADDING`: 패딩 크기

### 데이터셋 설정

```yaml
DATASETS:
  NAMES: 'aihub_kaist'
  ROOT_DIR: './data'
  TRAIN_FOLDER: 'bounding_box_train_1'
```

주요 옵션:
- `NAMES`: 데이터셋 이름
- `ROOT_DIR`: 데이터 루트 디렉토리
- `TRAIN_FOLDER`: 사용할 훈련 폴더 (train_1/2/3 선택)

### 데이터 로더 설정

```yaml
DATALOADER:
  SAMPLER: 'softmax_triplet'
  NUM_INSTANCE: 4
  NUM_WORKERS: 8
```

주요 옵션:
- `SAMPLER`: 샘플링 전략 (softmax/softmax_triplet/triplet)
- `NUM_INSTANCE`: 각 ID당 샘플 수
- `NUM_WORKERS`: 데이터 로딩 워커 수

### 옵티마이저 설정

```yaml
SOLVER:
  OPTIMIZER_NAME: 'Adam'
  MAX_EPOCHS: 120
  BASE_LR: 0.00035
  IMS_PER_BATCH: 64
  STEPS: [40, 70]
  GAMMA: 0.1
```

주요 옵션:
- `OPTIMIZER_NAME`: 옵티마이저 종류 (Adam/SGD)
- `MAX_EPOCHS`: 최대 에포크 수
- `BASE_LR`: 기본 학습률
- `IMS_PER_BATCH`: 배치 크기
- `STEPS`: 학습률 감소 에포크
- `GAMMA`: 학습률 감소 비율

### 평가 설정

```yaml
SOLVER:
  CHECKPOINT_PERIOD: 40
  LOG_PERIOD: 20
  EVAL_PERIOD: 40
```

주요 옵션:
- `CHECKPOINT_PERIOD`: 체크포인트 저장 주기 (에포크)
- `LOG_PERIOD`: 로그 출력 주기 (배치)
- `EVAL_PERIOD`: 평가 실행 주기 (에포크)

### W&B 설정

```yaml
WANDB:
  ENABLE: True
  PROJECT: "reid-aihub-kaist"
  ENTITY: ""
  NAME: ""
  TAGS: ["aihub", "kaist", "person-reid"]
```

주요 옵션:
- `ENABLE`: W&B 활성화 여부
- `PROJECT`: W&B 프로젝트 이름
- `ENTITY`: W&B 팀/사용자 이름
- `NAME`: 실행 이름 (비워두면 자동 생성)
- `TAGS`: 실행 태그

# 학습 실행

## 기본 학습

### 명령어

가장 기본적인 학습 명령은 다음과 같습니다:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml
```

이 명령은 `bounding_box_train_1` 폴더를 사용하여 120 에포크 동안 학습합니다.

### 출력 결과

학습 중 다음과 같은 정보가 출력됩니다:

1. 데이터셋 로딩 정보
2. 모델 구조 및 파라미터 수
3. 배치별 손실 값
4. 에포크별 평가 결과 (mAP, Rank-1, Rank-5)
5. 체크포인트 저장 메시지

## 다른 훈련 폴더 사용

### bounding_box_train_2 사용

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATASETS.TRAIN_FOLDER 'bounding_box_train_2'
```

이 설정은 175개 ID, 18,476개 이미지로 학습합니다.

### bounding_box_train_3 사용

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATASETS.TRAIN_FOLDER 'bounding_box_train_3'
```

이 설정은 175개 ID, 18,495개 이미지로 학습합니다.

## 하이퍼파라미터 조정

### 배치 크기 변경

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.IMS_PER_BATCH 128
```

GPU 메모리에 따라 배치 크기를 조정합니다.

### 학습률 변경

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.BASE_LR 0.0005
```

더 높거나 낮은 학습률로 실험할 수 있습니다.

### 에포크 수 변경

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.MAX_EPOCHS 200
```

더 긴 학습을 원할 경우 에포크 수를 증가시킵니다.

## 복수 파라미터 변경

여러 파라미터를 동시에 변경할 수 있습니다:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATASETS.TRAIN_FOLDER 'bounding_box_train_2' \
  SOLVER.IMS_PER_BATCH 128 \
  SOLVER.BASE_LR 0.0007 \
  SOLVER.MAX_EPOCHS 150 \
  WANDB.NAME "aihub-train2-exp1"
```

# 학습 모니터링

## 로컬 로그

### 콘솔 출력

학습 중 콘솔에서 다음 정보를 확인할 수 있습니다:

- 현재 에포크 및 배치 번호
- 각 손실 항목별 값 (Total Loss, CrossEntropy, Triplet)
- 배치 처리 시간
- 학습률

### 로그 파일

로그는 다음 위치에 저장됩니다:

```
log/aihub_kaist/
├── reid_baseline.log       # 텍스트 로그
└── model_name_*.pth        # 체크포인트
```

## W&B 모니터링

W&B가 활성화된 경우 다음 정보가 실시간으로 업로드됩니다:

### 손실 메트릭
- Total Loss
- CrossEntropy Loss
- Triplet Loss
- Center Loss (사용 시)

### 평가 메트릭
- mAP (mean Average Precision)
- Rank-1 Accuracy
- Rank-5 Accuracy
- Rank-10 Accuracy

### 시스템 메트릭
- GPU 사용률
- GPU 메모리
- CPU 사용률
- 시스템 메모리

### 학습 파라미터
- 학습률 (Learning Rate)
- 에포크 및 배치 정보

## 평가 지표 해석

### mAP (mean Average Precision)

Query 이미지에 대한 Gallery 검색 성능의 전체 평균입니다. 높을수록 좋으며 일반적으로 0.7 이상이면 양호한 성능입니다.

### Rank-1 Accuracy

Query 이미지에 대해 가장 유사한 1개 이미지가 정답인 비율입니다. Person Re-identification에서 가장 중요한 지표 중 하나입니다.

### Rank-5 Accuracy

Query 이미지에 대해 가장 유사한 5개 이미지 중에 정답이 포함된 비율입니다. Rank-1보다 높은 값을 가집니다.

# 고급 설정

## Center Loss 사용

Center Loss를 추가하여 학습 성능을 향상시킬 수 있습니다:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  MODEL.IF_WITH_CENTER 'yes' \
  SOLVER.CENTER_LOSS_WEIGHT 0.0005
```

Center Loss는 같은 ID의 특징을 중심으로 모으는 역할을 합니다.

## 다양한 Sampler 전략

### Softmax Only

분류 문제로만 접근할 경우:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATALOADER.SAMPLER 'softmax'
```

### Triplet Only

순수 메트릭 학습으로만 접근할 경우:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATALOADER.SAMPLER 'triplet'
```

### Softmax + Triplet (권장)

두 접근을 결합한 방식으로 가장 좋은 성능을 보입니다:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATALOADER.SAMPLER 'softmax_triplet'
```

## 입력 크기 조정

더 큰 입력 크기로 성능 향상을 시도할 수 있습니다:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  INPUT.SIZE_TRAIN '[384, 128]' \
  INPUT.SIZE_TEST '[384, 128]'
```

주의: 입력 크기가 클수록 GPU 메모리 사용량이 증가합니다.

## 사전 학습 모델 변경

처음부터 학습하거나 자체 모델로 시작할 수 있습니다:

```bash
# 자체 사전 학습 모델 사용
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  MODEL.PRETRAIN_CHOICE 'self' \
  MODEL.PRETRAIN_PATH 'path/to/your/model.pth'
```

## Warmup 설정

학습 초기 안정성을 위한 Warmup 설정:

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.WARMUP_ITERS 10 \
  SOLVER.WARMUP_METHOD 'linear' \
  SOLVER.WARMUP_FACTOR 0.01
```

# 문제 해결

## Out of Memory (OOM) 에러

GPU 메모리 부족 시 다음 방법을 시도합니다:

### 배치 크기 감소

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.IMS_PER_BATCH 32
```

### 입력 크기 감소

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  INPUT.SIZE_TRAIN '[192, 96]' \
  INPUT.SIZE_TEST '[192, 96]'
```

### NUM_WORKERS 감소

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATALOADER.NUM_WORKERS 4
```

## NumPy 버전 문제

NumPy 2.x와의 호환성 문제가 발생할 경우:

```bash
pip install "numpy<2"
```

## Query 폴더 없음 에러

다음 오류 발생 시:
```
RuntimeError: 'data/aihub_kaist/query' is not available
```

Query 폴더를 압축 해제합니다:

```bash
cd data/aihub_kaist
unzip query.zip
```

## 학습이 너무 느린 경우

### 데이터 로더 워커 증가

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  DATALOADER.NUM_WORKERS 16
```

### 평가 주기 조정

```bash
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  SOLVER.EVAL_PERIOD 60
```

## W&B 로그인 실패

W&B 사용 시 로그인 문제가 있다면:

```bash
# W&B 비활성화
python3 tools/train.py --config_file configs/aihub_kaist.yml \
  WANDB.ENABLE False
```

또는 다시 로그인 시도:

```bash
wandb login --relogin
```

# 참조

## 학습 관련 파일

### 메인 학습 스크립트
- 경로: `tools/train.py`
- `train` 함수 (26-123행): 전체 학습 파이프라인
- W&B 초기화 (28행): `initialize_wandb(cfg)`
- 데이터 로더 생성 (31행): `make_data_loader(cfg)`
- 모델 구성 (34행): `build_model(cfg, num_classes)`
- 학습 실행 (63-74행): `do_train()` 호출
- `main` 함수 (125-161행): 명령행 인자 파싱 및 설정

### 데이터 로더
- 경로: `data/build.py`
- `make_data_loader` 함수 (15-51행): 데이터 로더 생성
- Transform 적용 (16-17행): 훈련/검증 전처리
- 데이터셋 초기화 (20-29행): AIHub KAIST 특수 처리 포함
- Sampler 선택 (33-44행): Softmax vs Triplet 샘플링

### AIHub KAIST 데이터셋
- 경로: `data/datasets/aihub_kaist.py`
- `AIHubKAIST` 클래스 (13-96행): 데이터셋 구현
- 초기화 (23-53행): 훈련 폴더 선택 지원
- 파일명 파싱 (66-94행): Person ID 및 Camera ID 추출

## 설정 파일

### AIHub KAIST 전용 설정
- 경로: `configs/aihub_kaist.yml`
- 모델 설정 (1-6행): 사전 학습 및 손실 함수
- 입력 설정 (8-13행): 이미지 크기 및 증강
- 데이터셋 설정 (15-19행): 데이터셋 이름 및 폴더 선택
- 솔버 설정 (26-56행): 옵티마이저 및 학습 스케줄
- W&B 설정 (67-74행): 실험 추적 설정

### 기본 설정
- 경로: `config/defaults.py`
- 데이터셋 설정 (69-75행): 기본 데이터셋 옵션
- TRAIN_FOLDER 추가 (75행): AIHub KAIST 훈련 폴더 선택

## W&B 통합

### W&B 유틸리티
- 경로: `utils/wandb_utils.py`
- `initialize_wandb` 함수 (26-85행): W&B 초기화
- `log_metrics` 함수 (88-98행): 메트릭 로깅

## 모델 및 손실 함수

### 모델 구성
- 경로: `modeling/__init__.py`
- `build_model` 함수: ResNet-50 백본 기반 모델 생성

### 손실 함수
- 경로: `layers/__init__.py`
- `make_loss`: Softmax + Triplet Loss 구성
- `make_loss_with_center`: Center Loss 추가 구성

## 학습 엔진

### 트레이너
- 경로: `engine/trainer.py`
- `do_train` 함수: 표준 학습 루프
- `do_train_with_center` 함수: Center Loss 포함 학습 루프

## 평가

### 테스트 스크립트
- 경로: `tools/test.py`
- 학습된 모델 평가 및 추론

