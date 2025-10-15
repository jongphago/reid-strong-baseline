# AIHub KAIST 테스트 설정 파일 가이드

## 개요

본 디렉토리에는 AIHub KAIST 데이터셋 테스트를 위한 여러 설정 파일이 포함되어 있습니다. 각 설정 파일은 다른 테스트 시나리오에 최적화되어 있습니다.

## 설정 파일 비교

| 설정 파일 | 에포크 | 배치 | 입력 크기 | Train | Query | Gallery | Sampler | 용도 |
|-----------|--------|------|----------|-------|-------|---------|---------|------|
| test_aihub_kaist.yml | 5 | 32 | 256x128 | train_1 | query | test | softmax_triplet | 표준 테스트 |
| test_aihub_kaist_quick.yml | 2 | 16 | 192x96 | sample | sample | sample | softmax | 빠른 샘플 테스트 |
| test_aihub_kaist_sample.yml | 2 | 16 | 192x96 | sample | sample | sample | softmax_triplet | 샘플 데이터 테스트 |
| test_aihub_kaist_sample_reduced.yml | 2 | 8 | 192x96 | sample | reduced | reduced | softmax | 매우 빠른 테스트 |
| test_aihub_kaist_train2.yml | 5 | 32 | 256x128 | train_2 | query | test | softmax_triplet | Train2 폴더 테스트 |
| test_aihub_kaist_train3.yml | 5 | 32 | 256x128 | train_3 | query | test | softmax_triplet | Train3 폴더 테스트 |
| aihub_kaist.yml | 120 | 64 | 256x128 | train_1 | query | test | softmax_triplet | 전체 학습 |

## 설정 파일 상세

### 1. test_aihub_kaist.yml

**목적**: 표준 테스트 설정

**특징**:
- 5 에포크로 빠른 검증
- 배치 크기 32 (GPU 메모리 절약)
- 매 에포크마다 평가
- 2 에포크마다 체크포인트 저장
- Softmax + Triplet Loss

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml
```

**예상 실행 시간**: 약 20-30분 (GPU 성능에 따라 다름)

### 2. test_aihub_kaist_quick.yml

**목적**: 매우 빠른 테스트 (코드 검증용, Sample 데이터 사용)

**특징**:
- 2 에포크만 실행
- 배치 크기 16 (최소 메모리)
- 입력 크기 192x96 (빠른 처리)
- Random Erasing 비활성화
- Softmax만 사용 (Triplet 없음)
- Sample 데이터 사용 (100 IDs, ~3K 훈련 이미지, ~700 쿼리)
- 매 에포크마다 평가 및 체크포인트

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_quick.yml
```

**예상 실행 시간**: 약 5-10분

**권장 사용 시나리오**:
- 코드 수정 후 빠른 동작 확인
- 새로운 기능 추가 후 기본 검증
- GPU 메모리 부족 환경
- 파이프라인 전체 테스트

### 3. test_aihub_kaist_sample.yml

**목적**: Sample 데이터로 기본 테스트

**특징**:
- 2 에포크 실행
- Sample 데이터 사용 (100 IDs)
  - Train: 2,838 이미지
  - Query: 732 이미지
  - Gallery: 11,047 이미지
- Softmax + Triplet Loss
- 배치 크기 16
- 입력 크기 192x96

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_sample.yml
```

**예상 실행 시간**: 약 5-10분

### 4. test_aihub_kaist_sample_reduced.yml

**목적**: 매우 작은 데이터셋으로 초고속 테스트

**특징**:
- 2 에포크 실행
- Sample Reduced 데이터 사용 (100 IDs)
  - Train: 2,838 이미지
  - Query: 382 이미지 (축소)
  - Gallery: 1,199 이미지 (축소)
- Softmax만 사용
- 배치 크기 8 (매우 작음)
- 입력 크기 192x96

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_sample_reduced.yml
```

**예상 실행 시간**: 약 3-5분

**권장 사용 시나리오**:
- 가장 빠른 파이프라인 검증
- 알고리즘 변경 후 즉시 확인
- CPU 환경에서도 테스트 가능

### 5. test_aihub_kaist_train2.yml

**목적**: bounding_box_train_2 폴더 테스트

**특징**:
- Train2 폴더 사용 (175 PIDs, 18,476 이미지)
- 5 에포크 학습
- 배치 크기 32
- 다른 데이터 분할로 성능 비교 가능

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_train2.yml
```

**예상 실행 시간**: 약 20-30분

### 6. test_aihub_kaist_train3.yml

**목적**: bounding_box_train_3 폴더 테스트

**특징**:
- Train3 폴더 사용 (175 PIDs, 18,495 이미지)
- 5 에포크 학습
- 배치 크기 32
- Train2와 유사하나 다른 데이터 분할

**사용 예시**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_train3.yml
```

**예상 실행 시간**: 약 20-30분

## 사용 가이드

### 기본 테스트 워크플로우

1. **빠른 동작 확인**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_quick.yml
```

2. **표준 테스트**:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml
```

3. **훈련 폴더 비교**:
```bash
# Train1
python3 tools/train.py --config_file configs/test_aihub_kaist.yml

# Train2
python3 tools/train.py --config_file configs/test_aihub_kaist_train2.yml

# Train3
python3 tools/train.py --config_file configs/test_aihub_kaist_train3.yml
```

### W&B 비활성화

테스트 시 W&B를 사용하지 않으려면:

```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  WANDB.ENABLE False
```

### 파라미터 오버라이드

설정 파일의 특정 값을 명령줄에서 변경:

```bash
# 에포크 수 변경
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  SOLVER.MAX_EPOCHS 10

# 배치 크기 변경
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  SOLVER.IMS_PER_BATCH 16

# 평가 주기 변경
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  SOLVER.EVAL_PERIOD 2
```

## 출력 디렉토리

각 설정 파일은 서로 다른 출력 디렉토리를 사용합니다:

- `test_aihub_kaist.yml` → `./log/test_aihub_kaist/`
- `test_aihub_kaist_quick.yml` → `./log/test_aihub_kaist_quick/`
- `test_aihub_kaist_train2.yml` → `./log/test_aihub_kaist_train2/`
- `test_aihub_kaist_train3.yml` → `./log/test_aihub_kaist_train3/`

각 디렉토리에는 다음 파일들이 저장됩니다:
- `reid_baseline.log`: 학습 로그
- `resnet50_model_X.pth`: 체크포인트 파일

## 성능 평가

테스트 설정 파일로 학습 후 모델을 평가하려면:

```bash
python3 tools/test.py \
  --config_file configs/test_aihub_kaist.yml \
  TEST.WEIGHT './log/test_aihub_kaist/resnet50_model_5.pth'
```

## 주의사항

### GPU 메모리

- RTX 3090 (24GB): 모든 설정 사용 가능
- RTX 3080 (10GB): test_aihub_kaist_quick.yml 권장
- GPU 메모리 부족 시 배치 크기를 줄이거나 입력 크기를 축소

### 데이터셋 준비

모든 테스트 실행 전 다음 폴더가 존재해야 합니다:

```
data/aihub_kaist/
├── bounding_box_train_1/
├── bounding_box_train_2/
├── bounding_box_train_3/
├── query/
└── bounding_box_test/
```

query 폴더가 없다면:
```bash
cd data/aihub_kaist
unzip query.zip
```

### 사전 학습 모델

모든 설정은 ImageNet 사전 학습 ResNet-50을 사용합니다:
```
/home/jongphago/.torch/models/resnet50.pth
```

다른 경로를 사용할 경우 설정 파일에서 `MODEL.PRETRAIN_PATH`를 수정하거나 명령줄에서 오버라이드:

```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  MODEL.PRETRAIN_PATH '/path/to/your/resnet50.pth'
```

## 문제 해결

### Out of Memory

배치 크기를 줄입니다:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  SOLVER.IMS_PER_BATCH 16
```

또는 quick 설정 사용:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist_quick.yml
```

### 느린 학습 속도

워커 수를 증가시킵니다:
```bash
python3 tools/train.py --config_file configs/test_aihub_kaist.yml \
  DATALOADER.NUM_WORKERS 16
```

### NumPy 버전 문제

```bash
pip install "numpy<2"
```

## 참고

- 전체 학습: `configs/aihub_kaist.yml` (120 에포크)
- 학습 가이드: `reports/AIHub_KAIST_모델_학습_가이드.md`
- 데이터셋 추가 과정: `reports/AIHub_KAIST_데이터셋_추가_과정.md`

