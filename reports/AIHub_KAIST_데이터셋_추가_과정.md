# 목차

- [목차](#목차)
- [개요](#개요)
- [데이터셋 구조 분석](#데이터셋-구조-분석)
- [구현 과정](#구현-과정)
- [테스트 및 검증](#테스트-및-검증)
- [결론](#결론)
- [참조](#참조)

# 개요

## 배경

Reid Strong Baseline 프로젝트에 새로운 AIHub KAIST Person Re-identification 데이터셋을 추가하는 작업을 수행하였습니다. 기존 프로젝트는 Market1501, DukeMTMC, MSMT17 등의 표준 ReID 데이터셋을 지원하고 있었으며, 동일한 구조를 따라 새로운 데이터셋을 통합하였습니다.

## 목적

- AIHub KAIST 데이터셋을 프로젝트에 통합
- 기존 데이터셋과 동일한 인터페이스 제공
- 여러 훈련 데이터 폴더 선택 기능 구현
- 최소한의 코드 변경으로 확장성 유지

## 작업 범위

- 데이터셋 클래스 구현
- 데이터 로더 수정
- 설정 시스템 확장
- 테스트 및 검증

# 데이터셋 구조 분석

## 디렉토리 구조

AIHub KAIST 데이터셋은 다음과 같은 구조를 가지고 있습니다:

```
data/aihub_kaist/
├── bounding_box_train_1/    # 훈련 데이터 (옵션 1)
├── bounding_box_train_2/    # 훈련 데이터 (옵션 2)
├── bounding_box_train_3/    # 훈련 데이터 (옵션 3)
├── query/                   # 쿼리 이미지
└── bounding_box_test/       # 갤러리 이미지
```

## 파일명 패턴

데이터셋의 이미지 파일명은 다음 형식을 따릅니다:

```
{pid}_c{camid}s{seq}_{frame}.jpg
```

예시:
- `0000_c07s32_005875.jpg`: Person ID 0, Camera 7, Sequence 32, Frame 5875
- `0001_c04s31_004050.jpg`: Person ID 1, Camera 4, Sequence 31, Frame 4050

## 데이터 통계

세 가지 훈련 폴더 옵션의 통계:

| 구분 | Train PIDs | Train 이미지 | Query 이미지 | Gallery 이미지 | 카메라 수 |
|------|-----------|-------------|-------------|---------------|----------|
| train_1 | 188 | 20,119 | 1,065 | 15,385 | 16 |
| train_2 | 175 | 18,476 | 1,065 | 15,385 | 16 |
| train_3 | 175 | 18,495 | 1,065 | 15,385 | 16 |

# 구현 과정

## 1단계: 데이터셋 클래스 생성

### BaseImageDataset 상속

기존 데이터셋들과 동일하게 `BaseImageDataset`를 상속하여 구현하였습니다. 이를 통해 공통 인터페이스를 유지하고 코드 재사용성을 확보하였습니다.

### 초기화 메서드

`AIHubKAIST` 클래스는 다음과 같은 파라미터를 받습니다:
- `root`: 데이터 루트 디렉토리 (기본값: './data')
- `train_folder`: 훈련 폴더 선택 (기본값: 'bounding_box_train_1')
- `verbose`: 통계 출력 여부 (기본값: True)

### 파일명 파싱 로직

정규표현식을 사용하여 파일명에서 Person ID와 Camera ID를 추출합니다:

```python
pattern = re.compile(r'(\d+)_c(\d+)s')
```

이 패턴은 `{pid}_c{camid}s` 형식을 매칭하여 필요한 정보를 추출합니다.

### Relabeling 처리

훈련 데이터의 경우 Person ID를 0부터 시작하는 연속된 레이블로 변환합니다. 이는 분류 손실 함수에서 요구하는 형식입니다:

```python
pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
```

## 2단계: 데이터셋 등록

### 팩토리 패턴 활용

`data/datasets/__init__.py`에서 팩토리 패턴을 사용하여 데이터셋을 등록합니다:

1. 클래스 임포트 추가
2. `__factory` 딕셔너리에 등록
3. `init_dataset` 함수를 통한 인스턴스 생성

이를 통해 문자열 이름으로 데이터셋을 동적으로 생성할 수 있습니다.

## 3단계: 설정 시스템 확장

### 기본 설정 추가

`config/defaults.py`에 새로운 설정 옵션을 추가하였습니다:

```python
_C.DATASETS.TRAIN_FOLDER = 'bounding_box_train_1'
```

이 설정은 AIHub KAIST 데이터셋에서 어떤 훈련 폴더를 사용할지 지정합니다.

### 데이터 로더 수정

`data/build.py`의 `make_data_loader` 함수를 수정하여 AIHub KAIST 데이터셋 사용 시 `train_folder` 파라미터를 전달하도록 하였습니다:

```python
dataset_kwargs = {'root': cfg.DATASETS.ROOT_DIR}
if cfg.DATASETS.NAMES == 'aihub_kaist':
    dataset_kwargs['train_folder'] = cfg.DATASETS.TRAIN_FOLDER
```

## 4단계: 설정 파일 생성

### AIHub KAIST 전용 설정

`configs/aihub_kaist.yml` 파일을 생성하여 데이터셋에 최적화된 기본 설정을 제공합니다:

- 데이터셋 이름: `aihub_kaist`
- 입력 크기: 256x128
- Triplet Loss 사용
- Adam 옵티마이저
- 120 에포크 훈련
- W&B 통합 활성화

# 테스트 및 검증

## 테스트 스크립트 작성

데이터셋 로딩을 검증하기 위한 테스트 스크립트를 작성하여 실행하였습니다. 테스트는 다음 항목을 확인합니다:

1. 세 가지 훈련 폴더 모두 정상 로딩
2. 데이터셋 통계 정확성
3. 이미지 경로 및 레이블 정확성
4. Query와 Gallery 데이터 일관성

## 테스트 결과

모든 훈련 폴더에서 데이터가 정상적으로 로드되었으며, 다음과 같은 결과를 확인하였습니다:

- bounding_box_train_1: 188 PIDs, 20,119 이미지
- bounding_box_train_2: 175 PIDs, 18,476 이미지
- bounding_box_train_3: 175 PIDs, 18,495 이미지

모든 설정에서 Query 1,065개, Gallery 15,385개의 이미지가 일관되게 로드되었습니다.

## 샘플 데이터 검증

샘플 데이터를 출력하여 Person ID, Camera ID, 파일 경로가 올바르게 파싱되는지 확인하였습니다:

- Train: `0074_c14s32_001450.jpg` → PID=74, CamID=14
- Query: `1099_c09s43_006800.jpg` → PID=1099, CamID=9

# 결론

## 구현 결과

AIHub KAIST Person Re-identification 데이터셋을 Reid Strong Baseline 프로젝트에 성공적으로 통합하였습니다. 구현된 시스템은 다음과 같은 특징을 가집니다:

1. 기존 데이터셋과 동일한 인터페이스 제공
2. 여러 훈련 폴더 선택 가능한 유연성
3. 최소한의 코드 변경으로 확장성 유지
4. 완전한 W&B 통합 지원

## 확장 가능성

구현된 구조는 향후 다른 데이터셋 추가 시에도 동일한 패턴으로 쉽게 확장할 수 있습니다. 새로운 데이터셋 추가를 위해서는:

1. `data/datasets/` 폴더에 새 클래스 파일 생성
2. `BaseImageDataset` 상속 및 필요한 메서드 구현
3. `__init__.py`에 등록
4. 전용 config 파일 생성

## 활용 방법

사용자는 세 가지 방법으로 데이터셋을 활용할 수 있습니다:

1. 전용 설정 파일 사용: `configs/aihub_kaist.yml`
2. 명령줄 인자로 훈련 폴더 변경
3. 기존 설정 파일 수정하여 사용

이를 통해 다양한 실험 시나리오에 대응할 수 있습니다.

# 참조

## 생성된 파일

### 데이터셋 클래스
- 경로: `data/datasets/aihub_kaist.py`
- 주요 클래스: `AIHubKAIST` (13-96행)
- `__init__` 메서드 (23-53행): 데이터셋 초기화 및 로딩
- `_check_before_run` 메서드 (55-64행): 디렉토리 존재 확인
- `_process_dir` 메서드 (66-94행): 파일명 파싱 및 데이터 처리

### 설정 파일
- 경로: `configs/aihub_kaist.yml`
- DATASETS.NAMES (16행): 데이터셋 이름 지정
- DATASETS.TRAIN_FOLDER (19행): 훈련 폴더 선택

## 수정된 파일

### 데이터셋 등록
- 경로: `data/datasets/__init__.py`
- Import 추가 (11행): `from .aihub_kaist import AIHubKAIST`
- Factory 등록 (20행): `'aihub_kaist': AIHubKAIST`

### 데이터 로더
- 경로: `data/build.py`
- `make_data_loader` 함수 (15-29행): train_folder 파라미터 전달 로직
- 조건부 파라미터 추가 (20-23행): AIHub KAIST 전용 설정

### 기본 설정
- 경로: `config/defaults.py`
- TRAIN_FOLDER 설정 추가 (75행): 훈련 폴더 선택 옵션

## 참고 데이터셋

### Market1501 구현
- 경로: `data/datasets/market1501.py`
- 클래스 정의 (15-86행): 참조한 구현 패턴

### BaseImageDataset
- 경로: `data/datasets/bases.py`
- BaseImageDataset 클래스 (46-63행): 상속받은 베이스 클래스
- `get_imagedata_info` 메서드 (15-25행): 데이터셋 통계 계산
- `print_dataset_statistics` 메서드 (51-63행): 통계 출력

