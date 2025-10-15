#!/bin/bash
# AIHub KAIST 데이터셋 테스트 설정 실행 스크립트

set -e  # 에러 발생 시 스크립트 중단

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AIHub KAIST 테스트 설정 실행 스크립트${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 사용법 출력
usage() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  quick     - 빠른 테스트 (2 에포크, 작은 배치)"
    echo "  standard  - 표준 테스트 (5 에포크, train_1)"
    echo "  train2    - Train2 폴더 테스트 (5 에포크)"
    echo "  train3    - Train3 폴더 테스트 (5 에포크)"
    echo "  all       - 모든 훈련 폴더 순차 테스트 (train_1/2/3)"
    echo "  compare   - 모든 설정 비교 실행"
    echo ""
    echo "예시:"
    echo "  $0 quick      # 빠른 테스트만 실행"
    echo "  $0 standard   # 표준 테스트 실행"
    echo "  $0 all        # train_1/2/3 모두 테스트"
    exit 1
}

# Python 실행 파일 확인
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}오류: Python을 찾을 수 없습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}Python 실행 파일: $PYTHON${NC}"
echo ""

# 데이터셋 존재 확인
check_dataset() {
    echo -e "${YELLOW}데이터셋 존재 여부 확인...${NC}"
    
    if [ ! -d "data/aihub_kaist" ]; then
        echo -e "${RED}오류: data/aihub_kaist 폴더가 없습니다.${NC}"
        exit 1
    fi
    
    if [ ! -d "data/aihub_kaist/query" ]; then
        echo -e "${RED}경고: data/aihub_kaist/query 폴더가 없습니다.${NC}"
        echo -e "${YELLOW}query.zip을 압축 해제하시겠습니까? (y/n)${NC}"
        read -r answer
        if [ "$answer" = "y" ]; then
            cd data/aihub_kaist
            unzip query.zip
            cd ../..
            echo -e "${GREEN}Query 폴더 압축 해제 완료${NC}"
        else
            echo -e "${RED}Query 폴더가 없으면 실행할 수 없습니다.${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✓ 데이터셋 확인 완료${NC}"
    echo ""
}

# 빠른 테스트 실행
run_quick_test() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}빠른 테스트 실행 (2 에포크)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    $PYTHON tools/train.py --config_file configs/test_aihub_kaist_quick.yml
    
    echo -e "${GREEN}✓ 빠른 테스트 완료${NC}"
    echo -e "${YELLOW}결과 위치: ./log/test_aihub_kaist_quick/${NC}"
    echo ""
}

# 표준 테스트 실행
run_standard_test() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}표준 테스트 실행 (5 에포크, Train1)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    $PYTHON tools/train.py --config_file configs/test_aihub_kaist.yml
    
    echo -e "${GREEN}✓ 표준 테스트 완료${NC}"
    echo -e "${YELLOW}결과 위치: ./log/test_aihub_kaist/${NC}"
    echo ""
}

# Train2 테스트 실행
run_train2_test() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Train2 폴더 테스트 (5 에포크)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    $PYTHON tools/train.py --config_file configs/test_aihub_kaist_train2.yml
    
    echo -e "${GREEN}✓ Train2 테스트 완료${NC}"
    echo -e "${YELLOW}결과 위치: ./log/test_aihub_kaist_train2/${NC}"
    echo ""
}

# Train3 테스트 실행
run_train3_test() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Train3 폴더 테스트 (5 에포크)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    $PYTHON tools/train.py --config_file configs/test_aihub_kaist_train3.yml
    
    echo -e "${GREEN}✓ Train3 테스트 완료${NC}"
    echo -e "${YELLOW}결과 위치: ./log/test_aihub_kaist_train3/${NC}"
    echo ""
}

# 모든 테스트 실행
run_all_tests() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}모든 훈련 폴더 순차 테스트${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    run_standard_test
    run_train2_test
    run_train3_test
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}모든 테스트 완료!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}결과 요약:${NC}"
    echo -e "  Train1: ./log/test_aihub_kaist/"
    echo -e "  Train2: ./log/test_aihub_kaist_train2/"
    echo -e "  Train3: ./log/test_aihub_kaist_train3/"
    echo ""
}

# 비교 테스트 실행
run_compare_tests() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}모든 설정 비교 실행${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    run_quick_test
    run_standard_test
    run_train2_test
    run_train3_test
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}모든 비교 테스트 완료!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${YELLOW}결과 요약:${NC}"
    echo -e "  Quick:  ./log/test_aihub_kaist_quick/"
    echo -e "  Train1: ./log/test_aihub_kaist/"
    echo -e "  Train2: ./log/test_aihub_kaist_train2/"
    echo -e "  Train3: ./log/test_aihub_kaist_train3/"
    echo ""
}

# 메인 로직
if [ $# -eq 0 ]; then
    usage
fi

# 데이터셋 확인
check_dataset

# 옵션에 따라 실행
case "$1" in
    quick)
        run_quick_test
        ;;
    standard)
        run_standard_test
        ;;
    train2)
        run_train2_test
        ;;
    train3)
        run_train3_test
        ;;
    all)
        run_all_tests
        ;;
    compare)
        run_compare_tests
        ;;
    *)
        echo -e "${RED}알 수 없는 옵션: $1${NC}"
        echo ""
        usage
        ;;
esac

echo -e "${GREEN}스크립트 실행 완료!${NC}"

