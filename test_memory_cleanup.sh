#!/bin/bash
# GPU 메모리 정리 기능 테스트 스크립트
# 3개 실험 x 20 epoch = 약 15-20분 소요 예상

echo "=========================================="
echo "GPU 메모리 정리 기능 테스트"
echo "=========================================="
echo ""
echo "테스트 설정:"
echo "  - 실험 횟수: 3회"
echo "  - 각 실험: 20 epoch"
echo "  - 배치 크기: 64, 48, 64"
echo "  - 예상 소요 시간: 15-20분"
echo ""
echo "확인 사항:"
echo "  1. 각 실험 종료 후 'Cleaning up GPU memory...' 메시지"
echo "  2. GPU 메모리 사용량 로그"
echo "  3. 'Waiting 5 seconds before next run...' 대기"
echo "  4. OOM 에러 없이 3개 실험 모두 완료"
echo ""
echo "=========================================="
echo ""

# 이전 테스트 결과 삭제 (선택사항)
if [ -d "./log/sweep_memory_test" ]; then
    echo "이전 테스트 결과를 삭제합니다..."
    rm -rf ./log/sweep_memory_test
    echo "삭제 완료."
    echo ""
fi

# GPU 상태 확인
echo "현재 GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

# 테스트 실행
echo "테스트 시작..."
echo ""

python tools/sweep.py \
    --config_file configs/test_memory.yml \
    --sweep-config sweeps/sweep_memory_test.yaml

echo ""
echo "=========================================="
echo "테스트 완료!"
echo "=========================================="
echo ""

# 결과 확인
if [ -d "./log/sweep_memory_test/sweep" ]; then
    echo "실험 결과:"
    for dir in ./log/sweep_memory_test/sweep/*/*/; do
        if [ -d "$dir" ]; then
            name=$(basename "$dir")
            epochs=$(grep -c "Validation Results" "$dir/log.txt" 2>/dev/null || echo "0")
            oom=$(grep -c "OutOfMemory" "$dir/log.txt" 2>/dev/null || echo "0")
            cleaned=$(grep -c "Cleaning up GPU memory" "$dir/../../../*.log" 2>/dev/null || echo "?")
            
            if [ "$oom" -gt 0 ]; then
                echo "  ❌ $name: $epochs validations, OOM 발생!"
            elif [ "$epochs" -ge 2 ]; then
                echo "  ✅ $name: $epochs validations, 성공"
            else
                echo "  ⚠️  $name: $epochs validations, 미완료"
            fi
        fi
    done
else
    echo "결과 디렉토리를 찾을 수 없습니다."
fi

echo ""
echo "최종 GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo ""

echo "자세한 로그는 ./log/sweep_memory_test/ 에서 확인하세요."

