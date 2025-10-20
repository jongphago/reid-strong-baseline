#!/usr/bin/env python
# encoding: utf-8
"""
GPU 메모리 정리 기능 단독 테스트
sweep.py의 메모리 정리 로직을 간단하게 테스트

실행: python test_gpu_cleanup.py
"""

import gc
import time
import torch
import torch.nn as nn


def get_gpu_memory():
    """현재 GPU 메모리 사용량 반환 (MB)"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        return allocated, reserved
    return 0, 0


def cleanup_gpu_memory():
    """sweep.py와 동일한 GPU 메모리 정리 로직"""
    print("\nCleaning up GPU memory...")
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated, reserved = get_gpu_memory()
            print(f"GPU memory cleared. Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        gc.collect()
    except Exception as cleanup_error:
        print(f"Warning: GPU cleanup failed: {cleanup_error}")
    
    print("Waiting 5 seconds before next run...")
    time.sleep(5)
    print("Ready for next run.\n")


def simulate_training_run(run_id, size_mb=100):
    """학습 실행 시뮬레이션 (메모리 할당)"""
    print(f"\n{'='*60}")
    print(f"Run {run_id}: Simulating training...")
    print(f"{'='*60}")
    
    # 초기 메모리 상태
    allocated, reserved = get_gpu_memory()
    print(f"Before training - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    # 모델 생성 (메모리 할당)
    if torch.cuda.is_available():
        # 약 size_mb MB 할당
        tensor_size = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
        dummy_tensor = torch.randn(tensor_size, device='cuda')
        
        allocated, reserved = get_gpu_memory()
        print(f"After allocation - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        
        # 간단한 연산
        result = dummy_tensor * 2
        time.sleep(1)  # 학습 시뮬레이션
        
        # 명시적 삭제
        del dummy_tensor
        del result
        
    print(f"Run {run_id} completed.")
    
    # 정리 전 메모리 상태
    allocated, reserved = get_gpu_memory()
    print(f"After deletion - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def main():
    print("="*60)
    print("GPU 메모리 정리 기능 테스트")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CUDA를 사용할 수 없습니다!")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"Total memory: {total_memory:.2f} MB")
    
    # 초기 상태
    print("\n초기 GPU 메모리 상태:")
    allocated, reserved = get_gpu_memory()
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    # 3번의 학습 실행 시뮬레이션
    num_runs = 3
    allocation_sizes = [150, 200, 150]  # MB
    
    print(f"\n{num_runs}번의 실험을 연속으로 실행합니다...")
    print("각 실험 후 메모리가 정리되는지 확인하세요.\n")
    
    for i in range(num_runs):
        simulate_training_run(i + 1, allocation_sizes[i])
        cleanup_gpu_memory()
    
    # 최종 상태
    print("="*60)
    print("테스트 완료!")
    print("="*60)
    print("\n최종 GPU 메모리 상태:")
    allocated, reserved = get_gpu_memory()
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    
    # 결과 평가
    print("\n결과 평가:")
    if allocated < 50:  # 50MB 미만이면 정리 성공
        print("✅ GPU 메모리가 성공적으로 정리되었습니다!")
    else:
        print("⚠️  GPU 메모리가 완전히 정리되지 않았을 수 있습니다.")
        print(f"   할당된 메모리: {allocated:.2f} MB")


if __name__ == "__main__":
    main()

