# encoding: utf-8
"""
ONNX 변환 스크립트
학습된 PyTorch 모델을 ONNX 형식으로 변환합니다.
"""

import argparse
import os
import sys
import torch

sys.path.append('.')
from config import cfg
from modeling import build_model


def convert_to_onnx(cfg, model_path, output_path, input_size):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        cfg: 설정 객체
        model_path: 학습된 모델 파일 경로 (.pth)
        output_path: 출력할 ONNX 파일 경로 (.onnx)
        input_size: 입력 이미지 크기 (height, width)
    """
    
    # 모델 파일 로드하여 클래스 수 추정
    print(f"모델 파일 로드 중: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # checkpoint가 state_dict인지 확인
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # 모델 객체인 경우
        state_dict = checkpoint.state_dict()
    
    # classifier의 weight shape에서 클래스 수 추정
    # classifier.weight의 shape는 [num_classes, feature_dim]
    num_classes = None
    for key in state_dict.keys():
        if 'classifier.weight' in key:
            num_classes = state_dict[key].shape[0]
            print(f"감지된 클래스 수: {num_classes}")
            break
    
    if num_classes is None:
        raise ValueError("모델에서 클래스 수를 찾을 수 없습니다.")
    
    # 모델 생성
    print("모델 구조 생성 중...")
    model = build_model(cfg, num_classes)
    
    # 학습된 가중치 로드
    model.load_state_dict(state_dict)
    
    # evaluation 모드로 설정
    model.eval()
    
    # 더미 입력 생성 (batch_size=1, channels=3, height, width)
    height, width = input_size
    dummy_input = torch.randn(1, 3, height, width)
    
    print(f"입력 크기: {dummy_input.shape}")
    print("ONNX 변환 시작...")
    
    # ONNX로 변환
    torch.onnx.export(
        model,                      # 변환할 모델
        dummy_input,                # 더미 입력
        output_path,                # 출력 파일 경로
        export_params=True,         # 학습된 파라미터도 저장
        opset_version=11,           # ONNX 버전
        do_constant_folding=True,   # 최적화
        input_names=['input'],      # 입력 이름
        output_names=['output'],    # 출력 이름
        dynamic_axes={              # 동적 배치 크기 지원
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX 변환 완료: {output_path}")
    
    # 변환된 파일 정보 출력
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB 단위
    print(f"파일 크기: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="ReID 모델을 ONNX로 변환")
    parser.add_argument(
        "--config_file", 
        default="configs/softmax_triplet.yml", 
        help="설정 파일 경로", 
        type=str
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="변환할 모델 파일 경로 (.pth)",
        type=str
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="출력 ONNX 파일 경로",
        type=str
    )
    parser.add_argument(
        "opts", 
        help="명령줄을 통한 설정 옵션 수정", 
        default=None,
        nargs=argparse.REMAINDER
    )
    
    args = parser.parse_args()
    
    # 설정 로드
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    print("=" * 60)
    print("ONNX 변환 설정:")
    print(f"  설정 파일: {args.config_file}")
    print(f"  모델 파일: {args.model_path}")
    print(f"  출력 경로: {args.output_path}")
    print(f"  입력 크기: {cfg.INPUT.SIZE_TEST}")
    print(f"  모델 이름: {cfg.MODEL.NAME}")
    print(f"  Neck: {cfg.MODEL.NECK}")
    print(f"  Neck Feat: {cfg.TEST.NECK_FEAT}")
    print("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
    
    # 변환 실행
    convert_to_onnx(
        cfg=cfg,
        model_path=args.model_path,
        output_path=args.output_path,
        input_size=cfg.INPUT.SIZE_TEST
    )
    
    print("\n변환이 성공적으로 완료되었습니다!")


if __name__ == '__main__':
    main()

