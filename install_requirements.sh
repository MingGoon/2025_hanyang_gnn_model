# 객체 움직임 감지 시스템 설치 스크립트
echo "=============================================="
echo "🚀 객체 움직임 감지 시스템 설치 시작"
echo "=============================================="

# Python 버전 확인
echo "📋 Python 버전 확인 중..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "현재 Python 버전: $python_version"

# 최소 요구사항 확인 (Python 3.8+)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python 버전 요구사항 충족"
else
    echo "❌ Python 3.8 이상이 필요합니다."
    exit 1
fi

# pip 업그레이드
echo "📦 pip 업그레이드 중..."
python3 -m pip install --upgrade pip

# PyTorch 및 CUDA 지원 확인
echo "🔥 PyTorch 설치 중..."
echo "CUDA 사용 가능 여부 확인 중..."

# CUDA 확인
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU 감지됨. CUDA 버전 설치합니다."
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  GPU가 감지되지 않았습니다. CPU 버전을 설치합니다."
    python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# PyTorch Geometric 설치
echo "🕸️  PyTorch Geometric 설치 중..."
python3 -m pip install torch-geometric

# 나머지 라이브러리 설치
echo "📚 추가 라이브러리 설치 중..."
python3 -m pip install -r requirements.txt

# 설치 확인
echo "🔍 설치 확인 중..."
python3 -c "
import torch
import torch_geometric
import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib
print('✅ 모든 핵심 라이브러리가 성공적으로 설치되었습니다!')
print(f'PyTorch 버전: {torch.__version__}')
print(f'CUDA 사용 가능: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 개수: {torch.cuda.device_count()}')
"

echo "=============================================="
echo "🎉 설치 완료!"
echo "=============================================="
echo "다음 명령어로 시작하세요:"
echo "python validate_data.py --base_dir ./your_dataset"
echo "python train_data.py --base_dir ./your_dataset" 