#!/bin/bash
# Phase 3 GPU Environment Installation Script
# For fresh GPU servers (AutoDL, HengYuan, etc.)

echo "================================================================"
echo "         Phase 3 GPU Environment Installation Script"
echo "         Video Speed Estimation System - GPU Setup"
echo "================================================================"
echo ""
echo "This script will install all dependencies in a fresh environment"
echo "Suitable for: AutoDL, HengYuan Cloud, Aliyun GPU servers"
echo ""
echo "Installation includes:"
echo "  1. PyTorch 2.x + CUDA 11.8 (GPU version)"
echo "  2. Torchvision + RAFT support"
echo "  3. Depth Anything V2 dependencies"
echo "  4. YOLOv8 + ByteTrack"
echo "  5. All Phase 1/2/3 dependencies"
echo ""
echo "================================================================"
read -p "Press Enter to continue..."

echo ""
echo "================================================================"
echo "[1/5] Checking Python version..."
echo "================================================================"
python3 --version || python --version
if [ $? -ne 0 ]; then
    echo "❌ Error: Python not found. Please install Python 3.8+"
    exit 1
fi
echo "✅ Python version check passed"

echo ""
echo "================================================================"
echo "[2/5] Upgrading pip..."
echo "================================================================"
python3 -m pip install --upgrade pip || python -m pip install --upgrade pip
echo "✅ pip upgraded"

echo ""
echo "================================================================"
echo "[3/5] Installing PyTorch (GPU version - CUDA 11.8)"
echo "================================================================"
echo "This is the largest package (~2GB), please wait..."
echo ""
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 || \
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if [ $? -ne 0 ]; then
    echo "❌ PyTorch installation failed. Check network connection."
    exit 1
fi
echo "✅ PyTorch GPU version installed"

echo ""
echo "================================================================"
echo "[4/5] Installing Depth Anything V2 dependencies..."
echo "================================================================"
pip3 install timm>=0.9.0 transformers>=4.30.0 huggingface-hub>=0.19.0 || \
pip install timm>=0.9.0 transformers>=4.30.0 huggingface-hub>=0.19.0
if [ $? -ne 0 ]; then
    echo "❌ Depth dependencies installation failed"
    exit 1
fi
echo "✅ Depth Anything dependencies installed"

echo ""
echo "================================================================"
echo "[5/5] Installing other project dependencies..."
echo "================================================================"
pip3 install ultralytics>=8.0.0 opencv-python>=4.5.0 numpy>=1.21.0 supervision>=0.11.0 Pillow>=8.0.0 matplotlib>=3.5.0 pandas>=1.3.0 tqdm>=4.60.0 || \
pip install ultralytics>=8.0.0 opencv-python>=4.5.0 numpy>=1.21.0 supervision>=0.11.0 Pillow>=8.0.0 matplotlib>=3.5.0 pandas>=1.3.0 tqdm>=4.60.0
if [ $? -ne 0 ]; then
    echo "❌ Project dependencies installation failed"
    exit 1
fi
echo "✅ All project dependencies installed"

echo ""
echo "================================================================"
echo "[Verification] Checking GPU availability..."
echo "================================================================"
cd ..
if [ -f "check_gpu_status.py" ]; then
    python3 check_gpu_status.py || python check_gpu_status.py
else
    echo "⚠️  Warning: check_gpu_status.py not found, manual verification"
    python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || \
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
fi
cd gpu

echo ""
echo "================================================================"
echo "✅ GPU Environment Installation Complete!"
echo "================================================================"
echo ""
echo "📋 Installed components:"
echo "  ✅ PyTorch 2.x + CUDA 11.8"
echo "  ✅ Torchvision (RAFT optical flow)"
echo "  ✅ Depth Anything V2"
echo "  ✅ YOLOv8 + ByteTrack"
echo "  ✅ All Phase 1/2/3 dependencies"
echo ""
echo "🚀 Next steps:"
echo "  1. Confirm GPU is available (should see CUDA: True above)"
echo "  2. Return to scripts: cd .."
echo "  3. Run test: python test_dependencies.py"
echo "  4. Return to root and start: cd .. && python main.py"
echo ""
echo "================================================================"
