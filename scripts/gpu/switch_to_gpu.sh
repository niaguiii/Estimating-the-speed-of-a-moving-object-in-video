#!/bin/bash
# Switch to GPU version of PyTorch
# Run this script ONLY when you have rented a GPU server

echo "================================================"
echo "Switch to GPU version PyTorch"
echo "================================================"
echo ""
echo "This will uninstall CPU version and install GPU version"
echo "⚠️  WARNING: Only run this on a GPU server!"
echo ""
read -p "Press Enter to continue..."

echo ""
echo "[1/4] Detecting GPU model..."
nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1
echo ""
echo "IMPORTANT:"
echo "  - RTX 50 series (5090, 5080, etc.): Choose 1 (CUDA 12.8)"
echo "  - RTX 40 series and below: Choose 2 (CUDA 11.8)"
echo ""
read -p "Choose CUDA version [1=12.8, 2=11.8]: " cuda_choice
echo ""

echo ""
echo "[2/4] Uninstalling CPU version..."
pip uninstall torch torchvision -y

echo ""
echo "[3/4] Installing GPU version of PyTorch..."
if [ "$cuda_choice" = "1" ]; then
    echo "Installing PyTorch with CUDA 12.8 (for RTX 50 series)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    echo "Installing PyTorch with CUDA 11.8 (for RTX 40 series and below)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "[Supplement] Installing Mode 5/6 Metric3D v2 dependencies..."
pip install mmengine>=0.7.0 mmcv>=2.0.0 scipy>=1.7.0 einops>=0.6.0

echo ""
echo "[4/4] Verifying GPU support..."
cd ..
python3 check_gpu_status.py || python check_gpu_status.py
cd gpu

echo ""
echo "================================================"
echo "Done! If you see 'CUDA Available: True'"
echo "GPU version is installed successfully!"
echo "================================================"
