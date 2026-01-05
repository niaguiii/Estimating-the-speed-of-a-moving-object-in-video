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
echo "[1/3] Uninstalling CPU version..."
pip uninstall torch torchvision -y

echo ""
echo "[2/3] Installing GPU version (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "[3/3] Verifying GPU support..."
cd ..
python3 check_gpu_status.py || python check_gpu_status.py
cd gpu

echo ""
echo "================================================"
echo "Done! If you see 'CUDA Available: True'"
echo "GPU version is installed successfully!"
echo "================================================"
