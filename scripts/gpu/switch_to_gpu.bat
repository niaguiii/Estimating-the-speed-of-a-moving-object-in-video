@echo off
chcp 65001 >nul
echo ================================================
echo 切换到 GPU 版本 PyTorch
echo ================================================
echo.
echo 此脚本将卸载 CPU 版本的 PyTorch
echo 并安装支持 CUDA 的 GPU 版本
echo.
echo ⚠️  注意：仅在租用 GPU 服务器后运行此脚本！
echo.
pause

echo.
echo [1/4] 检测GPU型号...
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul
echo.
echo 重要提示：
echo   - RTX 50系列 (5090, 5080等): 选择 1 (CUDA 12.8)
echo   - RTX 40系列及以下 (4090, 3090等): 选择 2 (CUDA 11.8)
echo.
set /p cuda_choice="请选择CUDA版本 [1=CUDA 12.8, 2=CUDA 11.8]: "
echo.

echo.
echo [2/4] 卸载 CPU 版本的 PyTorch...
pip uninstall torch torchvision -y

echo.
echo [3/4] 安装 GPU 版本的 PyTorch...
if "%cuda_choice%"=="1" (
    echo 正在安装 PyTorch + CUDA 12.8 (RTX 50系列)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo 正在安装 PyTorch + CUDA 11.8 (RTX 40系列及以下)...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [4/4] 验证 GPU 支持...
cd ..
python check_gpu_status.py
cd gpu

echo.
echo ================================================
echo 完成！如果看到 CUDA Available: True
echo 说明 GPU 版本安装成功！
echo ================================================
pause
