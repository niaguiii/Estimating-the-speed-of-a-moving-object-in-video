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
echo [1/3] 卸载 CPU 版本的 PyTorch...
pip uninstall torch torchvision -y

echo.
echo [2/3] 安装 GPU 版本的 PyTorch (CUDA 11.8)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/3] 验证 GPU 支持...
cd ..
python check_gpu_status.py
cd gpu

echo.
echo ================================================
echo 完成！如果看到 CUDA Available: True
echo 说明 GPU 版本安装成功！
echo ================================================
pause
