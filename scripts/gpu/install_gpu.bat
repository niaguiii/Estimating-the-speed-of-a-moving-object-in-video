@echo off
chcp 65001 >nul
echo ================================================================
echo          Phase 3 GPU Environment Installation Script
echo          视频速度估计系统 - GPU环境完整安装
echo ================================================================
echo.
echo 此脚本将在全新环境中安装所有依赖（GPU版本）
echo 适用于：AutoDL、恒源云、阿里云等GPU服务器
echo.
echo 安装内容：
echo   1. PyTorch 2.x + CUDA (GPU版本 - 会提示选择版本)
echo      选项1: CUDA 12.8 适用于RTX 50系列 (5090, 5080等)
echo      选项2: CUDA 11.8 适用于RTX 40系列及以下
echo   2. Torchvision + RAFT支持
echo   3. Depth Anything V2依赖
echo   4. YOLOv8 + ByteTrack
echo   5. 所有Phase 1/2/3依赖
echo.
echo ================================================================
pause

echo.
echo ================================================================
echo [1/5] 检查Python版本...
echo ================================================================
python --version
if errorlevel 1 (
    echo ❌ 错误：未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)
echo ✅ Python版本检查通过

echo.
echo ================================================================
echo [2/5] 升级pip到最新版本...
echo ================================================================
python -m pip install --upgrade pip
echo ✅ pip升级完成

echo.
echo ================================================================
echo [3/5] 安装PyTorch (GPU版本)
echo ================================================================
echo 正在检测GPU型号...
nvidia-smi --query-gpu=name --format=csv,noheader 2>nul
echo.
echo 重要提示：
echo   - RTX 50系列 (5090, 5080等): 选择 1 (CUDA 12.8)
echo   - RTX 40系列及以下 (4090, 3090等): 选择 2 (CUDA 11.8)
echo.
set /p cuda_choice="请选择CUDA版本 [1=CUDA 12.8, 2=CUDA 11.8]: "
echo.

if "%cuda_choice%"=="1" (
    echo 正在安装 PyTorch + CUDA 12.8 适用于RTX 50系列...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
) else (
    echo 正在安装 PyTorch + CUDA 11.8 适用于RTX 40系列及以下...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
)

if errorlevel 1 (
    echo ❌ PyTorch安装失败，请检查网络连接
    pause
    exit /b 1
)
echo ✅ PyTorch GPU版本安装完成

echo.
echo ================================================================
echo [4/5] 安装Depth Anything V2依赖...
echo ================================================================
pip install timm>=0.9.0 transformers>=4.30.0 huggingface-hub>=0.19.0
if errorlevel 1 (
    echo ❌ Depth依赖安装失败
    pause
    exit /b 1
)
echo ✅ Depth Anything依赖安装完成

echo.
echo ================================================================
echo [5/5] 安装其他项目依赖...
echo ================================================================
pip install ultralytics>=8.0.0 opencv-python>=4.5.0 numpy>=1.21.0 supervision>=0.11.0 Pillow>=8.0.0 matplotlib>=3.5.0 pandas>=1.3.0 tqdm>=4.60.0
if errorlevel 1 (
    echo ❌ 项目依赖安装失败
    pause
    exit /b 1
)
echo ✅ 所有项目依赖安装完成

echo.
echo ================================================================
echo [验证] 检查GPU是否可用...
echo ================================================================
cd ..
python check_gpu_status.py
if errorlevel 1 (
    echo ⚠️  警告：check_gpu_status.py不存在，手动验证GPU
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
)
cd gpu

echo.
echo ================================================================
echo ✅ GPU环境安装完成！
echo ================================================================
echo.
echo 📋 已安装的核心组件：
if "%cuda_choice%"=="1" (
    echo   ✅ PyTorch 2.x + CUDA 12.8 (RTX 50系列)
) else (
    echo   ✅ PyTorch 2.x + CUDA 11.8 (RTX 40系列及以下)
)
echo   ✅ Torchvision (RAFT光流)
echo   ✅ Depth Anything V2
echo   ✅ YOLOv8 + ByteTrack
echo   ✅ 所有Phase 1/2/3依赖
echo.
echo 🚀 下一步：
echo   1. 确认GPU可用（上面应该显示 CUDA: True）
echo   2. 返回scripts目录: cd ..
echo   3. 运行测试: python test_dependencies.py
echo   4. 返回根目录开始处理: cd .. ^&^& python main.py
echo.
echo ================================================================
pause
