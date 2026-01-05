@echo off
chcp 65001 >nul
echo ================================================
echo 视频中移动物体速度估计项目 - 一键安装
echo ================================================

echo 正在检查Python...
python --version
if errorlevel 1 (
    echo 错误：未找到Python，请先安装Python 3.7+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo 正在运行安装和测试工具...
python setup_and_test.py

if errorlevel 1 (
    echo.
    echo 自动安装失败，请查看错误信息
    pause
    exit /b 1
)

echo.
echo ================================================
echo 安装完成！现在您可以运行：
echo python main.py
echo ================================================
pause