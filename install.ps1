# PowerShell安装脚本 - 简化版

Write-Host "================================================" -ForegroundColor Green
Write-Host "视频中移动物体速度估计项目 - 一键安装" -ForegroundColor Green  
Write-Host "================================================" -ForegroundColor Green

# 检查Python
Write-Host "正在检查Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python版本: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "错误：未找到Python，请先安装Python 3.7+" -ForegroundColor Red
    Write-Host "下载地址：https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "按任意键退出"
    exit 1
}

# 运行安装和测试工具
Write-Host "正在运行安装和测试工具..." -ForegroundColor Yellow
try {
    python setup_and_test.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "================================================" -ForegroundColor Green
        Write-Host "安装完成！现在您可以运行：" -ForegroundColor Green
        Write-Host "python main.py" -ForegroundColor Cyan
        Write-Host "================================================" -ForegroundColor Green
    } else {
        throw "安装失败"
    }
} catch {
    Write-Host "自动安装失败，请查看错误信息" -ForegroundColor Red
    Read-Host "按任意键退出"
    exit 1
}

Read-Host "按任意键退出"