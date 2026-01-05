# PowerShell Installation Script - Simplified Version

Write-Host "================================================" -ForegroundColor Green
Write-Host "Video Speed Estimation Project - One-Click Installation" -ForegroundColor Green  
Write-Host "================================================" -ForegroundColor Green

# Check Python
Write-Host "Checking Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python Version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.7+" -ForegroundColor Red
    Write-Host "Download: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press any key to exit"
    exit 1
}

# Run installation and test tool
Write-Host "Running installation and test tool..." -ForegroundColor Yellow
try {
    python setup_and_test.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "================================================" -ForegroundColor Green
        Write-Host "Installation Complete! You can now run:" -ForegroundColor Green
        Write-Host "python main.py" -ForegroundColor Cyan
        Write-Host "================================================" -ForegroundColor Green
    } else {
        throw "Installation failed"
    }
} catch {
    Write-Host "Installation failed. Please check error messages." -ForegroundColor Red
    Read-Host "Press any key to exit"
    exit 1
}

Read-Host "Press any key to exit"
