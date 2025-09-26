# 详细安装指南

## 🎯 快速安装（推荐）

```bash
# Windows用户 - 双击运行
install.bat      # 命令提示符版本
install.ps1      # PowerShell版本

# 程序将自动完成所有配置
```

## 🔧 手动安装

### 环境要求
- **Python**: 3.7+ (推荐3.9+)
- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 4GB+ 推荐 (原生版本需要更多内存)
- **存储**: 1GB可用空间 (包含两个版本的模型)

### 安装步骤

#### 1. Python环境准备
```bash
# 检查Python版本
python --version

# 如果版本过低，请从官网下载新版本
# https://www.python.org/downloads/
```

#### 2. 依赖安装
```bash
# 安装项目依赖
pip install -r requirements.txt

# 或者逐个安装核心依赖
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
```

#### 3. 环境验证
```bash
# 运行系统检测
python setup_and_test.py

# 运行主程序
python main.py
```

## 🛠️ 故障排除

### 常见问题

#### Python版本过低
```bash
# 错误: 需要Python 3.8或更高版本
# 解决: 安装新版本Python
# 下载地址: https://www.python.org/downloads/
```

#### 依赖安装失败
```bash
# 错误: pip install失败
# 解决: 更新pip并重试
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### OpenCV导入错误
```bash
# 错误: import cv2失败
# 解决: 重新安装opencv-python
pip uninstall opencv-python
pip install opencv-python
```

#### 内存不足
```bash
# 错误: 处理大视频时内存溢出
# 解决: 
# 1. 使用较小的视频文件
# 2. 关闭其他程序释放内存
# 3. 分段处理长视频
```

### 高级配置

#### 虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv speed_estimation_env

# 激活环境（Windows）
speed_estimation_env\Scripts\activate

# 激活环境（Linux/macOS）
source speed_estimation_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### GPU加速（可选）
```bash
# 如果有NVIDIA GPU，可安装CUDA版本
# 注意：当前项目使用CPU推理，GPU加速为未来扩展
pip install opencv-python[contrib]
```

## 📋 验证清单

安装完成后，请确认：
- [ ] `python --version` 显示3.7+
- [ ] `python -c "import cv2; print(cv2.__version__)"` 成功
- [ ] `python -c "import numpy; print(numpy.__version__)"` 成功
- [ ] `python main.py` 能够启动
- [ ] 程序显示"YOLOv8模型加载成功"或自动降级到备用检测

## 🚀 下一步

安装完成后：
1. 运行 `python main.py` 开始使用
2. 查看主目录的 `README.md` 了解详细功能
3. 查看 `docs/ARCHITECTURE.md` 了解技术架构
