# 详细安装指南

> 动态场景单摄像头全自动速度估计系统 - 安装指南

## 🎯 快速安装（推荐）

```bash
# Windows用户 - 双击运行
install.bat      # 命令提示符版本
install.ps1      # PowerShell版本

# 程序将自动完成所有配置
```

## 🔧 手动安装

### 环境要求

#### 当前阶段（第一、二阶段 ✅）
- **Python**: 3.7 - 3.11 (推荐 3.8/3.9)
- **操作系统**: Windows 10/11, Linux, macOS
- **内存**: 最低 4GB, 推荐 8GB+
- **存储**: 最低 2GB 可用空间
- **GPU**: 可选 (CPU 也可运行)

**已实现功能：**
- ✅ YOLOv8 物体检测
- ✅ ByteTrack 高精度追踪
- ✅ 简化版速度估算 (基于物体尺寸)

#### 未来阶段（第三阶段 🔄）
- **Python**: 3.8+
- **内存**: 8GB+ 推荐（RAFT、Depth Anything 模型）
- **存储**: 5GB+（包含大型深度学习模型）
- **GPU**: 强烈推荐（NVIDIA GPU + CUDA）

**计划功能：**
- ⏳ RAFT 光流运动分离
- ⏳ Metric Depth 深度估计

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

# 如果遇到网络问题，使用国内源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 核心依赖包：
pip install opencv-python>=4.5.0
pip install numpy>=1.21.0
pip install ultralytics>=8.0.0  # YOLOv8 + ByteTrack
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
- [ ] `python -c "from ultralytics import YOLO; print('OK')"` 成功
- [ ] `python main.py` 能够启动
- [ ] 程序显示"[OK] YOLOv8 model loaded"
- [ ] 程序显示"[OK] ByteTrack tracker ready"

## 🔮 第三阶段依赖（开发中）

第三阶段将引入光流运动分离和深度估计：

### 核心依赖
```bash
# PyTorch (RAFT, Depth Anything 的基础)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# RAFT 光流
pip install raft-optical-flow

# Depth Anything V2 (Metric 深度估计)
pip install depth-anything-v2
```

### 技术栈说明
| 模块 | 用途 | 状态 |
|------|------|------|
| **YOLOv8** | 物体检测 | ✅ 已完成 |
| **ByteTrack** | 高精度追踪 | ✅ 已完成 |
| **速度估算** | 物体尺寸标定 | ✅ 已完成 |
| **RAFT** | 光流运动分离 | 🔄 开发中 |
| **Depth Anything** | 深度估计 | 🔄 开发中 |

### GPU 加速配置
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 下一步

安装完成后：
1. 运行 `python main.py` 开始使用
2. 选择模式：
   - 模式 1: 检测 + 追踪 (ByteTrack)
   - 模式 2: 检测 + 追踪 + 速度估算 [推荐]
3. 查看 `README.md` 了解项目概述
4. 查看 `docs/ARCHITECTURE.md` 了解技术架构

### 项目进度
- ✅ **第一阶段**: YOLOv8检测 + SimpleTracker追踪
- ✅ **第二阶段**: ByteTrack + 简化版速度估算
- 🔄 **第三阶段**: RAFT光流 + Metric Depth（开发中）
- 📋 **第四阶段**: Web界面 + 系统增强（规划中）
