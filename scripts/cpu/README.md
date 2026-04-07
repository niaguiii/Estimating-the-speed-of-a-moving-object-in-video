# CPU 环境安装脚本（本地开发）

## 📋 概述

此文件夹包含**本地开发环境**（CPU版本）的所有安装脚本。

**适用场景：**
- ✅ 本地电脑（集显/无独显）
- ✅ 代码开发和调试
- ✅ 测试单帧处理（Mode 1-4）
- ⚠️ Mode 5/6 需要 GPU（请使用 GPU 版本）

---

**重要提醒：Mode 5 和 Mode 6 需要 GPU 才能运行！**
- Mode 5（Metric3D v2 绝对深度测速）
- Mode 6（自车速度估算）

如需使用 Mode 5/6，请参考 `../gpu/README.md` 安装 GPU 环境。

---

## 📂 文件列表

| 文件 | 平台 | 用途 |
|------|------|------|
| **install.bat** | Windows | 一键安装CPU环境 |
| **install.ps1** | PowerShell | PowerShell安装脚本 |
| **setup_and_test.py** | 通用 | 环境检查和测试工具 |

---

## 🚀 快速开始

### Windows用户（推荐）

```bash
cd scripts\cpu
install.bat
```

### PowerShell用户

```bash
cd scripts\cpu
.\install.ps1
```

### 手动安装

```bash
cd scripts\cpu
pip install -r requirements.txt
python setup_and_test.py
```

---

## 📦 安装内容

### 核心框架
- PyTorch 2.6.0+cpu
- Torchvision 0.19.0+cpu

### Phase 3 依赖
- timm (Depth Anything)
- transformers
- huggingface-hub

### Phase 1 & 2 依赖
- ultralytics (YOLOv8 + ByteTrack)
- opencv-python
- numpy
- supervision
- matplotlib
- pandas
- tqdm

---

## 验证安装

安装完成后，验证环境：

```bash
# 返回scripts目录
cd ..

# 运行验证脚本
python ../test_project.py --mode deps
```

**期望输出：**
```
 PyTorch: 2.6.0+cpu
 CUDA Available: False (CPU mode for local dev)
 RAFT Large model available
 Depth Anything dependencies ready
 YOLOv8 + ByteTrack ready
✅ RAFT Large model available
✅ Depth Anything dependencies ready
✅ YOLOv8 + ByteTrack ready
```

---

## ⚙️ 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高
- **内存**: 8GB+ 推荐
- **硬盘**: 10GB+ 可用空间
- **GPU**: 不需要（CPU版本）

---

## 🔧 常见问题

### Q1: 安装时间多长？
**A:** 10-30分钟，取决于网络速度。

### Q2: 可以处理视频吗？
**A:** 可以，但很慢（~5秒/帧）。建议只测试单帧或短视频。

### Q3: 与GPU版本有什么区别？
**A:** 
- CPU版本：用于开发调试（免费，慢）
- GPU版本：用于处理视频（租卡，快50倍）

### Q4: 网络慢怎么办？
**A:** 使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📊 性能说明

| 任务 | CPU处理时间 |
|------|------------|
| 单帧RAFT光流 | ~3秒 |
| 单帧Depth估计 | ~2秒 |
| 完整处理 | ~5秒/帧 |
| 1分钟视频(1800帧) | ~2.5小时 |

**建议：** 
- ✅ 本地测试单帧
- ⚠️ 完整视频去租GPU

---

## 🎯 下一步

安装完成后：

1. **验证环境**: `python ../test_project.py --mode deps`
2. **开始开发**: 编写Phase 3核心模块
3. **测试功能**: 测试单帧处理
4. **准备租GPU**: 需要处理完整视频时

---

## 📚 相关文档

- **../gpu/README.md** - GPU环境安装
- **../README.md** - 项目主文档

---

## ✅ 安装状态

根据检查，你的CPU环境已经完全安装成功！

```
✅ PyTorch 2.6.0+cpu - 已安装
✅ Torchvision 0.19.0+cpu - 已安装
✅ RAFT支持 - 已就绪
✅ Depth Anything - 已就绪
✅ YOLOv8 + ByteTrack - 已就绪
✅ 所有其他依赖 - 已安装
```

**你可以开始Phase 3开发了！** 🎉
