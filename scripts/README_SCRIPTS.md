# Scripts 文件夹说明

## 📂 文件夹结构

```
scripts/
├── cpu/                    # 🏠 CPU环境脚本（本地开发）
│   ├── README.md          # CPU安装说明
│   ├── requirements.txt   # CPU依赖列表
│   ├── install.bat        # Windows安装
│   ├── install.ps1        # PowerShell安装
│   └── setup_and_test.py  # 环境检查工具
│
├── gpu/                    # ☁️ GPU环境脚本（云服务器）
│   ├── README.md          # GPU部署说明
│   ├── requirements.txt   # GPU依赖列表
│   ├── install_gpu.bat    # Windows完整安装 ⭐
│   ├── install_gpu.sh     # Linux完整安装 ⭐
│   ├── switch_to_gpu.bat  # Windows快速切换
│   └── switch_to_gpu.sh   # Linux快速切换
│
├── check_project.py        # ✅ 统一检查工具（整合所有检查）
├── test_project.py         # ✅ 统一测试工具（整合所有测试）
│
└── README_SCRIPTS.md       # 本文件（总览）
```

---

## 🏠 CPU环境（本地开发）

**位置:** `scripts/cpu/`

**适用场景:**
- ✅ 本地电脑开发
- ✅ 代码调试
- ✅ 测试单帧
- ⚠️ 不适合完整视频

**快速开始:**
```bash
cd scripts\cpu
install.bat
```

📖 **详细说明:** 查看 `cpu/README.md`

---

## ☁️ GPU环境（云服务器）

**位置:** `scripts/gpu/`

**适用场景:**
- ✅ GPU服务器部署
- ✅ 处理完整视频
- ✅ 性能测试
- ✅ 答辩演示

**快速开始:**
```bash
cd scripts\gpu
install_gpu.bat    # Windows
./install_gpu.sh   # Linux
```

📖 **详细说明:** 查看 `gpu/README.md`

---

## 🎯 使用场景

### 场景1: 首次在本地安装（当前）✅
```bash
cd scripts\cpu
install.bat
```
**状态:** ✅ 已完成安装！所有CPU环境依赖已就绪。

### 场景2: 租GPU后全新安装（推荐）⭐
```bash
cd scripts\gpu

# Windows服务器
install_gpu.bat

# Linux服务器
chmod +x install_gpu.sh
./install_gpu.sh
```
**安装:** PyTorch GPU + CUDA + 所有依赖（15-30分钟）

### 场景3: 已有环境，快速切换到GPU
```bash
cd scripts\gpu

# Windows
switch_to_gpu.bat

# Linux
chmod +x switch_to_gpu.sh
./switch_to_gpu.sh
```
**切换:** 仅重装 torch 和 torchvision（5-10分钟）

---

## ✅ 当前项目状态

### CPU环境（本地）- 已完成 ✅

根据检查，你的CPU环境已完全就绪：

```
✅ PyTorch 2.9.1+cpu - 已安装
✅ Torchvision 0.24.1+cpu - 已安装
✅ RAFT支持 - 可用
✅ Depth Anything V2 - 可用
✅ YOLOv8 + ByteTrack - 可用
✅ 所有Phase 1/2/3依赖 - 已安装
```

**你可以开始Phase 3开发了！**

### GPU环境 - 待部署

租用GPU服务器后，运行 `scripts/gpu/install_gpu.bat` 即可。

---

## 🔍 如何选择脚本？

### 决策流程图

```
你现在在哪里？
│
├─ 本地电脑（集显/无独显）
│  └─ 已完成！✅
│     CPU环境已安装，可以开始开发
│
└─ GPU服务器（已租用）
   │
   ├─ 是全新环境吗？
   │  │
   │  ├─ 是 → cd scripts/gpu && install_gpu.bat ✅
   │  │        (完整安装，推荐)
   │  │
   │  └─ 否，已有CPU版本
   │             └─ cd scripts/gpu && switch_to_gpu.bat
   │                (快速切换)
   │
   └─ 完成！开始处理视频
```

---

## 💡 最佳实践

### 本地开发阶段 ✅ 已完成
1. ✅ CPU环境已安装
2. ✅ 可以开始Phase 3开发
3. ✅ 可以测试单帧处理

### GPU部署阶段（需要时）
1. 租用GPU服务器
2. 上传项目到服务器
3. 运行 `cd scripts/gpu && install_gpu.bat`
4. 验证GPU: `python check_gpu_status.py`
5. 处理完整视频: `python main.py`

---

## 🆘 问题排查

### 脚本运行失败

**检查1: Python版本**
```bash
python --version
# 需要 3.8 或更高
```

**检查2: 网络连接**
```bash
ping pypi.org
# 或使用国内镜像
```

**检查3: 权限问题（Linux）**
```bash
chmod +x install_gpu.sh
chmod +x switch_to_gpu.sh
```

**检查4: 磁盘空间**
```bash
df -h  # Linux
dir    # Windows
# 需要至少 10GB 可用空间
```

---

## 📊 脚本对比

| 特性 | install.bat | install_gpu.bat | switch_to_gpu.bat |
|------|-------------|-----------------|-------------------|
| **目标环境** | 本地CPU | GPU服务器 | GPU服务器 |
| **PyTorch** | CPU版本 | GPU版本 | GPU版本 |
| **是否需要预装** | 否 | 否 ✅ | 是 |
| **安装内容** | 全部 | 全部 ✅ | 仅torch |
| **安装时间** | 10-20分钟 | 15-30分钟 | 5-10分钟 |
| **推荐场景** | 本地开发 | GPU全新安装 ✅ | GPU快速切换 |

---

## 🎯 快速命令参考

### 本地开发（已完成✅）
```bash
cd scripts\cpu
install.bat
```

### GPU全新安装（推荐）⭐
```bash
cd scripts\gpu
install_gpu.bat      # Windows
./install_gpu.sh     # Linux
```

### GPU快速切换
```bash
cd scripts\gpu
switch_to_gpu.bat    # Windows
./switch_to_gpu.sh   # Linux
```

### 验证安装
```bash
# 测试依赖
python scripts/test_project.py --mode deps

# 测试GPU（如果有）
python scripts/test_project.py --mode gpu
```

---

## 🛠️ 统一工具脚本

### check_project.py - 项目状态检查工具
**功能:** 统一的项目检查工具，替代原有的多个独立检查脚本

**使用:**
```bash
# 快速检查Phase 3
python scripts/check_project.py

# 完整检查（包括导入测试）
python scripts/check_project.py --mode full

# 检查所有内容
python scripts/check_project.py --mode all
```

**支持的检查模式:**
- `basic` - 基本结构检查
- `phase1-2` - Phase 1 & 2 检查
- `phase3` - Phase 3 快速检查（默认）
- `full` - 完整检查（包括导入测试）
- `all` - 所有检查

**检查内容:**
- ✅ 项目目录结构
- ✅ 核心模块文件
- ✅ Phase 3 模块（RAFT + Depth）
- ✅ 预训练模型
- ✅ main.py集成
- ✅ 模块导入测试

---

### test_project.py - 项目测试工具
**功能:** 统一的测试工具，整合GPU、依赖、RAFT、Depth等测试

**使用:**
```bash
# 测试依赖包
python scripts/test_project.py --mode deps

# 测试GPU状态
python scripts/test_project.py --mode gpu

# 测试Phase 3完整功能
python scripts/test_project.py --mode phase3

# 运行所有测试
python scripts/test_project.py --mode all
```

**支持的测试模式:**
- `gpu` - GPU状态测试
- `deps` - 依赖包测试（默认）
- `yolo` - YOLOv8测试
- `raft` - RAFT光流测试
- `depth` - Depth Anything V2测试
- `phase3` - Phase 3完整测试
- `all` - 所有测试

**测试内容:**
- ✅ GPU可用性和CUDA版本
- ✅ 所有依赖包安装状态
- ✅ YOLOv8检测功能
- ✅ RAFT光流计算
- ✅ Depth Anything V2深度估计

---

## 📚 相关文档

- **cpu/README.md** - CPU环境详细说明
- **gpu/README.md** - GPU环境详细说明
- **../docs/INSTALLATION_GUIDE.md** - 完整安装指南
- **../README.md** - 项目主文档

---

## ✅ 总结

### 当前状态
✅ **Phase 3已完成！** RAFT光流 + Depth Anything V2

### 快速验证
```bash
# 检查项目状态
python scripts/check_project.py

# 测试依赖和功能
python scripts/test_project.py --mode all

# 运行主程序（选择模式4）
python main.py
```

### 租GPU时
1. ✅ 上传项目到GPU服务器
2. ✅ 运行 `cd scripts/gpu && install_gpu.sh`
3. ✅ 等待15-30分钟
4. ✅ 验证：`python scripts/test_project.py --mode gpu`
4. ✅ 验证GPU可用
5. ✅ 开始处理视频！

**就这么简单！** 🎉
