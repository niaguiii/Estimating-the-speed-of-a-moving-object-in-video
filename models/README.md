# 模型文件说明

## 📁 目录结构

```
models/
├── README.md                    # ✅ 本文件（已包含）
├── coco.names                   # ✅ COCO类别名称（已包含，1KB）
├── .gitkeep                     # ✅ 保持目录结构
├── yolov8n.pt                   # ✅ YOLOv8模型（已包含，6MB）
│
├── hub/                         # ⏳ RAFT模型缓存（首次运行自动下载）
│   └── checkpoints/
│       └── raft_small_*.pth     # RAFT模型 (~100MB)
└── models--depth-anything--*/   # ⏳ Depth模型缓存（首次运行自动下载）
    └── snapshots/
        └── */model.safetensors  # Depth Anything V2 (~400MB)
```

---

## 🤖 模型列表

### 1. YOLOv8 物体检测模型 ✅

| 属性 | 值 |
|------|-----|
| **文件** | `yolov8n.pt` |
| **大小** | 6MB |
| **用途** | 物体检测（汽车、行人等） |
| **来源** | Ultralytics |
| **状态** | ✅ 已包含在仓库中 |

**位置：**
- 已包含：`models/yolov8n.pt`
- 克隆仓库后即可直接使用

**使用场景：** 所有模式（Mode 1-4）

---

### 2. RAFT 光流模型 ⏳

| 属性 | 值 |
|------|-----|
| **文件** | `hub/checkpoints/raft_small_*.pth` |
| **大小** | ~100MB |
| **用途** | 光流估计，摄像头运动分离 |
| **来源** | PyTorch Torchvision |
| **状态** | ⏳ 首次运行Mode 3/4时自动下载 |

**下载位置：** 
- 自动下载到：`models/hub/checkpoints/`
- 由环境变量 `TORCH_HOME` 控制

**使用场景：** 
- Mode 3: RAFT Optical Flow（移动摄像头）
- Mode 4: Depth Perception（最高精度）

**下载时间：** 约1-2分钟（取决于网速）

---

### 3. Depth Anything V2 深度估计模型 ⏳

| 属性 | 值 |
|------|-----|
| **文件** | `models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/*/model.safetensors` |
| **大小** | ~400MB (Small) / ~1.3GB (Base) / ~1.6GB (Large) |
| **用途** | 单目深度估计，自动计算物体距离 |
| **来源** | Hugging Face (depth-anything) |
| **状态** | ⏳ 首次运行Mode 4时自动下载 |

**下载位置：** 
- 自动下载到：`models/models--depth-anything--*/`
- 由环境变量 `HF_HOME` 和 `TRANSFORMERS_CACHE` 控制

**使用场景：** 
- Mode 4: Depth Perception（最高精度）

**下载时间：** 约2-5分钟（取决于网速和模型大小）

---

## 🚀 自动下载机制

### 首次运行流程：

```bash
python main.py
# 选择 Mode 4

# 输出：
[1/4] Loading YOLOv8...
✅ YOLOv8 loaded from models/yolov8n.pt  # 已包含，直接加载

[2/4] Loading RAFT model...
[RAFT] Downloading to models/hub/checkpoints/...  # ⏳ 第一次：1-2分钟
Downloading: 100%|████████| 97.6M/97.6M
✅ RAFT loaded

[3/4] Loading Depth Anything V2...
[Depth] Downloading to models/models--depth-anything--*/...  # ⏳ 第一次：2-5分钟
Downloading: 100%|████████| 387M/387M
✅ Depth loaded

[4/4] Processing video...
```

### 后续运行：

```bash
python main.py
# 选择 Mode 4

# 输出：
[1/4] Loading YOLOv8...
✅ YOLOv8 loaded from models/yolov8n.pt (0.3s)

[2/4] Loading RAFT model...
✅ RAFT loaded from cache (0.5s)  # 从本地加载，很快！

[3/4] Loading Depth Anything V2...
✅ Depth loaded from cache (1.2s)  # 从本地加载，很快！

[4/4] Processing video...
```

---

## 📝 手动管理模型

### 备份模型（离线使用）

如果需要在没有网络的环境使用，可以：

1. **在有网络的环境运行一次**，自动下载模型到 `models/`
2. **打包整个 `models/` 文件夹**
3. **复制到离线环境的相同路径**

```bash
# 打包模型
tar -czf models_backup.tar.gz models/

# 在离线环境解压
tar -xzf models_backup.tar.gz
```

### 清理模型缓存

如果需要重新下载模型或清理空间：

```bash
# 删除RAFT模型
rm -rf models/hub/

# 删除Depth模型
rm -rf models/models--depth-anything--*/

# 删除所有大模型（保留YOLOv8）
rm -rf models/hub/ models/models--*/ models/*.pth models/*.safetensors
```

---

## 🔧 技术实现

### 代码中的配置：

#### RAFT模型（`src/optical_flow_raft.py`）

```python
import os

# ⚠️ 关键：必须在导入torch之前设置环境变量！
PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.environ['TORCH_HOME'] = MODEL_DIR
os.environ['TORCH_HUB'] = os.path.join(MODEL_DIR, 'hub')

# 现在才导入torch
import torch
```

#### Depth模型（`src/depth_estimation.py`）

```python
import os

# ⚠️ 关键：必须在导入transformers之前设置环境变量！
PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.environ['HF_HOME'] = MODEL_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR
os.environ['HF_HUB_CACHE'] = MODEL_DIR

# 现在才导入transformers
from transformers import AutoModelForDepthEstimation
```

### 测试配置：

在首次运行前，可以测试配置是否正确：

```bash
# 运行测试脚本
python scripts/test_model_paths.py

# 期望输出：
# ✅ optical_flow_raft 导入成功
# ✅ depth_estimation 导入成功
# ✅ 已正确设置到项目models目录
```

---

## ⚠️ 注意事项

### 1. Git版本控制

`.gitignore` 已配置排除大模型文件：
```
models/*.pth          # RAFT模型
models/*.safetensors  # Depth模型
models/hub/           # PyTorch缓存
models/models--*/     # Hugging Face缓存
```

**提交到GitHub的文件：**
- ✅ `yolov8n.pt` (6MB) - YOLOv8模型
- ✅ `coco.names` (1KB) - 类别名称
- ✅ `README.md` (本文件)

### 2. 磁盘空间

确保有足够空间：
- YOLOv8 nano: 6MB（已包含）
- RAFT small: ~100MB（首次下载）
- Depth Anything V2 small: ~400MB（首次下载）
- **总计：~500MB**

可选更大模型：
- YOLOv8 medium: 52MB
- RAFT large: ~250MB
- Depth V2 large: ~1.6GB

### 3. 网络连接

首次运行需要联网下载模型。后续运行可离线使用。

---

## 📊 模型对比

| 模型 | 大小 | 下载时间 | Git提交 | 用途 |
|------|------|---------|---------|------|
| **YOLOv8** | 6MB | - | ✅ 是 | 物体检测 |
| **RAFT** | ~100MB | 1-2分钟 | ❌ 否 | 光流估计 |
| **Depth V2** | ~400MB | 2-5分钟 | ❌ 否 | 深度估计 |
| **配置文件** | ~1KB | - | ✅ 是 | coco.names等 |

---

## 🎯 常见问题

### Q1: 为什么不把所有模型都上传到GitHub？

**A:** 统一采用自动下载策略：
- GitHub单文件限制100MB（Depth模型400MB超限）
- 大文件会让克隆仓库变慢
- GitHub可能删除/屏蔽大文件
- 自动下载更灵活（可选择模型大小）
- 所有模型统一管理，一致性更好

### Q2: 模型下载失败怎么办？

**A:** 检查网络连接：
```bash
# 测试PyTorch下载
ping download.pytorch.org

# 测试Hugging Face下载
ping huggingface.co

# 如果在中国，可以使用镜像：
export HF_ENDPOINT=https://hf-mirror.com
```

### Q3: 可以使用更大的模型吗？

**A:** 可以！修改代码：

```python
# src/optical_flow_raft.py
raft = RAFTOpticalFlow(model_type='large')  # 使用大模型

# src/depth_estimation.py
depth = DepthAnythingV2(model_size='base')  # 或 'large'
```

### Q4: 如何查看模型是否下载成功？

**A:** 检查文件：
```bash
# 查看models目录
ls -lh models/

# 查看RAFT模型
ls -lh models/hub/checkpoints/

# 查看Depth模型
ls -lh models/models--depth-anything--*/snapshots/
```

---

## 📚 相关文档

- **项目主文档**: `../README.md`
- **GPU环境安装**: `../scripts/gpu/README.md`
- **项目结构**: `../docs/PROJECT_STRUCTURE.md`

---

## ✅ 总结

所有模型统一管理在 `models/` 目录：
- ✅ **结构清晰** - 一个地方管理所有模型
- ✅ **自动下载** - 首次运行自动配置
- ✅ **离线友好** - 可以打包备份
- ✅ **Git优化** - 大文件不提交

**首次运行需要5-10分钟下载模型，之后秒开！** 🚀
