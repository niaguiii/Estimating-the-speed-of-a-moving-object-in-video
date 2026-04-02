# 项目结构文档

**[本文档 docs/PROJECT_STRUCTURE.md]** — 详细目录结构、模块关系、配置说明

**[项目概述 README.md]** — 项目概述、快速开始、功能演示、模式选择指南

**[技术报告 docs/TECHNICAL_REPORT.md]** — 六种模式 + 预处理的算法原理、公式推导、论文出处

**[Web部署 web/README.md]** — 前端/后端部署、API文档、开发指南

> 本文档详细介绍项目的目录结构、核心模块、数据流、以及各模块的技术选型。

---

## 1. 整体目录结构

```
Estimating-the-speed-of-a-moving-object-in-video/
│
├── main.py                     # CLI 主程序入口，统一调用全部 6 种模式
│
├── README.md                   # 项目概述、快速开始、功能演示
│
├── .gitignore                 # Git 忽略规则（模型、日志、缓存等）
│
├── src/                       # 核心源代码模块（16 个 .py 文件，扁平结构）
│   ├── __init__.py
│   ├── config.py              # 全局配置（阈值、路径、颜色等）
│   ├── model_config.py        # 模型下载路径管理（必须在 DL 库之前 import）
│   │
│   ├── mode1_detection_tracking.py    # Mode 1: YOLOv8 检测 + ByteTrack 追踪
│   ├── mode2_speed_estimation.py      # Mode 2: 固定摄像头速度估算（物体尺寸标定）
│   ├── mode3_raft_optical_flow.py     # Mode 3: RAFT 光流 + 摄像头运动补偿
│   ├── mode4_depth_anything_v2.py     # Mode 4: Depth Anything V2 相对深度感知
│   ├── mode5_metric3d_v2.py           # Mode 5: Metric3D v2 绝对度量深度（推荐）
│   ├── mode6_ego_speed.py             # Mode 6: 路面光流自车速度估算（无需 YOLO）
│   │
│   ├── optical_flow_raft.py            # RAFT 稠密光流算法封装
│   ├── depth_estimation.py             # Depth Anything V2 深度估计封装
│   ├── depth_estimation_metric3d.py     # Metric3D v2 深度估计封装
│   │
│   ├── enhance_video.py                # 视频预处理增强
│   │                                    #   - DCP 暗通道去雾
│   │                                    #   - 维纳反卷积去模糊
│   │                                    #   - CLAHE + Gamma 亮度增强
│   │
│   ├── quality_detector.py              # 视频质量检测
│   │                                    #   - Laplacian 方差法检测模糊
│   │                                    #   - DCP 暗通道法检测雾气
│   │                                    #   - 直方图统计检测亮度
│   │
│   ├── main_opencv.py                  # 遗留：OpenCV ONNX 检测（仅供参考）
│   └── main_yolov8_native.py           # 遗留：YOLOv8 原生检测（仅供参考）
│
├── cfg/                         # 配置文件
│   └── bytetrack_stable.yaml    # ByteTrack 稳定版配置文件
│
├── scripts/                     # 辅助脚本
│   ├── README_SCRIPTS.md        # 脚本目录导航
│   │
│   ├── check_project.py         # 统一项目状态检查器（替代旧的独立检查脚本）
│   ├── test_project.py          # 统一项目测试器（GPU、依赖、RAFT、深度、Phase 3）
│   │
│   ├── cpu/                     # CPU 环境
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── install.bat
│   │   ├── install.ps1
│   │   └── setup_and_test.py
│   │
│   └── gpu/                     # GPU 环境
│       ├── requirements.txt
│       ├── README.md
│       ├── install_gpu.bat
│       ├── install_gpu.sh
│       ├── switch_to_gpu.bat
│       └── switch_to_gpu.sh
│
├── web/                        # Web 前端 + 后端
│   ├── frontend/                # Vue 3 + Vite 前端
│   │   ├── package.json
│   │   ├── vite.config.js
│   │   ├── index.html
│   │   ├── main.js
│   │   ├── App.vue
│   │   ├── style.css
│   │   ├── .env.development
│   │   ├── .env.production
│   │   └── src/
│   │       ├── api/index.js    # Axios API 封装（与后端通信）
│   │       └── components/
│   │           ├── VideoUpload.vue    # 拖拽上传视频
│   │           ├── ModeSelector.vue   # 模式选择（1-6）
│   │           ├── ProgressBar.vue    # 实时进度条
│   │           └── ResultDisplay.vue  # 结果展示与下载
│   │
│   └── backend/                 # FastAPI 后端
│       ├── app.py              # FastAPI 主程序（上传、处理、轮询、下载）
│       ├── process_worker.py    # 独立子进程处理 worker（可 kill）
│       └── requirements.txt
│
├── models/                     # 模型文件与配置
│   ├── coco.names             # COCO 80 类物体名称
│   ├── Ultralytics/           # Ultralytics（YOLOv8）配置与缓存
│   └── README.md              # 模型说明与自动下载指引
│
├── data/                      # 运行时数据目录
│   ├── cli/
│   │   ├── input/             # CLI 模式输入视频存放（需手动创建）
│   │   └── output/            # CLI 模式处理结果输出
│   └── web/
│       ├── uploads/           # Web 上传视频存放（由后端创建）
│       └── outputs/           # Web 处理结果存放（由后端创建）
│
└── docs/                     # 项目文档
    ├── PROJECT_STRUCTURE.md   # 本文件：详细目录结构
    ├── TECHNICAL_REPORT.md    # 技术原理报告：六种模式 + 预处理的算法与论文
    └── FYP_Progress_Report_2nd.md  # FYP 阶段进度报告
```

---

## 2. 核心模块详解

### 2.1 核心模块入口

| 模块 | 文件 | 说明 |
|------|------|------|
| **Mode 1** | `src/mode1_detection_tracking.py` | YOLOv8 检测 + ByteTrack 追踪，仅追踪不测速 |
| **Mode 2** | `src/mode2_speed_estimation.py` | 物体标准尺寸标定 + EMA 平滑，固定摄像头 |
| **Mode 3** | `src/mode3_raft_optical_flow.py` | RAFT 光流提取摄像头运动，已补偿速度 |
| **Mode 4** | `src/mode4_depth_anything_v2.py` | Depth Anything V2 相对深度 + 透视修正 |
| **Mode 5** | `src/mode5_metric3d_v2.py` | Metric3D v2 绝对度量深度 + 滑动窗口 3D 速度（**推荐**） |
| **Mode 6** | `src/mode6_ego_speed.py` | 路面光流 + Metric3D v2，自车速度，无需 YOLO |

### 2.2 底层算法模块

| 模块 | 文件 | 说明 |
|------|------|------|
| **RAFT 光流** | `src/optical_flow_raft.py` | RAFT 稠密光流算法封装，支持 GPU/CPU 自动切换 |
| **Depth Anything V2** | `src/depth_estimation.py` | 单目相对深度估计（输出 0-1 归一化值） |
| **Metric3D v2** | `src/depth_estimation_metric3d.py` | 单目绝对度量深度（输出真实米数） |

### 2.3 预处理模块

| 模块 | 文件 | 说明 |
|------|------|------|
| **质量检测** | `src/quality_detector.py` | 检测模糊 / 雾气 / 亮度三类问题 |
| **视频增强** | `src/enhance_video.py` | DCP 去雾 / 维纳去模糊 / CLAHE 亮度增强 |

---

## 3. Web 模块架构

### 3.1 前端（Vue 3 + Vite）

前端为单页应用（SPA），包含以下组件：

| 组件 | 文件 | 功能 |
|------|------|------|
| **App.vue** | `src/App.vue` | 根组件，布局容器 |
| **VideoUpload** | `src/components/VideoUpload.vue` | 拖拽上传视频，支持 FOV 参数输入 |
| **ModeSelector** | `src/components/ModeSelector.vue` | 模式选择（Mode 1-6），不同模式展示不同参数选项 |
| **ProgressBar** | `src/components/ProgressBar.vue` | 实时进度条，显示当前处理阶段 |
| **ResultDisplay** | `src/components/ResultDisplay.vue` | 结果展示与下载，包含视频预览和 CSV 数据展示 |

### 3.2 后端（FastAPI）

| 端点 | 说明 |
|------|------|
| `POST /upload` | 上传视频文件 |
| `POST /process` | 启动视频处理（异步，后端启动子进程） |
| `GET /status` | 轮询处理进度 |
| `GET /download/{filename}` | 下载处理结果（视频或 CSV） |
| `GET /logs/{task_id}` | 获取处理日志 |
| `DELETE /cancel/{task_id}` | 取消正在运行的处理任务 |

### 3.3 前后端通信

前端通过 Axios（`src/api/index.js`）向后端 API 发送请求，使用 JSON 交互。生产环境下前后端同源部署（Vite proxy 代理 `/api` 请求到 FastAPI 端口）。

---

## 4. 数据目录说明

| 目录 | 用途 | 创建时机 |
|------|------|---------|
| `data/cli/input/` | 存放 CLI 模式输入视频 | 用户手动创建或运行时自动创建 |
| `data/cli/output/` | 存放 CLI 模式处理结果（视频、CSV） | 运行时自动创建 |
| `data/web/uploads/` | 存放 Web 上传的视频文件 | FastAPI 启动时自动创建 |
| `data/web/outputs/` | 存放 Web 处理结果 | FastAPI 启动时自动创建 |

---

## 5. 模型文件说明

### 5.1 随代码附送（已提交到 Git）

| 文件 | 说明 |
|------|------|
| `models/coco.names` | COCO 80 类物体名称列表 |
| `cfg/bytetrack_stable.yaml` | ByteTrack 追踪器配置文件 |
| `models/Ultralytics/settings.json` | Ultralytics 全局设置 |
| `models/Ultralytics/persistent_cache.json` | Ultralytics 模型缓存元数据 |

### 5.2 自动下载（运行时，首次使用时下载）

| 模型 | 下载方式 | 存放位置 |
|------|---------|---------|
| YOLOv8n | Ultralytics 自动下载 | `models/` 或 `~/.ultralytics/` |
| RAFT-Small | PyTorch Hub 自动下载 | `~/.cache/torch/hub/` |
| Depth Anything V2 | HuggingFace `transformers` 自动下载 | `~/.cache/huggingface/` |
| Metric3D v2 | PyTorch Hub 自动下载 | `~/.cache/torch/hub/` |

> **注意**：运行 `main.py --mode 1` 时，如果模型文件不存在，系统会自动下载，无需手动操作。详细说明请参考 `models/README.md`。

---

## 6. 主要配置参数

### 6.1 全局配置（`src/config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CONFIDENCE_THRESHOLD` | 0.25 | YOLOv8 检测置信度阈值 |
| `IOU_THRESHOLD` | 0.45 | ByteTrack IoU 匹配阈值 |
| `MAX_AGE` | 30 | ByteTrack 最大未匹配帧数 |
| `N_INIT_FRAMES` | 3 | ByteTrack 初始化帧数 |
| `EMA_ALPHA_SPEED` | 0.3 | Mode 2/3/4 速度 EMA 平滑系数 |
| `WARMUP_FRAMES` | 30 | Mode 6 预热帧数 |

### 6.2 Mode 5/6 专用参数（`src/mode5_metric3d_v2.py` / `src/mode6_ego_speed.py`）

| 参数 | Mode 5 | Mode 6 | 说明 |
|------|--------|--------|------|
| 滑动窗口大小 | 7 帧 | N/A | Mode 5 速度计算基线 |
| 深度采样 | BBox 中值 | 路面区域随机 500 点 | 物体 vs 路面 |
| 深度频率 | 每 5 帧 | 每 5 帧 | 深度估计频率 |
| 平滑系数 | EMA α=0.4 | Warmup α=0.5 / Steady α=0.2 | 速度 EMA |
| 路面区域比例 | N/A | 底部 40% | Mode 6 采样区域 |
| 速度上限 | 无 | 200 km/h | Mode 6 硬上限 |

---

## 7. 输出文件说明

### 7.1 CLI 输出（Mode 1-6）

| 文件类型 | 说明 |
|---------|------|
| `*_annotated.mp4` | 带检测框、速度标签的标注视频 |
| `*_frames.csv` | 逐帧明细数据（Mode 5） |
| `*_objects.csv` | 按车辆汇总数据（Mode 5） |
| `*_stats.csv` | 按秒汇总统计（Mode 6） |
| `*_crops/` | 每辆车首帧检测截图（Mode 5） |

### 7.2 Web 输出

通过 `ResultDisplay.vue` 组件在线预览和下载，结果与 CLI 输出格式相同。

---

## 8. 依赖关系图

```
main.py
  ├── src/model_config.py          ← 必须在任何 DL 库之前导入
  ├── src/mode1_detection_tracking.py
  │     ├── src/optical_flow_raft.py         (Mode 3-6 共享)
  │     ├── src/depth_estimation.py          (Mode 4)
  │     ├── src/depth_estimation_metric3d.py  (Mode 5-6)
  │     └── (Ultralytics 内置 ByteTrack via model.track)
  ├── src/mode2_speed_estimation.py
  ├── src/mode3_raft_optical_flow.py
  ├── src/mode4_depth_anything_v2.py
  ├── src/mode5_metric3d_v2.py
  └── src/mode6_ego_speed.py

web/backend/app.py
  ├── src/mode5_metric3d_v2.py    (或任意其他 mode)
  └── src/process_worker.py

web/frontend/ (Vite dev server)
  └── src/api/index.js            → FastAPI backend
```

---

## 9. 文档导航

| 文档 | 内容 |
|------|------|
| `README.md` | 项目概述、快速开始、演示截图 |
| `PROJECT_STRUCTURE.md` | 本文档：详细目录结构、模块关系、配置说明 |
| `TECHNICAL_REPORT.md` | 六种处理模式的算法原理、公式推导、论文出处 |
| `docs/FYP_Progress_Report_2nd.md` | FYP 阶段进度报告 |
| `scripts/README_SCRIPTS.md` | 脚本目录导航 |
| `models/README.md` | 模型文件说明与自动下载指引 |
| `web/README.md` | Web 部署指南 |
| `scripts/cpu/README.md` | CPU 环境安装指南 |
| `scripts/gpu/README.md` | GPU 环境安装与部署指南 |
