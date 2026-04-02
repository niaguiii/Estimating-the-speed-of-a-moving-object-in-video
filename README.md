# 视频速度估计系统

> 基于深度学习的单摄像头全自动速度估计 — 支持测外部物体速度（Mode 5）和摄像头自身速度（Mode 6），覆盖固定/移动视角多种场景

**[本文档 README.md]** — 项目概述、快速开始、功能演示、模式选择指南

**[项目结构 docs/PROJECT_STRUCTURE.md]** — 详细目录结构、模块关系、配置说明

**[技术报告 docs/TECHNICAL_REPORT.md]** — 六种模式 + 预处理的算法原理、公式推导、论文出处

**[Web部署 web/README.md]** — 前端/后端部署、API文档、开发指南

---

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

# 📑 目录

- [一、项目概述](#一项目概述)
- [二、功能与模式](#二功能与模式)
- [三、安装与使用](#三安装与使用)
- [四、项目结构](#四项目结构)
- [五、FYP 相关](#五fyp-相关)

---

# 一、项目概述

## 🎯 项目简介

本项目开发一个**通用视频速度估计系统**，提供两种测速能力：

- **物体速度（Mode 5）** — YOLOv8 检测 + ByteTrack 追踪 + RAFT 光流补偿摄像头运动 + Metric3D 绝对深度，测外部物体相对于地面的真实速度
- **自身速度（Mode 6）** — 纯光流 + 深度，无需检测任何目标，直接测出携带设备的移动速度
- **双模式覆盖** — 固定摄像头、移动手持、车载、无人机等任意场景

## ✨ 特性亮点

- 🎯 **YOLOv8 检测** — 80+ 类物体检测（人、车、卡车、自行车等）
- 🎯 **ByteTrack 追踪** — 卡尔曼滤波 + 两阶段匹配，追踪精度 80-90%
- 🌊 **RAFT 光流** — 分离摄像头运动，支持移动摄像头（Mode 3/4/5）
- 📏 **绝对深度估计** — Metric3D v2 单目绝对深度，无需标定，精度 ±2-5%（Mode 5）
- 🚶 **自车测速** — 纯光流 + 深度，无需 YOLO，测设备自身移动速度（Mode 6）
- 📊 **双模式界面** — CLI 命令行 + Web 图形界面
- 🛡️ **视频质量增强** — 自动检测模糊/雾气/亮度，自适应预处理

## 🧱 技术栈

| 模块 | 技术选型 | 用途 |
|------|---------|------|
| 物体检测 | YOLOv8 (Ultralytics) | 检测 80+ 类物体 |
| 目标追踪 | ByteTrack | 卡尔曼滤波 + 两阶段匹配 |
| 光流估计 | RAFT (Princeton) | 分离摄像头运动 |
| 相对深度 | Depth Anything V2 (ByteDance) | Mode 4 深度感知 |
| 绝对深度 | Metric3D v2 (HKUST) | Mode 5/6 绝对度量深度 |
| 后端 API | FastAPI | 高性能异步 API |
| 前端界面 | Vue 3 + Vite | Web 图形界面 |

> 📖 详细技术原理 → [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## 📈 开发路线图

```
✅ 第一阶段 - 基础检测追踪 ✅ 已完成
├── [√] 🤖 YOLOv8检测系统
├── [√] 🎯 SimpleTracker基础追踪
├── [√] 📹 完整视频处理管道
├── [√] 🖥️ 用户友好操作界面
└── [√] 📚 完整文档和使用指南

✅ 第二阶段 - ByteTrack + 速度估算 ✅ 已完成
├── [√] 🎯 ByteTrack高精度追踪
│   ├── [√] 卡尔曼滤波运动预测
│   ├── [√] 两阶段检测匹配策略
│   ├── [√] 轨迹可视化显示
│   └── [√] 追踪精度 80-90%
├── [√] ⚡ 简化版速度估算（基于物体尺寸）
│   ├── [√] 自动像素/米标定（车、人、卡车等）
│   ├── [√] 实时速度显示（km/h）
│   ├── [√] EMA速度平滑算法
│   └── [√] 速度统计面板（最大/平均）
└── [√] 📈 统一入口整合（main.py）

✅ 第三阶段 - 光流+深度估计 ✅ 已完成
├── [√] 🌊 RAFT光流运动分离
│   ├── [√] 摄像头自身运动估计
│   ├── [√] 目标真实运动分离
│   └── [√] 支持移动摄像头场景
├── [√] 📏 Metric Depth深度估计
│   ├── [√] 相对深度（Depth Anything V2）- Mode 4
│   ├── [√] 绝对度量深度（Metric3D v2, ±2-5%）- Mode 5 🔥
│   └── [√] 像素到真实世界自动转换
├── [√] 🎯 深度感知速度估计
│   ├── [√] 基于深度的自动标定，零配置
│   ├── [√] 3D世界坐标速度计算（Mode 5）
│   └── [√] 集成RAFT+Depth+YOLOv8 全流水线
├── [√] 📋 数据导出系统 - CSV格式
│   ├── [√] 全部6种模式均支持CSV导出
│   ├── [√] Mode 5双CSV（逐帧 + 按目标汇总 + 首帧截图）
│   └── [√] Mode 6双CSV（逐帧 + 按秒汇总统计）
└── [~] 🧪 系统测试与验证 - 进行中

✅ 第四阶段 - Web应用与部署 ✅ 已完成
├── [√] 🌐 Web前端界面开发
│   ├── [√] Vue 3前端框架搭建
│   ├── [√] FastAPI后端API服务
│   ├── [√] 6种处理模式支持（Mode 1-6）
│   ├── [√] 文件上传与进度显示
│   ├── [√] 任务取消功能（真实进程终止）
│   └── [√] 界面美化与CSS样式优化
└── [√] 🌧️ 恶劣条件增强
    ├── [√] 视频质量自动检测（模糊/雾气/亮度，自适应）
    ├── [√] Wiener去模糊（运动模糊/失焦修复）
    ├── [√] DCP暗通道去雾（雾/霾干扰消除）
    ├── [√] CLAHE+Gamma提亮（低光照/过曝校正）
    ├── [√] 增强效果左右对比预览（滑块拖拽实时对比）
    ├── [√] 预处理阶段三态流程（质量检测→预览→模式选择）
    └── [√] CLI + Web双端预处理支持

✅ 第五阶段 - 自车测速扩展 ✅ 已完成
└── [√] 🚶 移动视角自车测速（Mode 6）
    ├── [√] 路面特征点光流采样（不依赖YOLO）
    ├── [√] Metric3D全图深度 + 路面采样点测速
    ├── [√] 双阶段EMA平滑 + 冷启动抑制 + 3σ异常抑制
    ├── [√] 自车速度输出，适配手持/行走/行车记录仪等移动视角
    └── [√] 双CSV（逐帧 + 按秒汇总含累计位移）
```

**最后更新时间：2026-04-02**

---

# 二、功能与模式

## 🎮 六种处理模式

| 模式 | 名称 | 适用场景 | 速度精度 | 需要GPU |
|------|------|---------|---------|--------|
| **1** | 检测+追踪 | 固定摄像头，仅需追踪 | — | ❌ |
| **2** | 速度估算 | 固定摄像头，需要速度 | 中等 | ❌ |
| **3** | RAFT光流 | 移动摄像头（车载/手持） | 中等 | ✅ |
| **4** | 相对深度 | 移动摄像头+深度信息 | 较高 | ✅ |
| **5** | 绝对深度 🔥 | 通用场景，测外部物体最高精度 | ±2-5% | ✅ |
| **6** | 自车测速 | 手持/行车记录仪，测设备自身速度 | 较高 | ✅ |

> 💡 详细技术说明（论文原理、算法细节）→ [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

### 模式选择建议

```
需要速度吗？
  ├─ 否 → 模式1（检测+追踪）
  └─ 是
       摄像头固定吗？
         ├─ 是 → 模式2（速度估算）
         └─ 否
              测外部目标？
                ├─ 否 → 模式6（自车测速）
                └─ 是
                     需要最高精度？
                       ├─ 否 → 模式3/4
                       └─ 是 → 模式5 🔥
```

**快速选择：**
- 🎯 **只需追踪** → 模式1
- 🚗 **固定摄像头测速** → 模式2
- 📱 **移动摄像头测外部物体** → 模式5 🔥
- 🚶 **行走/行车记录仪测设备自身** → 模式6

---

### 模式简要说明

**模式1 — 检测+追踪（ByteTrack）**

YOLOv8 物体检测 + ByteTrack 多目标追踪，追踪 ID 可视化，轨迹线绘制。

**模式2 — 速度估算**

模式1 全部功能 + 基于物体尺寸的速度估算，自动识别车辆类型并标定，实时显示 km/h。

**模式3 — RAFT光流**

模式2 全部功能 + RAFT 光流计算全图运动场，分离摄像头自身运动与目标运动。

**模式4 — 相对深度**

模式3 全部功能 + Depth Anything V2 相对深度估计，自动深度感知标定，速度精度 ±10-15%。

**模式5 — 绝对深度测速（推荐）🔥**

模式3 全部功能 + **Metric3D v2 绝对深度估计**（直接输出米数，无需标定）+ RAFT 光流补偿摄像头运动 + 3D 空间速度计算，精度 ±2-5%。支持移动摄像头场景。输出真实世界速度 (km/h)，双 CSV 导出。

**模式6 — 自车测速**

纯光流 + Metric3D 深度，无需 YOLO 检测目标。采样画面底部路面特征点，结合竖直光流分量与深度计算设备自身移动速度。适配手持/行走/行车记录仪等移动视角，输出有符号速度（前进/倒车）。

---

## 🏗️ 系统架构

```
┌────────────────────────────────────────┐
│            输入视频（单摄像头）           │
└────────────────┬───────────────────────┘
                 ▼
┌────────────────────────────────────────┐
│  检测层：YOLOv8（80+类物体）            │
└────────────────┬───────────────────────┘
                 ▼
┌────────────────────────────────────────┐
│  追踪层：ByteTrack（ID + 轨迹）          │
└────────────────┬───────────────────────┘
                 ▼
        ┌─────────────────┐
        │  选择处理模式    │
        └───────┬─────────┘
    ┌─────┬─────┼─────┬─────┐
  模式1 模式2 模式3 模式4  模式5/6
   检测 尺寸  RAFT 相对   绝对深度
   追踪 估算 光流  深度    ↓
                              光流分离
                            深度感知
                              ↓
                          3D速度计算
                              ↓
              ┌──────────────────────────┐
              │  输出：视频 + CSV + 截图   │
              └──────────────────────────┘
```

> 📖 详细架构设计 → [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

---

## 📊 性能表现

### 处理速度

| 视频规格 | 模式1-2 | 模式3-4 | 模式5-6 | 内存占用 |
|---------|---------|---------|---------|---------|
| 640×480 | 12-15 FPS | 5-10 FPS | 2-4 FPS | ~200MB |
| 1280×720 | 8-10 FPS | 3-6 FPS | 1-3 FPS | ~400MB |
| 1920×1080 | 5-6 FPS | 2-4 FPS | 1-2 FPS | ~600MB |

*注：模式1-2 为 CPU 性能；GPU 可提升 3-5 倍。模式5/6 需要 GPU。*

### 精度对比

| 场景 | 模式1 | 模式2 | 模式3 | 模式4 | 模式5 | 模式6 |
|------|-------|-------|-------|-------|-------|-------|
| 固定摄像头 | 追踪 | 中等 | 较高 | 较高 | 最高 | N/A |
| 移动摄像头 | 追踪 | ❌ | 较高 | 较高 | 最高 | ✅ 自身 |
| 自身测速 | N/A | ❌ | ❌ | ❌ | ❌ | ✅ |
| 需标定 | 无 | 自动 | 自动 | 全自动 | 全自动 | 全自动 |

---

## 三、安装与使用
### 环境安装
> **Web 端与 CLI 端共用以下环境部署方式。**
#### CPU 版本
**方式一：脚本一键安装（推荐）**
```bash
# Windows
scripts\cpu\install_cpu.bat
# Linux/macOS
bash scripts/cpu/install_cpu.sh
方式二：手动安装

pip install -r requirements.txt
GPU 版本
如需使用 GPU 加速，请先安装 CUDA 和 cuDNN。

方式一：脚本一键安装（推荐）

# Windows
scripts\gpu\install_gpu.bat
# Linux/macOS
bash scripts/gpu/install_gpu.sh
方式二：手动安装

pip install -r requirements.txt
# GPU 相关依赖已包含在 requirements.txt 中
切换到 GPU 版本（如已安装 CPU 版本）

# Windows
scripts\gpu\switch_to_gpu.bat
# Linux/macOS
bash scripts/gpu/switch_to_gpu.sh
使用介绍
CLI 端
环境安装完成后，直接运行：

python main.py --input <视频路径> --output <输出路径>
详细参数说明请参见 main.py 使用指南。

Web 端
环境安装完成后，按以下步骤启动：

1. 启动后端

cd web/backend
python app.py
2. 启动前端

cd web/frontend
npm install
npm run dev
3. 访问网页

打开浏览器访问 http://localhost:5173，使用账号登录使用。

配置说明
配置项	说明	默认值
model_name	使用的检测模型	yolov8n
confidence_threshold	置信度阈值	0.25
nms_threshold	NMS 阈值	0.45
pixel_to_meter_ratio	像素到米的转换比例	0.1
Web 端配置文件位于 web/frontend/.env.production，CLI 端通过命令行参数传入。

文档导航
项目结构说明
技术报告
GPU 加速说明
CLI 脚本说明
Web 端说明

> 💡 **文档协作关系**：README.md → 概览入口；PROJECT_STRUCTURE.md → 详细结构；TECHNICAL_REPORT.md → 技术深度；web/README.md → Web 部署。

---

# 四、项目结构

```
项目根目录/
├── 📄 main.py                  # CLI 主程序入口（统一调用全部 6 种模式）
│
├── 📂 data/                    # 运行时数据目录
│   ├── cli/
│   │   ├── input/             # CLI 模式输入视频
│   │   └── output/            # CLI 模式输出（视频、CSV、截图）
│   └── web/
│       ├── uploads/           # Web 上传视频（后端自动创建）
│       └── outputs/           # Web 处理结果（后端自动创建）
│
├── 📂 src/                    # 核心源代码（16 个 .py 文件）
│   ├── __init__.py            # 模块初始化，导出所有核心接口
│   ├── config.py              # 全局配置（阈值、路径、颜色等）
│   ├── model_config.py        # 模型下载路径管理（必须最先 import）
│   │
│   ├── mode1_detection_tracking.py   # Mode 1: YOLOv8 检测 + ByteTrack 追踪
│   ├── mode2_speed_estimation.py     # Mode 2: 物体尺寸标定 + EMA 速度估算
│   ├── mode3_raft_optical_flow.py    # Mode 3: RAFT 光流 + 摄像头运动补偿
│   ├── mode4_depth_anything_v2.py    # Mode 4: Depth Anything V2 相对深度感知
│   ├── mode5_metric3d_v2.py          # Mode 5: Metric3D v2 绝对度量深度（推荐）
│   ├── mode6_ego_speed.py            # Mode 6: 路面光流自车速度估算（无需 YOLO）
│   │
│   ├── optical_flow_raft.py           # RAFT 稠密光流封装（Mode 3-6 共享）
│   ├── depth_estimation.py            # Depth Anything V2 封装（Mode 4）
│   ├── depth_estimation_metric3d.py   # Metric3D v2 封装（Mode 5-6）
│   │
│   ├── enhance_video.py               # 视频预处理：去雾 / 去模糊 / 亮度增强
│   ├── quality_detector.py            # 质量检测：模糊 / 雾气 / 亮度
│   │
│   ├── main_opencv.py               # 遗留：OpenCV ONNX 检测（仅供参考）
│   └── main_yolov8_native.py        # 遗留：YOLOv8 原生检测（仅供参考）
│   # 注：ByteTrack 追踪功能已集成到 Ultralytics 中，通过 model.track() 使用
│
├── 📂 cfg/                    # 配置文件
│   └── bytetrack_stable.yaml  # ByteTrack 稳定版配置
│
├── 📂 models/                 # 模型文件
│   ├── coco.names            # COCO 80 类物体名称
│   ├── Ultralytics/          # Ultralytics 配置与缓存
│   └── README.md             # 模型说明（含自动下载指引）
│   # 以下模型在首次运行时由各框架自动下载，无需手动操作：
│   #   YOLOv8n.pt  →  Ultralytics 自动下载
│   #   raft-small.pth  →  PyTorch Hub 自动下载
│   #   Depth Anything V2  →  HuggingFace transformers 自动下载
│   #   Metric3D v2  →  PyTorch Hub 自动下载
│
├── 📂 scripts/                # 安装与测试脚本
│   ├── check_project.py      # 统一项目状态检查
│   ├── test_project.py       # 统一项目测试
│   ├── cpu/                  # CPU 环境安装
│   └── gpu/                  # GPU 环境安装
│
├── 📂 web/                   # Web 应用
│   ├── backend/              # FastAPI 后端
│   │   ├── app.py           # 主程序（上传 / 处理 / 轮询 / 下载）
│   │   ├── process_worker.py # 子进程 worker（可 kill）
│   │   └── requirements.txt
│   └── frontend/             # Vue 3 前端
│       ├── src/
│       │   ├── api/index.js
│       │   └── components/
│       │       ├── VideoUpload.vue   # 拖拽上传
│       │       ├── ModeSelector.vue  # 模式选择
│       │       ├── ProgressBar.vue   # 进度条
│       │       └── ResultDisplay.vue # 结果展示
│       └── package.json
│
└── 📂 docs/                  # 项目文档
    ├── PROJECT_STRUCTURE.md   # 详细目录结构与模块说明
    ├── TECHNICAL_REPORT.md    # 六种模式 + 预处理的算法原理与论文出处
    └── FYP_Progress_Report_2nd.md  # FYP 阶段进度报告
```

> 📖 完整目录结构与说明 → [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

---

# 五、FYP 相关

## 🎯 项目信息

### 基本信息
- **项目类型**: Final Year Project (FYP) - 计算机视觉/深度学习应用
- **核心定位**: 动态场景单摄像头全自动速度估计系统
- **技术栈**:
  - 检测: YOLOv8 (Ultralytics原生)
  - 追踪: ByteTrack (卡尔曼滤波+两阶段匹配)
  - 光流: RAFT (Princeton Vision Lab)
  - 相对深度: Depth Anything V2 (TikTok/ByteDance) - Mode 4
  - 绝对深度: Metric3D v2 (HKUST) - Mode 5 & 6 🔥
  - 后端: FastAPI
  - 前端: Vue 3 + Vite
- **开发状态**: 全部6种模式完成，Web界面支持模式1-6

### 项目规模
- **代码量**: ~12,000+ 行
- **模块数**: 16 个核心模块（src/ 下 .py 文件数）
- **支持模式**: 6种处理模式
- **文档**: 8个完整文档
- **开发周期**: 2023-2026

### 最新更新
- **日期**: 2026-04-02
- **版本**: Phase 1-5 全部完成，支持模式1-6

---

## 🚀 FYP 创新亮点

### 核心技术创新

| # | 创新点 | 传统方案的局限 | 本系统的解决方案 | 技术依据 |
|---|--------|---------------|---------------|----------|
| 1 | **移动摄像头运动分离** | 必须固定摄像头，否则无法测速 | 支持任意移动摄像头场景 | RAFT光流算法分离背景运动 |
| 2 | **全自动深度标定** | 需要手动输入相机参数或标定 | 零配置，AI自动估计深度 | Metric Depth单目深度估计 |
| 3 | **真实世界速度输出** | 只能输出像素速度 | 直接输出km/h、m/s真实速度 | 深度+光流联合计算 |

---

### 应用场景创新

| # | 场景 | 创新描述 | 实用价值 |
|---|------|---------|---------|
| 4 | 🚗 **车载测速** | 在行车途中检测并测量周围车辆速度 | 行车记录仪智能化、ADAS辅助驾驶 |
| 5 | 🚁 **大疆无人机联动** | 综合DJI飞行数据（高度、GPS）精确测速 | 交通监控、赛事分析、事故还原 |
| 6 | 🚶 **手持拍摄测速** | 手机/运动相机移动时仍能准确测速 | 运动分析、体能训练 |
| 7 | 🏃 **体育运动测速** | 篮球、足球等高速运动物体测速 | 训练辅助、比赛数据化 |

---

### 技术实现亮点

| # | 亮点 | 说明 |
|---|------|------|
| 8 | **单目测速方案** | 不需要双目相机或LiDAR，任意单摄像头即可 |
| 9 | **多目标同时测速** | ByteTrack支持同时追踪并测速100+目标 |
| 10 | **通用物体支持** | YOLOv8检测80+类物体，不限于车辆（行人、球类、动物等）|

---

### 🚁 大疆无人机联动（扩展功能）

本系统预留了无人机联动扩展接口，支持读取大疆视频的元数据实现更高精度测速：

```
模式A：纯视觉分析                    模式B：大疆SDK联动  [扩展方向]
    │                                    │
    ├─ 深度：Metric Depth估计               ├─ 深度：直接使用真实飞行高度
    ├─ 摄像头运动：RAFT光流估计              ├─ 摄像头运动：使用GPS地速数据
    └─ 精度：中上                           └─ 精度：高

可获取的大疆数据：
  ● 飞行高度 (altitude) — 直接作为深度，无须估计
  ● GPS地速 (ground_speed) — 摄像头运动的真实值
  ● 云台角度 (gimbal_pitch) — 计算投影关系
  ● IMU数据 — 姿态补偿

数据来源：
  ● SRT字幕文件 - 无人机视频自带，包含GPS、高度、速度
  ● DJI Mobile SDK - 实时获取飞行数据
  ● DJI FlightRecord - 分析历史飞行记录
```

---

## 📊 技术优势对比

### 与传统方案对比

| 维度 | 传统测速系统 | 本系统 |
|------|------------|--------|
| 摄像头要求 | 固定摄像头 | 🔥 移动/固定均支持 |
| 标定方式 | 手动标定 | 🔥 全自动AI标定 |
| 速度单位 | 像素/帧 | 🔥 真实速度(km/h) |
| 适用场景 | 单一场景 | 🔥 多场景通用 |
| 检测物体 | 仅车辆 | 🔥 80+类物体 |
| 外部数据 | 无 | 🔥 支持无人机数据联动 |
| 交互方式 | 命令行 | 🔥 CLI + Web双界面 |

### 与同类研究对比

| 特性 | 传统方法 | 部分研究 | 本系统 |
|------|---------|---------|--------|
| 移动摄像头支持 | ❌ | ⚠️ 部分 | 🔥 完整 |
| 自动标定 | ❌ | ⚠️ 部分 | 🔥 全自动 |
| 多物体追踪 | ⚠️ 基础 | ⚠️ 基础 | 🔥 ByteTrack |
| 深度感知 | ❌ | ⚠️ 部分 | 🔥 Metric Depth |
| Web界面 | ❌ | ❌ | 🔥 Vue3 + API |
| 实时进度 | ❌ | ❌ | 🔥 0.1%精度 |

---

## 🎬 应用场景

| 场景 | 描述 | 推荐模式 |
|------|------|---------|
| 🚗 **车载测速** | 行车记录仪测量周围车辆速度 | 模式5 🔥 |
| 🚁 **无人机监控** | 航拍视频分析地面目标速度 | 模式3/4 |
| 🏃 **运动分析** | 手持拍摄分析运动员速度 | 模式3/4 |
| 📹 **固定监控** | 监控摄像头场景 | 模式2 |
| 🚶 **自车测速** | 手持/行车记录仪测设备自身移动速度 | 模式6 |

---

## 🔮 扩展方向（Future Work）

**大疆无人机联动** — 读取大疆视频的 SRT 字幕文件获取飞行元数据（高度、GPS地速、云台角度），直接使用真实飞行高度代替深度估计，使用 GPS 地速辅助补偿摄像头运动，进一步提升航拍场景的速度测量精度。

**融合 IMU/GPS 数据** — 结合设备惯性测量单元和 GPS 数据，弥补纯视觉估计在快速运动或低纹理场景下的不足。

**扩展检测类别** — 接入更多专用检测模型（如体育运动物体检测），适配更多专业应用场景。

---

## 🙏 致谢

感谢以下开源项目和技术社区的支持：

- **[YOLOv8 / Ultralytics](https://github.com/ultralytics/ultralytics)** — 目标检测算法
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** — 多目标追踪算法
- **[RAFT](https://github.com/princeton-vl/RAFT)** — 光流估计网络 (Princeton Vision Lab)
- **[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)** — 单目相对深度估计 (ByteDance)
- **[Metric3D v2](https://github.com/YvanYin/Metric3D)** — 单目绝对深度估计 (HKUST)
- **[FastAPI](https://fastapi.tiangolo.com/)** — Python Web 框架
- **[Vue.js](https://vuejs.org/)** — 前端框架
- **[OpenCV](https://opencv.org/)** — 计算机视觉库
- **[PyTorch](https://pytorch.org/)** — 深度学习框架

---

**感谢使用本项目！如有问题或建议，欢迎提 Issue 或 PR！**
