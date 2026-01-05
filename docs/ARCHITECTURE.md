# 技术架构文档

> 动态场景单摄像头全自动速度估计系统 - 完整技术架构设计

**最后更新：** 2026-01-05  
**当前版本：** Phase 3 核心完成，Phase 4 Web开发中

---

## 📑 目录

- [系统架构概览](#系统架构概览)
- [四种处理模式架构](#四种处理模式架构)
- [核心技术模块](#核心技术模块)
- [Web应用架构](#web应用架构)
- [数据流程详解](#数据流程详解)
- [技术决策](#技术决策)
- [性能优化](#性能优化)

---

## 🏗️ 系统架构概览

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          用户交互层                                   │
│  ┌──────────────────┐              ┌──────────────────┐             │
│  │   CLI命令行界面   │              │   Web网页界面     │             │
│  │   main.py        │              │   Vue 3 Frontend │             │
│  └──────────────────┘              └──────────────────┘             │
└─────────────────────────────┬───────────────┬───────────────────────┘
                              │               │
                        ┌─────┴─────┐   ┌────┴────┐
                        │  本地处理  │   │  FastAPI │
                        │           │   │  Backend │
                        └─────┬─────┘   └────┬────┘
                              │               │
┌─────────────────────────────┴───────────────┴───────────────────────┐
│                          处理引擎层                                   │
│                                                                      │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ 模式1  │  │  模式2   │  │  模式3   │  │  模式4   │             │
│  │检测追踪│  │ 速度估算 │  │ RAFT光流 │  │ 深度感知 │             │
│  └────────┘  └──────────┘  └──────────┘  └──────────┘             │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                          AI算法层                                     │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  YOLOv8    │  │ ByteTrack  │  │    RAFT    │  │  Depth V2  │   │
│  │  检测引擎  │  │ 追踪引擎   │  │  光流引擎  │  │  深度引擎  │   │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
│                                                                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────┐
│                          框架层                                       │
│  PyTorch │ OpenCV │ NumPy │ FastAPI │ Vue 3 │ Vite                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎮 四种处理模式架构

### 模式1: 检测 + 追踪

```
输入视频
   │
   ↓
┌─────────────┐
│  帧提取     │ → OpenCV VideoCapture
└──────┬──────┘
       ↓
┌─────────────┐
│ YOLOv8检测  │ → 80+类物体检测
│  • 边界框   │
│  • 类别     │
│  • 置信度   │
└──────┬──────┘
       ↓
┌─────────────┐
│ ByteTrack  │ → 多目标追踪
│  • 卡尔曼滤波│
│  • 两阶段匹配│
│  • 唯一ID   │
└──────┬──────┘
       ↓
┌─────────────┐
│  轨迹可视化 │ → 渲染
│  • ID标签   │
│  • 轨迹线   │
│  • 检测框   │
└──────┬──────┘
       ↓
   输出视频
```

**关键技术：**
- YOLOv8: Ultralytics官方实现
- ByteTrack: 卡尔曼滤波 + 两阶段关联
- 追踪精度: 80-90%

---

### 模式2: 速度估算

```
输入视频
   │
   ↓
[模式1的所有步骤]
   │
   ↓
┌──────────────┐
│ 物体尺寸标定 │
│  • 自动识别类别│
│  • 标准尺寸库 │
│    - 车: 4.5m│
│    - 人: 1.7m│
│    - 卡车:12m│
└──────┬───────┘
       ↓
┌──────────────┐
│  速度计算    │
│ 像素位移 → 米│
│ 米/秒 → km/h │
└──────┬───────┘
       ↓
┌──────────────┐
│  EMA平滑     │ → 速度稳定显示
│  α = 0.3     │
└──────┬───────┘
       ↓
┌──────────────┐
│ 速度渲染     │
│  • 速度标签  │
│  • 统计面板  │
│  • 最大/平均 │
└──────┬───────┘
       ↓
   输出视频
```

**关键技术：**
- 基于物体尺寸的自动标定
- EMA指数移动平均平滑
- 假设摄像头固定

**局限：**
- 摄像头移动时精度下降

---

### 模式3: RAFT光流（核心创新）

```
输入视频
   │
   ├────────────┐
   │            │
   ↓            ↓
[模式2流程]  ┌────────────┐
   │         │ RAFT光流   │
   │         │  • 全图光流场│
   │         │  • 像素级运动│
   │         └──────┬─────┘
   │                ↓
   │         ┌────────────┐
   │         │ 背景运动估计│
   │         │  • 背景点检测│
   │         │  • 运动均值  │
   │         │  • 摄像头速度│
   │         └──────┬─────┘
   │                │
   ↓                ↓
┌─────────────────────┐
│   运动分离算法      │
│ 目标真实运动 =      │
│ 观测运动 - 摄像头运动│
└──────┬──────────────┘
       ↓
┌──────────────┐
│ 补偿后速度   │ → 真实速度输出
└──────┬───────┘
       ↓
   输出视频
```

**关键技术：**
- RAFT: Princeton Vision Lab光流网络
- 运动分离算法: 背景均值法
- 支持移动摄像头

**技术原理：**
```python
# 伪代码
optical_flow = RAFT(frame_t, frame_t+1)
background_flow = mean(optical_flow[background_pixels])
camera_motion = background_flow

for obj in tracked_objects:
    observed_motion = obj.displacement
    real_motion = observed_motion - camera_motion
    real_speed = calculate_speed(real_motion)
```

---

### 模式4: 深度感知速度（最高精度）

```
输入视频
   │
   ├────────────┬────────────┐
   │            │            │
   ↓            ↓            ↓
[模式3流程]  ┌────────┐  ┌────────┐
   │         │ RAFT   │  │Depth V2│
   │         │ 光流   │  │深度估计│
   │         └────┬───┘  └───┬────┘
   │              │          │
   │              ↓          ↓
   │         ┌──────────────────┐
   │         │  深度感知标定     │
   │         │ • 每像素深度值    │
   │         │ • 自动像素/米比例 │
   │         │ • 无需人工标定    │
   │         └────────┬─────────┘
   │                  │
   ↓                  ↓
┌───────────────────────────┐
│     深度修正速度计算      │
│  • 基于真实深度           │
│  • RAFT运动补偿          │
│  • 最高精度输出           │
└────────┬──────────────────┘
         ↓
    输出视频 + 深度图
```

**关键技术：**
- Depth Anything V2 Metric: TikTok/ByteDance
- 单目深度估计 → 绝对深度值（米）
- 全自动标定，零人工参与

**技术原理：**
```python
# 伪代码
depth_map = DepthAnythingV2(frame)
depth_at_obj = depth_map[obj.center]
pixel_to_meter_ratio = calculate_ratio(depth_at_obj, focal_length)

for obj in tracked_objects:
    pixel_displacement = obj.displacement
    meter_displacement = pixel_displacement * pixel_to_meter_ratio
    
    # RAFT补偿
    camera_motion = RAFT_background_motion()
    real_displacement = meter_displacement - camera_motion
    
    speed = real_displacement / time_interval
```

---

## 🔧 核心技术模块

### 1. YOLOv8检测引擎

**实现文件：** `src/main_yolov8_native.py`

```python
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5
        
    def detect(self, frame):
        results = self.model(frame, conf=self.confidence_threshold)
        return results[0].boxes
```

**特点：**
- 80+类COCO物体检测
- 支持车、人、卡车、自行车、球等
- FP16半精度加速（GPU）

---

### 2. ByteTrack追踪引擎

**实现文件：** `trackers/byte_tracker.py`

```python
class ByteTrack:
    def __init__(self, max_distance=100, max_disappeared=10):
        self.kalman_filter = KalmanFilter()
        self.tracks = {}
        
    def update(self, detections):
        # 第一阶段：高置信度匹配
        high_conf_matches = self._match(detections_high, tracks)
        
        # 第二阶段：低置信度匹配
        low_conf_matches = self._match(detections_low, unmatched_tracks)
        
        # 卡尔曼滤波预测
        for track in tracks:
            track.predict()
            
        return updated_tracks
```

**关键算法：**
- 卡尔曼滤波：运动预测
- 两阶段匹配：高/低置信度分开处理
- IoU匹配：边界框重叠度

---

### 3. RAFT光流引擎

**实现文件：** `src/optical_flow_raft.py`

```python
import torch
from raft import RAFT

class RAFTOpticalFlow:
    def __init__(self, model_path="models/raft-things.pth"):
        self.model = RAFT()
        self.model.load_state_dict(torch.load(model_path))
        
    def compute_flow(self, frame1, frame2):
        # 前向推理
        flow = self.model(frame1, frame2)
        return flow  # [H, W, 2] - (dx, dy)
    
    def estimate_camera_motion(self, flow, mask):
        # 背景点的光流均值
        background_flow = flow[mask == 0]
        camera_motion = torch.mean(background_flow, dim=0)
        return camera_motion
```

**技术细节：**
- 输入：连续两帧
- 输出：每个像素的运动向量 (dx, dy)
- 背景检测：排除检测到的物体区域
- 摄像头运动：背景光流的均值

---

### 4. Depth深度引擎

**实现文件：** `src/depth_estimation.py`

```python
from depth_anything_v2 import DepthAnythingV2

class DepthEstimator:
    def __init__(self, model="depth_anything_v2_vitl"):
        self.model = DepthAnythingV2.from_pretrained(model)
        
    def estimate_depth(self, frame):
        # 单目深度估计
        depth_map = self.model(frame)  # [H, W]
        return depth_map  # 单位：米
    
    def get_object_depth(self, depth_map, bbox):
        # 获取物体的深度
        x1, y1, x2, y2 = bbox
        obj_depth = torch.median(depth_map[y1:y2, x1:x2])
        return obj_depth.item()
```

**技术细节：**
- Depth Anything V2 Metric版本
- 输出绝对深度值（米）
- 不需要相机内参
- 支持任意场景

---

### 5. 速度计算引擎

**实现文件：** `src/main_yolov8_speed.py`

```python
class SpeedCalculator:
    def __init__(self, fps=30):
        self.fps = fps
        self.time_interval = 1.0 / fps
        
    def calculate_speed(self, pixel_displacement, pixel_to_meter, 
                        camera_motion=None):
        # 米位移
        meter_displacement = pixel_displacement * pixel_to_meter
        
        # RAFT补偿（如果有）
        if camera_motion is not None:
            meter_displacement -= camera_motion
        
        # 速度计算
        speed_ms = meter_displacement / self.time_interval
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def ema_smooth(self, current_speed, prev_speed, alpha=0.3):
        return alpha * current_speed + (1 - alpha) * prev_speed
```

---

## 🌐 Web应用架构

### 前后端分离架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户浏览器                            │
│              http://localhost:3000                      │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP请求
                     ↓
┌─────────────────────────────────────────────────────────┐
│                  Vue 3 前端 (端口3000)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │VideoUpload│  │ModeSelect│  │ Progress │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│                                                         │
│  Vite开发服务器 + Axios HTTP客户端                       │
└────────────────────┬────────────────────────────────────┘
                     │ RESTful API
                     │ WebSocket (实时进度)
                     ↓
┌─────────────────────────────────────────────────────────┐
│                FastAPI 后端 (端口8000)                   │
│                                                         │
│  ┌─────────────────────────────────────┐              │
│  │          app.py (主服务器)           │              │
│  │  • 文件上传接口                      │              │
│  │  • 模式选择接口                      │              │
│  │  • 进度查询接口                      │              │
│  │  • 取消任务接口                      │              │
│  └────────────┬────────────────────────┘              │
│               │                                        │
│               ↓                                        │
│  ┌─────────────────────────────────────┐              │
│  │    process_worker.py (处理进程)      │              │
│  │  • 独立进程运行                      │              │
│  │  • 实时进度更新                      │              │
│  │  • 可强制终止                        │              │
│  └─────────────────────────────────────┘              │
│                                                         │
└────────────────────┬────────────────────────────────────┘
                     │ 调用
                     ↓
┌─────────────────────────────────────────────────────────┐
│              核心处理模块 (src/)                         │
│  main_yolov8_bytetrack.py | main_yolov8_speed.py       │
│  main_yolov8_raft.py | main_phase3_complete.py         │
└─────────────────────────────────────────────────────────┘
```

### 数据流向

```
用户上传视频
      │
      ↓
  data/web/uploads/
      │
      ↓
FastAPI接收 → 创建worker进程 → 处理视频
      │                            │
      ↓                            ↓
  实时进度更新              data/web/outputs/
      │                            │
      ↓                            ↓
前端轮询进度  ←──────────── 处理完成
      │
      ↓
  用户下载结果
```

### API接口设计

```python
# 后端API (web/backend/app.py)

@app.post("/api/upload")
async def upload_video(file: UploadFile):
    """上传视频文件"""
    file_path = f"data/web/uploads/{file.filename}"
    # 保存文件
    return {"task_id": task_id}

@app.post("/api/process")
async def start_processing(task_id: str, mode: int):
    """启动处理任务"""
    # 创建独立进程
    process = multiprocessing.Process(
        target=process_worker,
        args=(task_id, mode)
    )
    process.start()
    return {"status": "processing"}

@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """查询处理进度"""
    progress = get_task_progress(task_id)
    return {
        "progress": progress,  # 0-100
        "fps": fps,
        "status": "processing"
    }

@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """取消处理任务"""
    process.terminate()  # 真实终止进程
    return {"status": "cancelled"}
```

---

## 📊 数据流程详解

### 模式3：RAFT光流完整流程

```
Frame N-1          Frame N
┌─────────┐      ┌─────────┐
│  🚗     │      │    🚗   │
│    🌲   │      │  🌲     │
│  🏠     │      │🏠       │
└─────────┘      └─────────┘
     │                │
     └────────┬───────┘
              ↓
      ┌──────────────┐
      │ RAFT光流估计 │
      │ 全图运动向量 │
      └──────┬───────┘
             │
    ┌────────┴────────┐
    │                 │
    ↓                 ↓
┌─────────┐     ┌─────────┐
│ 物体区域│     │ 背景区域│
│ 光流    │     │ 光流    │
│(车+人)  │     │(树+房子)│
└────┬────┘     └────┬────┘
     │               │
     │               ↓
     │         ┌──────────┐
     │         │背景光流均值│
     │         │= 摄像头运动│
     │         └─────┬─────┘
     │               │
     └───────┬───────┘
             ↓
    ┌─────────────────┐
    │ 运动分离         │
    │ 车真实运动 =     │
    │ 车观测运动 -     │
    │ 摄像头运动       │
    └────────┬─────────┘
             ↓
        真实速度
```

### 模式4：深度感知完整流程

```
输入帧
   │
   ├──────────┬─────────┬────────┐
   │          │         │        │
   ↓          ↓         ↓        ↓
YOLOv8    ByteTrack   RAFT    Depth V2
检测      追踪        光流     深度估计
   │          │         │        │
   └──────────┴─────────┴────────┘
              │
              ↓
    ┌────────────────────┐
    │  数据融合           │
    │ • 检测框 + ID       │
    │ • 光流向量          │
    │ • 深度值            │
    └──────┬─────────────┘
           │
           ↓
    ┌────────────────────┐
    │ 自动标定            │
    │ 像素/米 = f(深度)   │
    └──────┬─────────────┘
           │
           ↓
    ┌────────────────────┐
    │ 运动补偿速度计算    │
    │ • RAFT分离摄像头运动│
    │ • 深度修正距离      │
    │ • 输出真实速度      │
    └──────┬─────────────┘
           ↓
      输出视频 + 深度图
```

---

## 🎯 技术决策

### AI模型选择

| 模块 | 候选方案 | 最终选择 | 理由 |
|------|---------|---------|------|
| 物体检测 | YOLOv5/v8/v10 | **YOLOv8** | 精度速度平衡最优，生态成熟 |
| 目标追踪 | SORT/DeepSORT/ByteTrack | **ByteTrack** | 精度最高，卡尔曼滤波稳定 |
| 光流估计 | FlowNet/PWC-Net/RAFT | **RAFT** | Princeton出品，精度最高 |
| 深度估计 | MiDaS/DPT/Depth Anything | **Depth Anything V2** | Metric版本，绝对深度 |

### 框架选择

| 层次 | 候选方案 | 最终选择 | 理由 |
|------|---------|---------|------|
| 深度学习 | TensorFlow/PyTorch | **PyTorch** | 模型生态丰富，易用 |
| Web后端 | Flask/FastAPI/Django | **FastAPI** | 异步高性能，自动文档 |
| Web前端 | React/Vue/Angular | **Vue 3** | 易学易用，组合式API |
| 构建工具 | Webpack/Vite | **Vite** | 快速热更新，开发体验好 |

---

## ⚡ 性能优化

### 1. 模型推理优化

```python
# GPU加速
model = YOLO("yolov8n.pt").to("cuda")
model.half()  # FP16半精度

# 批处理（如果内存充足）
results = model(frames_batch, batch=8)
```

### 2. 内存优化

```python
# 逐帧处理，避免加载整个视频
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 处理单帧
    result = process_frame(frame)
    
    # 立即释放
    del frame, result
    
cap.release()
```

### 3. 多进程并行

```python
# Web模式：独立进程处理
import multiprocessing

process = multiprocessing.Process(
    target=process_video,
    args=(video_path, mode)
)
process.start()

# 主进程继续响应请求
# 处理进程独立运行，可终止
```

### 4. 实时进度反馈

```python
# 0.1%精度的进度更新
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
for i, frame in enumerate(frames):
    # 处理帧
    process_frame(frame)
    
    # 更新进度
    progress = (i / total_frames) * 100
    update_progress(task_id, progress)
```

---

## 📦 依赖管理

### CPU版本 (scripts/cpu/requirements.txt)

```
# 核心依赖
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0

# Web后端（可选）
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# 工具
tqdm>=4.66.0
```

### GPU版本 (scripts/gpu/requirements.txt)

```
# 核心依赖
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0

# GPU加速
torch>=2.0.0+cu118
torchvision>=0.15.0+cu118

# RAFT光流
raft-pytorch>=1.0.0

# Depth深度估计
depth-anything-v2>=1.0.0

# Web后端
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
```

---

## 🔄 架构演进历史

```
Phase 1 (2023-2024)
├─ OpenCV + ONNX检测
├─ SimpleTracker追踪
└─ 基础视频处理管道

Phase 2 (2024)
├─ YOLOv8原生集成
├─ ByteTrack高精度追踪
├─ 基于物体尺寸的速度估算
└─ EMA速度平滑

Phase 3 (2024-2025) ✅
├─ RAFT光流运动分离
├─ Depth Anything V2深度估计
├─ 移动摄像头支持
└─ 深度感知速度计算

Phase 4 (2025-2026) 🔧
├─ Vue 3 + FastAPI Web应用
├─ 实时进度反馈
├─ 任务取消功能
└─ 在线部署（待完成）
```

---

## 📊 性能指标

### 处理速度（CPU版本）

| 视频规格 | 模式1 | 模式2 | 模式3 | 模式4 |
|---------|-------|-------|-------|-------|
| 640×480 | 12-15 FPS | 10-12 FPS | - | - |
| 1280×720 | 8-10 FPS | 6-8 FPS | - | - |

*模式3、4需要GPU*

### 处理速度（GPU版本）

| 视频规格 | 模式1 | 模式2 | 模式3 | 模式4 |
|---------|-------|-------|-------|-------|
| 640×480 | 40-50 FPS | 35-45 FPS | 20-25 FPS | 15-20 FPS |
| 1280×720 | 25-30 FPS | 20-25 FPS | 12-15 FPS | 8-12 FPS |
| 1920×1080 | 15-20 FPS | 12-18 FPS | 8-10 FPS | 5-8 FPS |

### 内存占用

| 模式 | 640×480 | 1280×720 | 1920×1080 |
|------|---------|----------|-----------|
| 模式1 | ~200MB | ~400MB | ~600MB |
| 模式2 | ~200MB | ~400MB | ~600MB |
| 模式3 | ~1.5GB | ~2.5GB | ~4GB |
| 模式4 | ~2GB | ~3.5GB | ~5GB |

---

**文档更新：** 2026-01-05  
**对应版本：** Phase 3 核心完成，Phase 4 Web开发中
