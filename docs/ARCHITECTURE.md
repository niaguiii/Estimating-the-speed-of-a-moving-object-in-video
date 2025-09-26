# 技术架构文档

## 🏗️ 系统架构概览

### 分层架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    用户界面层                             │
│   main.py (统一入口) │ install.bat/ps1 (一键安装)       │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    应用逻辑层                             │
│ main_opencv.py (ONNX版) │ main_yolov8_native.py (原生版) │
│              config.py (系统配置)                        │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    AI算法层                              │
│ YOLOv8检测 │ SimpleTracker追踪 │ 美观显示 │ 错误恢复      │
└─────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────┐
│                    框架层                                │
│ OpenCV DNN (ONNX) │ Ultralytics (原生) │ OpenCV Core     │
└─────────────────────────────────────────────────────────┘
```

## 🔄 数据处理流程

### 当前流程 (第一阶段)
```
输入视频 → 帧提取 → YOLOv8检测 → SimpleTracker → 美观渲染 → 输出视频
   │         │         │          │             │          │
  MP4     OpenCV   ONNX/原生    距离匹配      类别+ID显示  保存文件
```

### 规划流程 (第二阶段 - ByteTrack集成)
```
输入视频 → 帧提取 → YOLOv8检测 → ByteTrack → 速度计算 → 输出分析
   │         │         │          │         │          │
  MP4     OpenCV   ONNX/原生   卡尔曼滤波  真实速度    CSV/JSON
                              两阶段匹配
```

## 🧩 核心模块设计

### 1. YOLOv8检测引擎 (main_opencv.py)

#### OpenCVObjectDetector类
```python
class OpenCVObjectDetector:
    def __init__(self):
        self.net = None
        self.model_type = None  # 'onnx' 或 'cascade'
        self.setup_yolo_model()
    
    def setup_yolo_model(self):
        """智能模型加载：ONNX -> 备用检测"""
        if os.path.exists('models/yolov8n.onnx'):
            self.net = cv2.dnn.readNetFromONNX('models/yolov8n.onnx')
            self.model_type = 'onnx'
        else:
            self.setup_fallback_detector()  # 人脸检测备用
```

#### YOLOv8 ONNX实现细节
- **输入格式**: 640×640 RGB图像，归一化到[0,1]
- **输出格式**: [1, 84, 8400] 张量
- **后处理**: 置信度筛选 → NMS → 坐标转换

### 2. 物体追踪算法

#### SimpleTracker类
```python
class SimpleTracker:
    def __init__(self):
        self.tracks = []           # 当前追踪列表
        self.next_id = 1          # ID生成器
        self.max_disappeared = 10  # 消失阈值
    
    def update(self, detections):
        """核心追踪逻辑"""
        # 1. 距离匹配：欧几里得距离最近邻
        # 2. ID分配：新物体分配新ID
        # 3. 状态管理：消失计数器
```

#### 追踪算法特点
- **匹配策略**: 基于中心点距离的贪婪匹配
- **ID管理**: 单调递增，避免ID重复
- **状态维护**: 简单有效的消失/重现处理

### 3. 用户界面系统 (main.py)

#### 门面模式设计
```python
def main():
    """用户友好的统一入口"""
    setup_directories()          # 自动创建文件夹
    videos = get_input_videos()  # 扫描输入视频
    selected = select_video()    # 用户选择
    output_path = get_output_filename()  # 自动命名
    
    # 调用核心引擎
    from main_opencv import process_video
    process_video(selected, output_path)
```

## 🎯 关键技术决策

### AI模型选择

| 方案 | 精度 | 速度 | 兼容性 | 最终选择 |
|------|------|------|--------|----------|
| YOLOv8 ONNX | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **采用** |
| 人脸检测 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🛡️ **备用** |

### 架构优势

#### 1. 智能降级机制
```python
def create_detector():
    try:
        return YOLOv8Detector()      # 主力方案
    except ModelNotFoundError:
        return FaceDetector()        # 保底方案
```

#### 2. 零依赖冲突
- **纯OpenCV实现**: 避免深度学习框架冲突
- **ONNX标准**: 跨平台模型格式
- **最小依赖**: 仅需opencv-python和numpy

#### 3. 内存优化
- **逐帧处理**: 避免整个视频加载到内存
- **即时释放**: 处理完的帧立即释放
- **流式设计**: 支持任意长度视频

## 🚀 性能优化设计

### 计算优化
```python
# ONNX推理优化
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True)
net.setInput(blob)
outputs = net.forward()  # OpenCV DNN硬件优化

# 内存管理
del frame, outputs  # 主动释放大对象
```

### 追踪优化
```python
# 距离计算优化
def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# 避免全对全计算，使用距离阈值提前剪枝
```

## 🔧 配置系统设计

### 集中配置管理 (config.py)
```python
YOLO_CONFIG = {
    'onnx_model': 'models/yolov8n.onnx',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'input_size': (640, 640)
}

TRACKING_CONFIG = {
    'max_distance': 100,        # 像素
    'max_disappeared': 10       # 帧数
}
```

### 可扩展设计
- **插件化检测器**: 易于替换不同AI模型
- **配置驱动**: 关键参数可调整
- **接口标准化**: 统一的检测和追踪API

## 🔮 未来扩展架构

### 第二阶段：速度计算扩展
```python
class SpeedCalculator:
    def __init__(self):
        self.calibration = CameraCalibration()
        self.trajectory = TrajectoryAnalyzer()
    
    def calculate_speed(self, tracks, time_delta):
        """像素轨迹 → 真实速度"""
        # 1. 坐标转换：像素 → 世界坐标
        # 2. 距离计算：轨迹长度
        # 3. 速度计算：距离/时间
```

### 第三阶段：Web服务架构
```python
# FastAPI后端
@app.post("/api/process_video")
async def process_video_api(video: UploadFile):
    # 调用当前检测引擎
    result = await process_video_async(video)
    return {"speed_data": result}

# 前端React/Vue.js
// 视频上传 → API调用 → 结果展示
```

## 📊 性能监控

### 关键指标
- **检测延迟**: 单帧检测时间
- **内存使用**: 峰值内存占用
- **追踪精度**: ID切换频率
- **系统稳定性**: 错误恢复成功率

### 监控实现
```python
class PerformanceMonitor:
    def track_detection_time(self, start_time):
        self.detection_times.append(time.time() - start_time)
    
    def log_memory_usage(self):
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
```

## 🎯 架构演进路径

```
当前架构 (第一阶段) ✅
├── 双版本YOLOv8检测 (ONNX + 原生)
├── SimpleTracker追踪
├── 美观可视化显示
└── 完整视频处理管道

速度计算架构 (第二阶段) 🔄
├── ByteTrack高精度追踪
│   ├── 卡尔曼滤波预测
│   ├── 两阶段匹配策略
│   └── IoU + 运动一致性
├── 摄像头标定系统
├── 轨迹分析引擎
└── 精确速度计算

完整系统架构 (第三阶段) 📋
├── RESTful API服务层
├── React/Vue Web前端
├── 数据库存储系统
└── 云端部署优化
```

## 🚀 ByteTrack集成技术规划

### 架构对比
| 组件 | SimpleTracker (当前) | ByteTrack (规划) |
|------|---------------------|------------------|
| **匹配策略** | 距离最近邻 | 两阶段检测匹配 |
| **运动预测** | 无 | 卡尔曼滤波 |
| **相似度计算** | 欧氏距离 | IoU + 运动 + 外观 |
| **遮挡处理** | disappeared计数 | 低置信度恢复 |
| **精度** | ~60-70% | ~80-90% |

### 集成实现计划
```python
# 第二阶段目标架构
class ByteTrackDetector:
    def __init__(self):
        self.yolo_detector = YOLOv8Detector()
        self.byte_tracker = BYTETracker()
        self.kalman_filter = KalmanFilter()
    
    def track_objects(self, frame):
        # 高低置信度分离
        high_conf_dets, low_conf_dets = self.separate_detections(frame)
        
        # 两阶段匹配
        tracks = self.byte_tracker.update(high_conf_dets, low_conf_dets)
        
        return tracks
```

这个架构设计确保了系统的**高性能**、**高可靠性**和**强扩展性**，为实现完整的速度估计解决方案提供了坚实的技术基础。