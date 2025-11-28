"""
ByteTrack 追踪器封装
支持多种实现方式：
1. Ultralytics 内置追踪（推荐，YOLOv8原生版本）
2. supervision 库（如果可用）
3. SimpleTracker（备用）

ByteTrack 优势：
- 卡尔曼滤波运动预测
- 两阶段检测匹配策略（高置信度+低置信度）
- 更好的遮挡处理
- 追踪精度 80-90%
"""

import numpy as np

# 检查可用的追踪库
SUPERVISION_AVAILABLE = False
SUPERVISION_HAS_BYTETRACK = False

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
    # 检查是否有ByteTrack
    if hasattr(sv, 'ByteTrack'):
        SUPERVISION_HAS_BYTETRACK = True
except ImportError:
    pass


class ByteTrackWrapper:
    """
    ByteTrack 追踪器封装类
    
    提供与 SimpleTracker 兼容的接口
    当 supervision ByteTrack 不可用时，自动降级到 SimpleTracker
    """
    
    def __init__(self, 
                 track_thresh: float = 0.25,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        """
        初始化追踪器
        """
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        # 存储轨迹历史（用于速度计算）
        self.track_history = {}
        self.frame_count = 0
        
        # 尝试使用 supervision ByteTrack
        if SUPERVISION_HAS_BYTETRACK:
            try:
                self.tracker = sv.ByteTrack(
                    track_activation_threshold=track_thresh,
                    lost_track_buffer=track_buffer,
                    minimum_matching_threshold=match_thresh,
                    frame_rate=frame_rate
                )
                self.tracker_type = 'bytetrack'
                print(f"[OK] ByteTrack initialized (supervision)")
            except Exception as e:
                print(f"[WARN] ByteTrack init failed: {e}")
                self._init_simple_tracker()
        else:
            self._init_simple_tracker()
        
        if self.tracker_type == 'bytetrack':
            print(f"   - track_thresh: {track_thresh}")
            print(f"   - track_buffer: {track_buffer} frames")
    
    def _init_simple_tracker(self):
        """初始化简单追踪器作为备用"""
        self.tracker = None
        self.tracker_type = 'simple'
        self.tracks = []
        self.next_id = 1
        self.max_disappeared = self.track_buffer
        self.distance_threshold = 100
        print("[INFO] Using SimpleTracker (built-in)")
    
    def update(self, detections: list, frame: np.ndarray = None) -> list:
        """更新追踪器"""
        self.frame_count += 1
        
        if self.tracker_type == 'bytetrack':
            return self._update_bytetrack(detections)
        else:
            return self._update_simple(detections)
    
    def _update_bytetrack(self, detections: list) -> list:
        """使用 ByteTrack 更新"""
        if len(detections) == 0:
            return []
        
        # 转换为 supervision 格式
        xyxy = []
        confidences = []
        class_ids = []
        
        for det in detections:
            x, y, w, h = det['bbox']
            xyxy.append([x, y, x + w, y + h])
            confidences.append(det.get('confidence', 1.0))
            class_ids.append(det.get('class_id', 0))
        
        sv_detections = sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int)
        )
        
        # ByteTrack 更新
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # 转换回我们的格式
        tracks = []
        class_name_map = {det.get('class_id', 0): det.get('class_name', 'unknown') for det in detections}
        
        if tracked is not None and len(tracked) > 0:
            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i]
                x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i + 1
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                
                tracks.append({
                    'id': track_id,
                    'bbox': [x, y, w, h],
                    'class_name': class_name_map.get(class_id, 'unknown'),
                    'class_id': class_id,
                    'confidence': confidence,
                    'disappeared': 0
                })
        
        self._update_track_history(tracks)
        return tracks
    
    def _update_simple(self, detections: list) -> list:
        """使用简单追踪器更新"""
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection.get('class_name', 'unknown'),
                    'class_id': detection.get('class_id', 0),
                    'confidence': detection.get('confidence', 1.0),
                    'disappeared': 0
                })
                self.next_id += 1
        else:
            self._match_detections_simple(detections)
        
        self.tracks = [t for t in self.tracks if t['disappeared'] < self.max_disappeared]
        self._update_track_history([t for t in self.tracks if t['disappeared'] == 0])
        return self.tracks
    
    def _match_detections_simple(self, detections: list):
        """简单的最近邻匹配"""
        for track in self.tracks:
            track['disappeared'] += 1
        
        matched_tracks = set()
        
        for detection in detections:
            best_match = None
            min_distance = float('inf')
            det_center = self._get_center(detection['bbox'])
            
            for track in self.tracks:
                if track['id'] in matched_tracks:
                    continue
                if track['disappeared'] < self.max_disappeared:
                    track_center = self._get_center(track['bbox'])
                    distance = np.sqrt((det_center[0] - track_center[0])**2 + 
                                       (det_center[1] - track_center[1])**2)
                    
                    if distance < min_distance and distance < self.distance_threshold:
                        min_distance = distance
                        best_match = track
            
            if best_match:
                best_match['bbox'] = detection['bbox']
                best_match['class_name'] = detection.get('class_name', best_match['class_name'])
                best_match['confidence'] = detection.get('confidence', 1.0)
                best_match['disappeared'] = 0
                matched_tracks.add(best_match['id'])
            else:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection.get('class_name', 'unknown'),
                    'class_id': detection.get('class_id', 0),
                    'confidence': detection.get('confidence', 1.0),
                    'disappeared': 0
                })
                self.next_id += 1
    
    def _get_center(self, bbox: list) -> tuple:
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _update_track_history(self, tracks: list):
        """更新轨迹历史"""
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            center = self._get_center(bbox)
            
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame': self.frame_count,
                'bbox': bbox,
                'center': center,
                'class_name': track['class_name']
            })
            
            if len(self.track_history[track_id]) > 100:
                self.track_history[track_id] = self.track_history[track_id][-100:]
    
    def get_track_history(self, track_id: int) -> list:
        return self.track_history.get(track_id, [])
    
    def get_pixel_velocity(self, track_id: int, num_frames: int = 5) -> tuple:
        """计算像素速度"""
        history = self.track_history.get(track_id, [])
        if len(history) < 2:
            return (0, 0)
        
        recent = history[-min(num_frames, len(history)):]
        if len(recent) < 2:
            return (0, 0)
        
        start, end = recent[0], recent[-1]
        dx = end['center'][0] - start['center'][0]
        dy = end['center'][1] - start['center'][1]
        dt = end['frame'] - start['frame']
        
        return (dx / dt, dy / dt) if dt > 0 else (0, 0)
    
    def reset(self):
        """重置追踪器"""
        if self.tracker_type == 'bytetrack' and self.tracker:
            self.tracker.reset()
        else:
            self.tracks.clear()
            self.next_id = 1
        self.track_history.clear()
        self.frame_count = 0


class SimpleTracker:
    """
    简单的物体追踪器（备用）
    基于欧几里得距离的最近邻匹配
    """
    
    def __init__(self, max_disappeared: int = 30, distance_threshold: int = 100):
        """
        初始化简单追踪器
        
        Args:
            max_disappeared: 最大容忍消失帧数
            distance_threshold: 匹配距离阈值（像素）
        """
        self.tracks = []
        self.next_id = 1
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.track_history = {}
        self.frame_count = 0
        
        print(f"✅ SimpleTracker 初始化成功")
        print(f"   - 最大消失帧数: {max_disappeared}")
        print(f"   - 距离阈值: {distance_threshold} 像素")
    
    def update(self, detections: list, frame: np.ndarray = None) -> list:
        """更新追踪"""
        self.frame_count += 1
        
        if len(self.tracks) == 0:
            # 初始化追踪
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection.get('class_name', 'unknown'),
                    'class_id': detection.get('class_id', 0),
                    'confidence': detection.get('confidence', 1.0),
                    'disappeared': 0
                })
                self.next_id += 1
        else:
            self._match_detections(detections)
        
        # 移除消失太久的追踪
        self.tracks = [t for t in self.tracks if t['disappeared'] < self.max_disappeared]
        
        # 更新轨迹历史
        self._update_track_history()
        
        return self.tracks
    
    def _match_detections(self, detections: list):
        """匹配检测结果到追踪"""
        for track in self.tracks:
            track['disappeared'] += 1
        
        matched_tracks = set()
        
        for detection in detections:
            best_match = None
            min_distance = float('inf')
            det_center = self._get_center(detection['bbox'])
            
            for track in self.tracks:
                if track['id'] in matched_tracks:
                    continue
                    
                if track['disappeared'] < self.max_disappeared:
                    track_center = self._get_center(track['bbox'])
                    distance = self._calculate_distance(det_center, track_center)
                    
                    if distance < min_distance and distance < self.distance_threshold:
                        min_distance = distance
                        best_match = track
            
            if best_match:
                best_match['bbox'] = detection['bbox']
                best_match['class_name'] = detection.get('class_name', best_match['class_name'])
                best_match['class_id'] = detection.get('class_id', best_match.get('class_id', 0))
                best_match['confidence'] = detection.get('confidence', 1.0)
                best_match['disappeared'] = 0
                matched_tracks.add(best_match['id'])
            else:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection.get('class_name', 'unknown'),
                    'class_id': detection.get('class_id', 0),
                    'confidence': detection.get('confidence', 1.0),
                    'disappeared': 0
                })
                self.next_id += 1
    
    def _get_center(self, bbox: list) -> tuple:
        """获取边界框中心点"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def _calculate_distance(self, p1: tuple, p2: tuple) -> float:
        """计算两点距离"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _update_track_history(self):
        """更新轨迹历史"""
        for track in self.tracks:
            if track['disappeared'] == 0:
                track_id = track['id']
                bbox = track['bbox']
                center = self._get_center(bbox)
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append({
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'center': center,
                    'class_name': track['class_name']
                })
                
                if len(self.track_history[track_id]) > 100:
                    self.track_history[track_id] = self.track_history[track_id][-100:]
    
    def get_track_history(self, track_id: int) -> list:
        """获取指定ID的轨迹历史"""
        return self.track_history.get(track_id, [])
    
    def get_pixel_velocity(self, track_id: int, num_frames: int = 5) -> tuple:
        """计算像素速度"""
        history = self.track_history.get(track_id, [])
        
        if len(history) < 2:
            return (0, 0)
        
        recent = history[-min(num_frames, len(history)):]
        
        if len(recent) < 2:
            return (0, 0)
        
        start = recent[0]
        end = recent[-1]
        
        dx = end['center'][0] - start['center'][0]
        dy = end['center'][1] - start['center'][1]
        dt = end['frame'] - start['frame']
        
        if dt == 0:
            return (0, 0)
        
        return (dx / dt, dy / dt)
    
    def reset(self):
        """重置追踪器"""
        self.tracks.clear()
        self.track_history.clear()
        self.next_id = 1
        self.frame_count = 0


def create_tracker(tracker_type: str = 'bytetrack', **kwargs):
    """
    创建追踪器的工厂函数
    
    Args:
        tracker_type: 'bytetrack' 或 'simple'
        **kwargs: 追踪器参数
    
    Returns:
        追踪器实例
    """
    if tracker_type.lower() == 'bytetrack':
        if SUPERVISION_AVAILABLE:
            return ByteTrackWrapper(**kwargs)
        else:
            print("⚠️ supervision 未安装，回退到 SimpleTracker")
            return SimpleTracker(**kwargs)
    else:
        return SimpleTracker(**kwargs)
