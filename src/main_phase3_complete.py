# -*- coding: utf-8 -*-
"""
Phase 3 Complete: RAFT + Depth Anything V2 + YOLOv8
完整的Phase 3实现：光流 + 深度估计 + 物体检测追踪 + 速度估计

Features:
- YOLOv8 object detection + ByteTrack tracking
- RAFT optical flow for camera motion separation
- Depth Anything V2 for metric depth estimation
- Depth-aware speed estimation
- Supports moving camera scenarios
- Real-world distance conversion
"""
import cv2
import numpy as np
import os
import argparse
import sys
from ultralytics import YOLO

# 兼容相对导入和绝对导入
try:
    from .optical_flow_raft import RAFTOpticalFlow
    from .depth_estimation import DepthAnythingV2
except ImportError:
    from optical_flow_raft import RAFTOpticalFlow
    from depth_estimation import DepthAnythingV2


# Standard object sizes in meters (width, height)
OBJECT_REAL_SIZES = {
    'car': {'width': 1.8, 'height': 1.5},
    'truck': {'width': 2.5, 'height': 3.0},
    'bus': {'width': 2.5, 'height': 3.2},
    'motorcycle': {'width': 0.8, 'height': 1.2},
    'bicycle': {'width': 0.6, 'height': 1.0},
    'person': {'width': 0.5, 'height': 1.7},
}


class Phase3SpeedEstimator:
    """Phase 3 完整速度估计器（RAFT + Depth）"""
    
    def __init__(self, fps: float = 30.0):
        """
        初始化
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        self.pixel_to_meter = {}  # 每个track_id的像素/米比例
        self.speed_history = {}   # 速度历史（用于平滑）
        self.depth_history = {}   # 深度历史
        self.ema_alpha = 0.3      # EMA平滑系数
        
        print(f"[Phase3Estimator] Initialized with FPS={fps}")
    
    def estimate_pixel_to_meter_with_depth(self, bbox_width: float, bbox_height: float,
                                          class_name: str, depth_value: float) -> float:
        """
        根据物体尺寸和深度估计像素/米比例
        
        Args:
            bbox_width: 边界框宽度（像素）
            bbox_height: 边界框高度（像素）
            class_name: 物体类别
            depth_value: 物体深度值（归一化）
            
        Returns:
            pixel_per_meter: 像素/米比例
        """
        if class_name not in OBJECT_REAL_SIZES:
            return None
        
        real_size = OBJECT_REAL_SIZES[class_name]
        
        # 基础像素/米比例
        ppm_width = bbox_width / real_size['width']
        ppm_height = bbox_height / real_size['height']
        ppm_base = 0.7 * ppm_width + 0.3 * ppm_height
        
        # 深度修正（深度越大，物体看起来越小，需要更大的ppm）
        # 使用简单的线性关系
        depth_correction = 1.0 + depth_value * 0.5  # 深度从0-1，修正从1.0-1.5
        
        ppm = ppm_base * depth_correction
        
        return ppm
    
    def update_speed_with_depth(self, track_id: int, bbox: tuple, class_name: str,
                               real_motion: tuple, depth_value: float) -> float:
        """
        更新物体速度（使用RAFT补偿和深度信息）
        
        Args:
            track_id: 追踪ID
            bbox: 边界框 (x1, y1, x2, y2)
            class_name: 物体类别
            real_motion: RAFT补偿后的真实运动 (dx, dy) 像素
            depth_value: 物体深度值（归一化）
            
        Returns:
            speed_kmh: 速度(km/h)
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 估计像素/米比例（考虑深度）
        if track_id not in self.pixel_to_meter:
            ppm = self.estimate_pixel_to_meter_with_depth(
                bbox_width, bbox_height, class_name, depth_value
            )
            if ppm is None:
                return 0.0
            self.pixel_to_meter[track_id] = ppm
        
        ppm = self.pixel_to_meter[track_id]
        
        # 存储深度历史
        if track_id not in self.depth_history:
            self.depth_history[track_id] = []
        self.depth_history[track_id].append(depth_value)
        if len(self.depth_history[track_id]) > 30:
            self.depth_history[track_id].pop(0)
        
        # 真实运动（已由RAFT补偿）
        dx_pixel, dy_pixel = real_motion
        
        # 计算运动距离（像素）
        distance_pixel = np.sqrt(dx_pixel**2 + dy_pixel**2)
        
        # 深度动态修正（如果深度变化，调整距离）
        if len(self.depth_history[track_id]) >= 2:
            depth_change = self.depth_history[track_id][-1] - self.depth_history[track_id][-2]
            # 深度减小（靠近），实际距离可能更大
            # 深度增加（远离），实际距离可能更小
            distance_pixel *= (1.0 - depth_change * 0.3)
        
        # 转换为米
        distance_meter = distance_pixel / ppm
        
        # 计算速度（米/秒）
        speed_ms = distance_meter * self.fps
        
        # 转换为km/h
        speed_kmh = speed_ms * 3.6
        
        # EMA平滑
        if track_id in self.speed_history:
            speed_kmh = self.ema_alpha * speed_kmh + (1 - self.ema_alpha) * self.speed_history[track_id]
        
        self.speed_history[track_id] = speed_kmh
        
        return speed_kmh


def process_video_phase3(input_path: str, output_path: str,
                        show_video: bool = True,
                        conf_threshold: float = 0.25,
                        show_depth: bool = True,
                        depth_frequency: int = 10):
    """
    Phase 3完整处理：RAFT + Depth + YOLOv8
    
    Args:
        input_path: 输入视频
        output_path: 输出视频
        show_video: 是否显示窗口
        conf_threshold: 检测阈值
        show_depth: 是否显示深度图
        depth_frequency: 深度估计频率（每N帧）
    """
    print("=" * 60)
    print("Phase 3 Complete: RAFT + Depth Anything V2 + YOLOv8")
    print("=" * 60)
    
    # 1. 初始化YOLOv8
    print("\n[1/4] Loading YOLOv8...")
    model = YOLO('yolov8n.pt')
    print("✅ YOLOv8 loaded")
    
    # 2. 初始化RAFT
    print("\n[2/4] Loading RAFT...")
    raft = RAFTOpticalFlow(model_type='small', device='auto')
    print("✅ RAFT loaded")
    
    # 3. 初始化Depth Anything V2
    print("\n[3/4] Loading Depth Anything V2...")
    depth_estimator = DepthAnythingV2(model_size='small', device='auto')
    print("✅ Depth Anything V2 loaded")
    
    # 4. 打开视频
    print("\n[4/4] Opening video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open: {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
    
    # 5. 初始化输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 6. 初始化估计器
    speed_estimator = Phase3SpeedEstimator(fps=fps)
    
    print("\nProcessing video...")
    print("-" * 60)
    
    frame_idx = 0
    prev_frame = None
    track_positions = {}
    depth_map_cache = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # YOLOv8检测和追踪
        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)
        
        # 计算光流和摄像头运动
        camera_motion = (0.0, 0.0)
        if prev_frame is not None:
            flow = raft.compute_flow(prev_frame, frame)
            camera_motion = raft.estimate_camera_motion(flow, method='median')
        
        # 深度估计（每N帧）
        if frame_idx % depth_frequency == 1 or depth_map_cache is None:
            depth_map_cache = depth_estimator.estimate_depth(frame)
        
        prev_frame = frame.copy()
        
        # 绘制结果
        annotated_frame = frame.copy()
        
        # 处理检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes))
            
            for i, (box, class_id, conf, track_id) in enumerate(zip(boxes, class_ids, confidences, track_ids)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = model.names[class_id]
                
                # 当前中心
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 获取物体深度
                depth_value = depth_estimator.get_object_depth(depth_map_cache, (x1, y1, x2, y2))
                # 归一化深度值
                depth_normalized = (depth_value - depth_map_cache.min()) / (depth_map_cache.max() - depth_map_cache.min())
                
                # 计算速度
                speed = 0.0
                if class_name in OBJECT_REAL_SIZES:
                    if track_id in track_positions:
                        prev_cx, prev_cy = track_positions[track_id]
                        apparent_dx = cx - prev_cx
                        apparent_dy = cy - prev_cy
                        
                        # 真实运动 = 表观运动 - 摄像头运动
                        real_dx = apparent_dx - camera_motion[0]
                        real_dy = apparent_dy - camera_motion[1]
                        real_motion = (real_dx, real_dy)
                        
                        # 使用深度信息计算速度
                        speed = speed_estimator.update_speed_with_depth(
                            track_id, (x1, y1, x2, y2), class_name, real_motion, depth_normalized
                        )
                    
                    track_positions[track_id] = (cx, cy)
                
                # 绘制边界框（颜色根据深度）
                depth_color = int(depth_normalized * 255)
                color = (0, 255 - depth_color, depth_color)  # 近：绿色，远：红色
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"ID:{track_id} {class_name}"
                if speed > 0:
                    label += f" {speed:.1f}km/h"
                label += f" D:{depth_normalized:.2f}"
                
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 250, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示深度图
        if show_depth and frame_idx % depth_frequency == 1:
            depth_vis = depth_estimator.visualize_depth(depth_map_cache)
            depth_small = cv2.resize(depth_vis, (width // 4, height // 4))
            annotated_frame[10:10+height//4, width-width//4-10:width-10] = depth_small
            cv2.putText(annotated_frame, "Depth Map", (width-width//4-10, height//4+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 信息显示
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames} | Phase 3 Complete",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Camera: dx={camera_motion[0]:.1f} dy={camera_motion[1]:.1f}",
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 写入和显示
        out.write(annotated_frame)
        if show_video:
            cv2.imshow('Phase 3: RAFT + Depth + YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_idx % 30 == 0:
            print(f"Progress: {frame_idx/total_frames*100:.1f}% ({frame_idx}/{total_frames})", end='\r')
    
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Phase 3 Processing Complete!")
    print(f"📹 Output: {output_path}")
    print(f"🎯 Processed {frame_idx} frames")
    print(f"{'=' * 60}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 3 Complete: RAFT + Depth + YOLOv8')
    parser.add_argument('--input', type=str, default='input/test_video.mp4')
    parser.add_argument('--output', type=str, default='output/phase3_output.mp4')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--depth-freq', type=int, default=10,
                       help='Depth estimation frequency (every N frames)')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    success = process_video_phase3(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_show,
        conf_threshold=args.conf,
        show_depth=not args.no_depth,
        depth_frequency=args.depth_freq
    )
    
    if success:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed!")
        sys.exit(1)


# 兼容性包装函数（用于Web API）
def process_video(input_path: str, output_path: str, 
                  show_video: bool = True,
                  conf_threshold: float = 0.25,
                  **kwargs):
    """
    兼容性包装函数，统一接口
    """
    return process_video_phase3(
        input_path=input_path,
        output_path=output_path,
        show_video=show_video,
        conf_threshold=conf_threshold,
        show_depth=kwargs.get('show_depth', True),
        depth_frequency=kwargs.get('depth_frequency', 10)
    )
