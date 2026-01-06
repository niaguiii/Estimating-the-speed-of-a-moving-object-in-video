# -*- coding: utf-8 -*-
"""
YOLOv8 + ByteTrack + RAFT Optical Flow + Speed Estimation
Phase 3 Implementation - Camera Motion Separation

Features:
- YOLOv8 object detection
- ByteTrack high-precision tracking
- RAFT optical flow for camera motion estimation
- Object real motion separation (removes camera motion)
- Speed estimation with camera motion compensation
- Real-time visualization

This version supports moving camera scenarios!
"""
import cv2
import numpy as np
import os
import argparse
import sys

# ⚠️ 必须先导入model_config设置环境变量
try:
    from . import model_config
except ImportError:
    import model_config

from ultralytics import YOLO

# 兼容相对导入和绝对导入
try:
    from .optical_flow_raft import RAFTOpticalFlow
except ImportError:
    from optical_flow_raft import RAFTOpticalFlow


# Standard object sizes in meters (width, height)
OBJECT_REAL_SIZES = {
    'car': {'width': 1.8, 'height': 1.5},
    'truck': {'width': 2.5, 'height': 3.0},
    'bus': {'width': 2.5, 'height': 3.2},
    'motorcycle': {'width': 0.8, 'height': 1.2},
    'bicycle': {'width': 0.6, 'height': 1.0},
    'person': {'width': 0.5, 'height': 1.7},
}


class SpeedEstimatorWithRAFT:
    """速度估计器（带RAFT光流补偿）"""
    
    def __init__(self, fps: float = 30.0):
        """
        初始化
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        self.pixel_to_meter = {}  # 每个track_id的像素/米比例
        self.speed_history = {}   # 速度历史（用于平滑）
        self.position_history = {}  # 位置历史
        self.ema_alpha = 0.3  # EMA平滑系数
        
        print(f"[SpeedEstimator] Initialized with FPS={fps}")
    
    def estimate_pixel_to_meter(self, bbox_width: float, bbox_height: float, 
                                class_name: str) -> float:
        """
        根据物体尺寸估计像素/米比例
        
        Args:
            bbox_width: 边界框宽度（像素）
            bbox_height: 边界框高度（像素）
            class_name: 物体类别
            
        Returns:
            pixel_per_meter: 像素/米比例，如果无法估计返回None
        """
        if class_name not in OBJECT_REAL_SIZES:
            return None
        
        real_size = OBJECT_REAL_SIZES[class_name]
        
        # 使用宽度和高度分别估计，取平均
        ppm_width = bbox_width / real_size['width']
        ppm_height = bbox_height / real_size['height']
        
        # 加权平均（宽度权重更高，因为透视畸变影响较小）
        ppm = 0.7 * ppm_width + 0.3 * ppm_height
        
        return ppm
    
    def update_speed(self, track_id: int, bbox: tuple, class_name: str,
                    real_motion: tuple) -> float:
        """
        更新物体速度（使用RAFT补偿后的真实运动）
        
        Args:
            track_id: 追踪ID
            bbox: 边界框 (x1, y1, x2, y2)
            class_name: 物体类别
            real_motion: RAFT补偿后的真实运动 (dx, dy) 像素
            
        Returns:
            speed_kmh: 速度(km/h)，如果无法计算返回0
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # 估计像素/米比例（只在第一次或周期性更新）
        if track_id not in self.pixel_to_meter:
            ppm = self.estimate_pixel_to_meter(bbox_width, bbox_height, class_name)
            if ppm is None:
                return 0.0
            self.pixel_to_meter[track_id] = ppm
        
        ppm = self.pixel_to_meter[track_id]
        
        # 真实运动（已由RAFT补偿）
        dx_pixel, dy_pixel = real_motion
        
        # 计算运动距离（像素）
        distance_pixel = np.sqrt(dx_pixel**2 + dy_pixel**2)
        
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


def process_video_with_raft(input_path: str, output_path: str, 
                            show_video: bool = True,
                            conf_threshold: float = 0.25,
                            show_flow: bool = False):
    """
    使用RAFT光流处理视频
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        show_video: 是否显示实时窗口
        conf_threshold: 检测置信度阈值
        show_flow: 是否显示光流可视化
    """
    print("=" * 60)
    print("Phase 3: YOLOv8 + RAFT Optical Flow + Speed Estimation")
    print("=" * 60)
    
    # 1. 初始化YOLO模型
    print("\n[1/4] Loading YOLOv8 model...")
    yolo_path = model_config.get_model_path('yolov8n.pt')
    model = YOLO(yolo_path)
    print(f"✅ YOLOv8 loaded from {yolo_path}")
    
    # 2. 初始化RAFT
    print("\n[2/4] Loading RAFT model...")
    raft = RAFTOpticalFlow(model_type='small', device='auto')
    print("✅ RAFT loaded")
    
    # 3. 打开视频
    print("\n[3/4] Opening video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {input_path}")
        return False
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
    
    # 4. 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 5. 初始化速度估计器
    speed_estimator = SpeedEstimatorWithRAFT(fps=fps)
    
    print("\n[4/4] Processing video...")
    print("-" * 60)
    
    frame_idx = 0
    prev_frame = None
    camera_motion_history = []
    track_positions = {}  # 存储每个track的历史位置
    
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
            camera_motion_history.append(camera_motion)
            
            # 可视化光流（可选）
            if show_flow and frame_idx % 10 == 0:  # 每10帧显示一次
                flow_vis = raft.visualize_flow(flow)
                cv2.imshow('Optical Flow', cv2.resize(flow_vis, (width//2, height//2)))
        
        prev_frame = frame.copy()
        
        # 绘制结果
        annotated_frame = frame.copy()
        
        # 处理检测结果
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # 获取track IDs
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes))
            
            for i, (box, class_id, conf, track_id) in enumerate(zip(boxes, class_ids, confidences, track_ids)):
                x1, y1, x2, y2 = box.astype(int)
                class_name = model.names[class_id]
                
                # 当前物体中心
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 计算物体的真实运动（RAFT补偿）
                speed = 0.0
                if class_name in OBJECT_REAL_SIZES:
                    # 计算表观运动（使用track历史位置）
                    if track_id in track_positions:
                        prev_cx, prev_cy = track_positions[track_id]
                        # 表观运动（像素/帧）
                        apparent_dx = cx - prev_cx
                        apparent_dy = cy - prev_cy
                        
                        # 真实运动 = 表观运动 - 摄像头运动
                        # 注意：RAFT的摄像头运动是全局平均，需要取反
                        real_dx = apparent_dx - camera_motion[0]
                        real_dy = apparent_dy - camera_motion[1]
                        real_motion = (real_dx, real_dy)
                        
                        # 计算速度
                        speed = speed_estimator.update_speed(track_id, (x1, y1, x2, y2), 
                                                            class_name, real_motion)
                    
                    # 更新位置历史
                    track_positions[track_id] = (cx, cy)
                
                # 绘制边界框
                color = (0, 255, 0) if speed > 0 else (255, 0, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                if speed > 0:
                    label += f" {speed:.1f}km/h"
                
                # 背景
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), 
                            (x1 + label_w, y1), color, -1)
                
                # 文字
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制摄像头运动信息
        if camera_motion != (0.0, 0.0):
            cam_text = f"Camera Motion: dx={camera_motion[0]:.1f} dy={camera_motion[1]:.1f}"
            cv2.putText(annotated_frame, cam_text, (10, height - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 绘制帧信息
        info_text = f"Frame: {frame_idx}/{total_frames} | RAFT+YOLOv8"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 写入视频
        out.write(annotated_frame)
        
        # 显示
        if show_video:
            cv2.imshow('RAFT + YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[STOP] User interrupted")
                break
        
        # 进度
        if frame_idx % 30 == 0:
            progress = frame_idx / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames})", end='\r')
    
    # 清理
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Processing complete!")
    print(f"📹 Output: {output_path}")
    print(f"🎯 Processed {frame_idx} frames")
    print(f"{'=' * 60}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 3: RAFT Optical Flow Speed Estimation')
    parser.add_argument('--input', type=str, default='input/test_video.mp4', 
                       help='Input video path')
    parser.add_argument('--output', type=str, default='output/raft_output.mp4',
                       help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not show video window')
    parser.add_argument('--show-flow', action='store_true',
                       help='Show optical flow visualization')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 处理视频
    success = process_video_with_raft(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_show,
        conf_threshold=args.conf,
        show_flow=args.show_flow
    )
    
    if success:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed!")


# 兼容性包装函数（用于Web API）
def process_video(input_path: str, output_path: str, 
                  show_video: bool = True,
                  conf_threshold: float = 0.25,
                  **kwargs):
    """
    兼容性包装函数，统一接口
    """
    return process_video_with_raft(
        input_path=input_path,
        output_path=output_path,
        show_video=show_video,
        conf_threshold=conf_threshold,
        show_flow=kwargs.get('show_flow', False)
    )
