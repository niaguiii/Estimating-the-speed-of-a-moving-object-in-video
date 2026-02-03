# -*- coding: utf-8 -*-
"""
Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8
升级版Phase 3：光流 + 绝对深度估计 + 物体检测追踪 + 精确速度估计

Features:
- YOLOv8 object detection + ByteTrack tracking
- RAFT optical flow for camera motion separation
- Metric3D v2 for ABSOLUTE metric depth (输出真实米数！)
- Depth-aware 3D speed estimation
- Supports moving camera scenarios
- NO manual calibration needed (自动标定)
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
    from .depth_estimation_metric3d import Metric3Dv2
except ImportError:
    from optical_flow_raft import RAFTOpticalFlow
    from depth_estimation_metric3d import Metric3Dv2


class Metric3DSpeedEstimator:
    """Metric3D 3D速度估计器（真实3D空间测速）"""
    
    def __init__(self, fps: float = 30.0):
        """
        初始化
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        self.speed_history = {}   # 速度历史（用于平滑）
        self.position_history = {} # 3D位置历史
        self.ema_alpha = 0.3      # EMA平滑系数
        
        print(f"[Metric3DEstimator] Initialized with FPS={fps}")
    
    def calculate_3d_speed(self, track_id: int, 
                          current_pos_2d: tuple, 
                          current_depth: float,
                          camera_motion_2d: tuple,
                          intrinsics: dict) -> float:
        """
        计算3D空间速度（m/s）
        
        Args:
            track_id: 追踪ID
            current_pos_2d: 当前2D位置 (x_pixel, y_pixel)
            current_depth: 当前深度（米）
            camera_motion_2d: 摄像头2D运动 (dx, dy) 像素
            intrinsics: 相机内参
            
        Returns:
            speed_ms: 速度(m/s)
        """
        # 补偿摄像头运动
        cx, cy = current_pos_2d
        compensated_cx = cx - camera_motion_2d[0]
        compensated_cy = cy - camera_motion_2d[1]
        
        # 2D + 深度 → 3D坐标
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx_cam = intrinsics['cx']
        cy_cam = intrinsics['cy']
        
        X = (compensated_cx - cx_cam) * current_depth / fx
        Y = (compensated_cy - cy_cam) * current_depth / fy
        Z = current_depth
        
        current_pos_3d = (X, Y, Z)
        
        # 如果有历史位置，计算速度
        if track_id in self.position_history:
            prev_pos_3d = self.position_history[track_id]
            
            # 计算3D距离（米）
            dx_3d = current_pos_3d[0] - prev_pos_3d[0]
            dy_3d = current_pos_3d[1] - prev_pos_3d[1]
            dz_3d = current_pos_3d[2] - prev_pos_3d[2]
            
            distance_3d = np.sqrt(dx_3d**2 + dy_3d**2 + dz_3d**2)
            
            # 计算速度（米/秒）
            speed_ms = distance_3d * self.fps
            
            # EMA平滑
            if track_id in self.speed_history:
                speed_ms = self.ema_alpha * speed_ms + (1 - self.ema_alpha) * self.speed_history[track_id]
            
            self.speed_history[track_id] = speed_ms
            
        else:
            speed_ms = 0.0
        
        # 更新历史位置
        self.position_history[track_id] = current_pos_3d
        
        return speed_ms


def process_video_metric3d(input_path: str, output_path: str,
                           show_video: bool = True,
                           conf_threshold: float = 0.25,
                           show_depth: bool = True,
                           depth_frequency: int = 10,
                           model_size: str = 'small'):
    """
    Phase 3 Metric3D处理：RAFT + Metric3D v2 + YOLOv8
    
    Args:
        input_path: 输入视频
        output_path: 输出视频
        show_video: 是否显示窗口
        conf_threshold: 检测阈值
        show_depth: 是否显示深度图
        depth_frequency: 深度估计频率（每N帧）
        model_size: Metric3D模型大小 ('small', 'large', 'giant2')
    """
    print("=" * 60)
    print("Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8")
    print("=" * 60)
    
    # 1. 初始化YOLOv8
    print("\n[1/4] Loading YOLOv8...")
    yolo_path = model_config.get_model_path('yolov8n.pt')
    model = YOLO(yolo_path)
    print(f"✅ YOLOv8 loaded from {yolo_path}")
    
    # 2. 初始化RAFT
    print("\n[2/4] Loading RAFT...")
    raft = RAFTOpticalFlow(model_type='small', device='auto')
    print("✅ RAFT loaded")
    
    # 3. 初始化Metric3D v2
    print("\n[3/4] Loading Metric3D v2...")
    depth_estimator = Metric3Dv2(model_size=model_size, device='auto')
    print("✅ Metric3D v2 loaded")
    
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
    
    # 估算相机内参（如果没有实际内参）
    intrinsics = depth_estimator.estimate_camera_intrinsics(width, height, fov_degrees=60.0)
    
    # 5. 初始化输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 6. 初始化估计器
    speed_estimator = Metric3DSpeedEstimator(fps=fps)
    
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
            depth_map_cache = depth_estimator.estimate_depth(frame, intrinsics)
        
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
                
                # 获取物体深度（米！）
                depth_meters = depth_estimator.get_object_depth(depth_map_cache, (x1, y1, x2, y2))
                
                # 计算3D速度
                speed = 0.0
                if track_id in track_positions:
                    # 使用Metric3D v2的真实深度计算3D速度
                    speed = speed_estimator.calculate_3d_speed(
                        track_id=track_id,
                        current_pos_2d=(cx, cy),
                        current_depth=depth_meters,  # 真实深度（米）
                        camera_motion_2d=camera_motion,
                        intrinsics=intrinsics
                    )
                
                track_positions[track_id] = (cx, cy)
                
                # 绘制边界框（颜色根据深度）
                # 深度越近越绿，越远越红
                if depth_meters > 0:
                    depth_color_val = min(255, int(depth_meters / 30.0 * 255))  # 30米为最远
                    color = (0, 255 - depth_color_val, depth_color_val)
                else:
                    color = (128, 128, 128)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # ✅ 优化标签格式：显示置信度+像素速度+真实距离+3D速度
                # 计算像素速度
                pixel_speed = 0
                if track_id in track_positions:
                    prev_cx, prev_cy = track_positions[track_id]
                    # 表观运动 - 摄像头运动 = 真实运动
                    real_dx = (cx - prev_cx) - camera_motion[0]
                    real_dy = (cy - prev_cy) - camera_motion[1]
                    pixel_speed = np.sqrt(real_dx**2 + real_dy**2)
                
                if speed > 0:
                    label = f"ID{track_id} {class_name} (conf:{conf:.2f}) {pixel_speed:.1f}px/f | {speed:.1f}m/s | {depth_meters:.1f}m"
                else:
                    label = f"ID{track_id} {class_name} (conf:{conf:.2f}) | {depth_meters:.1f}m"
                
                # ✅ 更美观的字体
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                thickness = 2
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w + 6, y1), color, -1)
                
                # ✅ 文字加黑色描边（更清晰）
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                           font, font_scale, (0, 0, 0), thickness + 2)  # 黑色描边
                cv2.putText(annotated_frame, label, (x1 + 3, y1 - 5),
                           font, font_scale, (255, 255, 255), thickness)  # 白色文字
        
        # 显示深度图
        if show_depth and frame_idx % depth_frequency == 1:
            depth_vis = depth_estimator.visualize_depth(depth_map_cache, max_depth=50.0)
            depth_small = cv2.resize(depth_vis, (width // 4, height // 4))
            annotated_frame[10:10+height//4, width-width//4-10:width-10] = depth_small
            cv2.putText(annotated_frame, "Metric3D (meters)", (width-width//4-10, height//4+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # ✅ 优化信息面板颜色
        panel_color = (0, 200, 200)  # 深青色
        cv2.putText(annotated_frame, f"Frame: {frame_idx}/{total_frames} | Metric3D v2 + RAFT",
                   (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, panel_color, 2)
        cv2.putText(annotated_frame, f"Camera: dx={camera_motion[0]:.1f} dy={camera_motion[1]:.1f}",
                   (10, height - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 2)
        
        # 写入和显示
        out.write(annotated_frame)
        if show_video:
            cv2.imshow('Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_idx % 30 == 0:
            print(f"Progress: {frame_idx/total_frames*100:.1f}% ({frame_idx}/{total_frames})", end='\r')
    
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Phase 3 Metric3D Processing Complete!")
    print(f"📹 Output: {output_path}")
    print(f"🎯 Processed {frame_idx} frames")
    print(f"{'=' * 60}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8')
    parser.add_argument('--input', type=str, default='input/test_video.mp4')
    parser.add_argument('--output', type=str, default='output/phase3_metric3d_output.mp4')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument('--depth-freq', type=int, default=10,
                       help='Depth estimation frequency (every N frames)')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'large', 'giant2'],
                       help='Metric3D model size')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    success = process_video_metric3d(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_show,
        conf_threshold=args.conf,
        show_depth=not args.no_depth,
        depth_frequency=args.depth_freq,
        model_size=args.model_size
    )
    
    if success:
        print("\n✅ Done!")
    else:
        print("\n❌ Failed!")
        sys.exit(1)
