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
import csv
from pathlib import Path
from datetime import datetime
from collections import deque, defaultdict

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
    
    def __init__(self, fps: float = 30.0, display_interval: int = 8, window_size: int = 7):
        """
        初始化

        Args:
            fps: 视频帧率
            window_size: 滑动窗口帧数（默认7，约0.23s@30fps，拉长测量基线降低噪声）
            display_interval: 视频标签每N帧刷新一次显示速度（默认8，减少视觉跳动）
        """
        self.fps = fps
        self.window_size = window_size
        self.position_window = {}   # 每个track的3D位置滑动窗口 {id: deque}
        self.speed_history = {}     # 速度历史（轻EMA用）
        self.depth_history = {}     # 每个物体的平滑深度
        self.track_frame_count = {} # 每个track被观测的帧数
        self.depth_alpha = 0.15     # 深度EMA系数（平滑深度台阶跳变）
        self.ema_alpha = 0.4        # 轻EMA速度系数（窗口已滤噪，无需重度平滑）
        self.display_delay = max(3, window_size - 1)  # 满窗口后再输出，避免短基线噪声进入CSV
        self.display_interval = display_interval
        self.display_speed_history = {}
        self.display_counter = {}
        self.last_valid_frame = {}  # 记录每个track最后有效深度帧号，用于遮挡检测（depth=0不计入）

        print(f"[Metric3DEstimator] FPS={fps}, window={window_size}f "
              f"({window_size/fps*1000:.0f}ms), display_interval={display_interval}f")
    
    def calculate_3d_speed(self, track_id: int,
                          current_pos_2d: tuple,
                          current_depth: float,
                          camera_motion_2d: tuple,
                          intrinsics: dict,
                          frame_idx: int = 0) -> float:
        """
        计算完整3D速度（m/s），包含 XYZ 三个方向。

        关键：对每个物体的深度单独做 EMA 平滑，消除深度缓存
        每N帧更新时产生的突变，同时保留 Z 方向（靠近/远离）信息。

        Args:
            track_id: 追踪ID
            current_pos_2d: 当前2D像素位置 (cx, cy)
            current_depth: Metric3D 估计的当前深度（米）
            camera_motion_2d: RAFT 摄像头运动补偿 (dx, dy) 像素
            intrinsics: 相机内参 {fx, fy, cx, cy}

        Returns:
            speed_ms: 平滑后的3D速度 (m/s)
        """
        cx, cy = current_pos_2d
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx_cam = intrinsics['cx']
        cy_cam = intrinsics['cy']

        # ── 深度无效防护：depth=0 不更新任何状态，间隔检测仅基于有效帧 ──
        if current_depth <= 0:
            return 0.0

        # ── 深度 EMA 平滑（缓解每N帧深度更新时的台阶跳变）────
        if track_id in self.depth_history:
            smooth_depth = (self.depth_alpha * current_depth
                            + (1 - self.depth_alpha) * self.depth_history[track_id])
        else:
            smooth_depth = current_depth
        self.depth_history[track_id] = smooth_depth

        # ── 摄像头运动补偿 → 3D 坐标 ─────────────────────────
        comp_cx = cx - camera_motion_2d[0]
        comp_cy = cy - camera_motion_2d[1]
        X = (comp_cx - cx_cam) * smooth_depth / fx
        Y = (comp_cy - cy_cam) * smooth_depth / fy
        Z = smooth_depth

        # ── 遮挡检测：有效帧间隔 > 1 → 清空滑动窗口 ─────────
        if track_id in self.last_valid_frame:
            if frame_idx - self.last_valid_frame[track_id] > 1:
                if track_id in self.position_window:
                    self.position_window[track_id].clear()
                self.speed_history.pop(track_id, None)
                self.track_frame_count[track_id] = 0
                self.display_speed_history[track_id] = 0.0
                self.display_counter[track_id] = 0
        self.last_valid_frame[track_id] = frame_idx

        # ── 更新帧计数 + 位置滑动窗口 ────────────────────────
        self.track_frame_count[track_id] = self.track_frame_count.get(track_id, 0) + 1
        frame_count = self.track_frame_count[track_id]

        if track_id not in self.position_window:
            self.position_window[track_id] = deque(maxlen=self.window_size)
        self.position_window[track_id].append((X, Y, Z))

        # ── 滑动窗口速度：最旧→最新位移 / 窗口帧数 ──────────
        # 测量基线 = window_size/fps 秒，比逐帧差分噪声低约 √window_size 倍
        # 无冷启动从0爬升问题：窗口积累到2帧即可给出有效速度
        window = self.position_window[track_id]
        if len(window) >= 2:
            oldest = window[0]
            n_frames = len(window) - 1
            dx = X - oldest[0]
            dy = Y - oldest[1]
            dz = Z - oldest[2]
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            raw_speed = distance_3d * self.fps / n_frames

            # 轻EMA：窗口已滤大部分噪声，α=0.4 兼顾平滑与快速响应
            if track_id in self.speed_history:
                speed_ms = (self.ema_alpha * raw_speed
                            + (1 - self.ema_alpha) * self.speed_history[track_id])
            else:
                speed_ms = raw_speed
            self.speed_history[track_id] = speed_ms
        else:
            speed_ms = 0.0

        # ── 显示速度：每 display_interval 帧刷新一次 ─────────
        if track_id not in self.display_counter:
            self.display_counter[track_id] = 0
            self.display_speed_history[track_id] = 0.0
        self.display_counter[track_id] += 1
        if self.display_counter[track_id] >= self.display_interval:
            self.display_speed_history[track_id] = speed_ms
            self.display_counter[track_id] = 0

        # 前 display_delay 帧不对外输出速度（隐藏窗口积累期噪声）
        if frame_count <= self.display_delay:
            return 0.0
        return speed_ms

    def get_display_speed(self, track_id: int) -> float:
        """返回视频标签用的稳定显示速度（每 display_interval 帧刷新一次）"""
        if self.track_frame_count.get(track_id, 0) <= self.display_delay:
            return 0.0
        return self.display_speed_history.get(track_id, 0.0)


def process_video_metric3d(input_path: str, output_path: str,
                           show_video: bool = True,
                           conf_threshold: float = 0.25,
                           show_depth: bool = True,
                           depth_frequency: int = 5,
                           model_size: str = 'small',
                           fov_degrees: float = 60.0):
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
        fov_degrees: 相机水平视场角（度），由等效全画幅焦段换算而来
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
    
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print("⚠️  FPS not detected from container, defaulting to 25.0")
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video: {width}x{height} @ {fps}FPS, {total_frames} frames")
    
    # 估算相机内参（如果没有实际内参）
    intrinsics = depth_estimator.estimate_camera_intrinsics(width, height, fov_degrees=fov_degrees)
    print(f"[Intrinsics] FOV={fov_degrees}°  fx={intrinsics['fx']:.1f}  fy={intrinsics['fy']:.1f}")
    
    # 5. 初始化输出
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _op = Path(output_path)
    output_path = str(_op.with_name(_op.stem + '_' + run_ts + _op.suffix))
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
    csv_rows = []
    crops_dir = str(Path(output_path).with_suffix('')) + '_crops'
    os.makedirs(crops_dir, exist_ok=True)
    first_crop_paths = {}
    
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
                
                # 计算3D速度（滑动窗口内部处理首帧，无需外部守卫）
                prev_pos_2d = track_positions.get(track_id)
                speed = speed_estimator.calculate_3d_speed(
                    track_id=track_id,
                    current_pos_2d=(cx, cy),
                    current_depth=depth_meters,  # 真实深度（米）
                    camera_motion_2d=camera_motion,
                    intrinsics=intrinsics,
                    frame_idx=frame_idx
                )
                
                if track_id not in track_positions:
                    pad = 8
                    crop = frame[max(0, y1-pad):min(height, y2+pad),
                                 max(0, x1-pad):min(width, x2+pad)]
                    crop_name = f"track_{track_id}_{class_name}.jpg"
                    crop_path = os.path.join(crops_dir, crop_name)
                    cv2.imwrite(crop_path, crop)
                    first_crop_paths[track_id] = crop_path

                track_positions[track_id] = (cx, cy)
                display_speed = speed_estimator.get_display_speed(track_id)
                
                smooth_depth = speed_estimator.depth_history.get(track_id, depth_meters)
                csv_rows.append({
                    'frame': frame_idx,
                    'track_id': int(track_id),
                    'class_name': class_name,
                    'confidence': round(float(conf), 4),
                    'cx': round(float(cx), 1),
                    'cy': round(float(cy), 1),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'camera_dx': round(float(camera_motion[0]), 3),
                    'camera_dy': round(float(camera_motion[1]), 3),
                    'depth_meters': round(float(smooth_depth), 3),
                    'speed_ms': round(float(speed), 3),
                })
                
                # 绘制边界框（颜色根据深度）
                # 深度越近越绿，越远越红
                if depth_meters > 0:
                    depth_color_val = min(255, int(depth_meters / 30.0 * 255))  # 30米为最远
                    color = (0, 255 - depth_color_val, depth_color_val)
                else:
                    color = (128, 128, 128)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 计算像素速度（修复：使用保存的上一帧位置，而非已更新的当前位置）
                pixel_speed = 0.0
                if prev_pos_2d is not None:
                    prev_cx, prev_cy = prev_pos_2d
                    real_dx = (cx - prev_cx) - camera_motion[0]
                    real_dy = (cy - prev_cy) - camera_motion[1]
                    pixel_speed = np.sqrt(real_dx**2 + real_dy**2)
                
                if display_speed > 0:
                    label = f"ID{track_id} {class_name} (conf:{conf:.2f}) {pixel_speed:.1f}px/f | {display_speed:.1f}m/s | {smooth_depth:.1f}m"
                else:
                    label = f"ID{track_id} {class_name} (conf:{conf:.2f}) | {smooth_depth:.1f}m"
                
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
            if total_frames > 0:
                print(f"Progress: {frame_idx/total_frames*100:.1f}% ({frame_idx}/{total_frames})", end='\r')
            else:
                print(f"Progress: {frame_idx} frames processed", end='\r')
    
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    
    if csv_rows:
        base_csv = str(Path(output_path).with_suffix(''))

        # ── CSV 1: 逐帧数据（每行=一帧×一辆车）──────────────────
        frames_csv_path = base_csv + '_frames.csv'
        with open(frames_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[CSV] Per-frame: {frames_csv_path} ({len(csv_rows)} rows)")

        # ── CSV 2: 按车辆汇总（每行=一辆车的统计）────────────────
        vehicle_data = defaultdict(list)
        for row in csv_rows:
            vehicle_data[row['track_id']].append(row)

        object_rows = []
        for tid, rows in sorted(vehicle_data.items()):
            moving = [r for r in rows if r['speed_ms'] > 0.28]
            speeds_ms = [r['speed_ms'] for r in moving] if moving else [0.0]
            depths = [r['depth_meters'] for r in rows if r['depth_meters'] > 0]
            object_rows.append({
                'track_id': tid,
                'class_name': rows[0]['class_name'],
                'first_time_s': round((rows[0]['frame'] - 1) / fps, 2),
                'last_time_s': round((rows[-1]['frame'] - 1) / fps, 2),
                'duration_s': round((rows[-1]['frame'] - rows[0]['frame'] + 1) / fps, 2),
                'avg_speed_ms': round(sum(speeds_ms) / len(speeds_ms), 3),
                'max_speed_ms': round(max(speeds_ms), 3),
                'min_speed_ms': round(min(speeds_ms), 3),
                'avg_depth_m': round(sum(depths) / len(depths), 2) if depths else 0.0,
                'status': 'moving' if max(speeds_ms) > 0.56 else 'slow/stationary',
                'first_crop_path': first_crop_paths.get(tid, ''),
            })

        objects_csv_path = base_csv + '_objects.csv'
        with open(objects_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=object_rows[0].keys())
            writer.writeheader()
            writer.writerows(object_rows)
        print(f"[CSV] Per-object: {objects_csv_path} ({len(object_rows)} objects)")
    
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
    parser.add_argument('--depth-freq', type=int, default=5,
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
