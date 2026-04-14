# -*- coding: utf-8 -*-
"""
Mode 6: Ego-Vehicle Speed Estimation (新版设计方案)
全图 dy 光流 + Metric3D 绝对深度 + 双帧一致性 YOLO 掩码 + 光流掩码

核心公式:
    forward_speed = dy * Z * fps / (y - cy)

适用范围：全场景（行车记录仪 / 手持步行 / 无人机俯拍 / 室内）
"""
import os
import sys
import cv2
import csv
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime

from src.enhance_video import get_video_writer, _safe_cv2_write

# ⚠️ 必须先导入model_config设置环境变量
try:
    from . import model_config
except ImportError:
    import model_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.optical_flow_raft import RAFTOpticalFlow
from src.depth_estimation_metric3d import Metric3Dv2


# =============================================================================
# 可独立运动目标类别（仅这些类别参与 YOLO 双帧交集掩码）
# =============================================================================
MOVABLE_CLASSES = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle',
    'person', 'pedestrian', 'rider',
    'train', 'boat', 'airplane',
    'dog', 'cat', 'horse', 'bird', 'cow', 'sheep',
}


# =============================================================================
# CSV 工具函数
# =============================================================================

def write_csv_with_header(csv_path: str, fieldnames: list, rows: list,
                          header_lines: list = None):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if header_lines:
            for line in header_lines:
                f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# =============================================================================
# 掩码工具函数
# =============================================================================

def _boxes_overlap(box_a, box_b, iou_thresh=0.3):
    """检测两 bbox 是否重叠（IoU >= thresh）"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return False

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return (inter_area / union_area) >= iou_thresh if union_area > 0 else False


def _merge_boxes(box_a, box_b):
    """合并两个 bbox（取并集边界）"""
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    return (x1, y1, x2, y2)


# =============================================================================
# EgoSpeedEstimator（新版）
# =============================================================================

class EgoSpeedEstimator:
    """
    自车速度估计器 — 新版设计方案

    关键设计：
    - 正确速度公式：dy × Z × fps / (y - cy)
    - 两步掩码：YOLO 双帧交集掩码（去动态目标）+ 光流掩码（去无光流区域）
    - 全图采样（不再限于底部区域）
    - 质量分级：OK / WARN / TURN
    """

    def __init__(self, fx, fy, cy, fps,
                 n_samples=800,
                 depth_min=1.0, depth_max=100.0,
                 y_min_offset=20,
                 flow_min=0.05,
                 iou_thresh=0.3, box_area_thresh=500.0, box_conf_thresh=0.5,
                 warmup_frames=30, warmup_alpha=0.5, steady_alpha=0.2,
                 display_interval=15, display_delay=5):
        self.fx = fx
        self.fy = fy
        self.cy = cy            # 主点 y 坐标（像素）
        self.fps = fps

        self.n_samples = n_samples
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.y_min_offset = y_min_offset
        self.flow_min = flow_min

        # YOLO 掩码参数
        self.iou_thresh = iou_thresh
        self.box_area_thresh = box_area_thresh
        self.box_conf_thresh = box_conf_thresh

        # EMA 参数
        self.warmup_frames = warmup_frames
        self.warmup_alpha = warmup_alpha
        self.steady_alpha = steady_alpha
        self.display_interval = display_interval
        self.display_delay = display_delay

        self.current_speed_ms = 0.0
        self.speed_history = deque(maxlen=120)
        self.frame_count = 0
        self.display_speed_ms = 0.0
        self._display_counter = display_interval - 1

        # 当前帧诊断信息
        self.last_quality = "OK"
        self.last_flow_valid_rate = 0.0
        self.last_valid_pixels = 0

        # YOLO 检测结果（双帧缓存）
        self._prev_detections = []   # [{'bbox': (x1,y1,x2,y2), 'class_name': str, 'confidence': float}, ...]
        self._curr_detections = []

        print(f"[EgoSpeedEstimator] fps={fps:.1f}  "
              f"cy={cy:.1f}  y_min_offset={y_min_offset}  "
              f"warmup={warmup_frames}f α={warmup_alpha}→{steady_alpha}  "
              f"flow_min={flow_min}")

    # ----------------------------------------------------------------- 公共 API

    def set_detections(self, detections):
        """
        注入当前帧的 YOLO 检测结果（外部调用者负责逐帧传入）。
        detections: [{'bbox': (x1,y1,x2,y2), 'class_name': str, 'confidence': float}, ...]
        """
        self._prev_detections = self._curr_detections
        self._curr_detections = [
            d for d in detections
            if d.get('class_name', '').lower() in MOVABLE_CLASSES
        ]

    def estimate_speed(self, flow, depth_map):
        """
        核心速度估计函数。

        Args:
            flow:       RAFT 光流场 [H, W, 2]，prev→curr
            depth_map:  Metric3D 深度图 [H, W]，单位米

        Returns:
            current_speed_ms: 估算的前向速度（m/s）
        """
        H, W = flow.shape[:2]

        # ── Step 1：双帧交集 YOLO 掩码 ──────────────────────────────────────
        yolo_mask = self._build_yolo_mask(H, W)

        # ── Step 2：光流掩码（去除无光流的纯色/无纹理区域）─────────────────
        texture_mask = self._build_texture_mask(flow)

        # 合并两步掩码：True = 该像素可参与速度计算
        valid = yolo_mask & texture_mask

        # ── Step 3：深度 & 主点掩码 ─────────────────────────────────────────
        y_rel = np.arange(H, dtype=np.float32).reshape(-1, 1) - self.cy  # (H,1) shape

        depth_ok = (depth_map > self.depth_min) & (depth_map < self.depth_max)
        y_ok = (np.abs(y_rel) > self.y_min_offset) & (y_rel > 0)  # |y-cy|>y_min AND y>cy
        valid &= depth_ok & y_ok

        valid_idx = np.where(valid)
        self.last_valid_pixels = int(valid.sum())
        self.last_flow_valid_rate = self.last_valid_pixels / (H * W)

        if len(valid_idx[0]) < 20:
            self._advance_frame()
            return self.current_speed_ms

        # ── Step 4：均匀采样 ─────────────────────────────────────────────────
        n = len(valid_idx[0])
        n_sample = min(self.n_samples, n)
        sel = np.linspace(0, n - 1, n_sample, dtype=int)
        rows, cols = valid_idx[0][sel], valid_idx[1][sel]

        dy = flow[rows, cols, 1]                 # RAFT 垂直光流（像素/帧）
        Z = depth_map[rows, cols]                 # 绝对深度（米）
        y_minus_cy = (rows - self.cy).astype(float)  # 避免整数除法

        # ── Step 5：正确速度公式 ────────────────────────────────────────────
        # dy ≈ (y - cy) / Z × vz / fps  →  vz = dy × Z × fps / (y - cy)
        speeds = dy * Z * self.fps / y_minus_cy   # 逐像素速度（m/s）

        # ── Step 6：质量检查（转弯检测）────────────────────────────────────
        dx = flow[rows, cols, 0]
        dx_median = float(np.median(dx))
        self.last_quality = "TURN" if abs(dx_median) > 2.0 else "OK"

        # ── Step 7：中位数速度 ─────────────────────────────────────────────
        raw_speed = float(np.median(speeds))

        # 异常值钳制
        alpha = self.warmup_alpha if self.frame_count < self.warmup_frames else self.steady_alpha
        if self.frame_count >= self.warmup_frames and abs(self.current_speed_ms) > 0.5:
            cap_val = max(abs(self.current_speed_ms) * 5.0, 40.0)
            raw_speed = max(-cap_val, min(raw_speed, cap_val))

        self.current_speed_ms = alpha * raw_speed + (1.0 - alpha) * self.current_speed_ms
        self.speed_history.append(self.current_speed_ms)
        self._advance_frame()

        return self.current_speed_ms

    def get_display_speed_ms(self):
        """稳定显示速度（每 N 帧更新一次，冷启动延迟）"""
        if self.frame_count < self.display_delay:
            return 0.0
        return self.display_speed_ms

    # ----------------------------------------------------------------- 私有方法

    def _build_yolo_mask(self, H, W):
        """
        双帧交集 YOLO 掩码：
        - 仅掩码可独立运动目标（车/人/自行车/摩托）
        - 静态背景不掩码（参与速度计算）
        - 仅当同一物体在 prev/curr 两帧中均出现时才掩码（防跳帧）
        """
        mask = np.ones((H, W), dtype=bool)

        if not self._prev_detections or not self._curr_detections:
            return mask

        for box_curr in self._curr_detections:
            bbox_c = box_curr['bbox']
            conf_c = box_curr.get('confidence', 1.0)
            if conf_c < self.box_conf_thresh:
                continue

            for box_prev in self._prev_detections:
                conf_p = box_prev.get('confidence', 1.0)
                if conf_p < self.box_conf_thresh:
                    continue

                if _boxes_overlap(bbox_c, bbox_prev['bbox'], self.iou_thresh):
                    merged = _merge_boxes(bbox_c, bbox_prev['bbox'])
                    x1, y1, x2, y2 = [int(v) for v in merged]
                    area = (x2 - x1) * (y2 - y1)
                    if area >= self.box_area_thresh:
                        x1 = np.clip(x1, 0, W)
                        y1 = np.clip(y1, 0, H)
                        x2 = np.clip(x2, 0, W)
                        y2 = np.clip(y2, 0, H)
                        mask[y1:y2, x1:x2] = False

        return mask

    def _build_texture_mask(self, flow, prev_frame):
        """
        光流掩码：去除无光流的静态区域（纯色天空/平滑墙面/极低速静止）。
        对全图操作，不限于路面。
        flow_min=0.05：全场景通用阈值
          - 行车 30m/s + 远距离 100m → 光流 ≈ 0.5 → 完全覆盖
          - 步行 1.5m/s + 远距离 50m → 光流 ≈ 0.05 → 刚好覆盖
          - RAFT 噪声底噪 ≈ 0.02~0.05，低于此值无法区分信号与噪声
        """
        flow_mag = np.linalg.norm(flow, axis=2)   # 光流幅值
        return flow_mag > self.flow_min

    def _advance_frame(self):
        self.frame_count += 1
        self._display_counter += 1
        if self._display_counter >= self.display_interval:
            self.display_speed_ms = self.current_speed_ms
            self._display_counter = 0


# =============================================================================
# OSD 可视化
# =============================================================================

def _draw_speed_panel(frame, speed_ms, quality, flow_valid_rate,
                      speed_history, frame_idx):
    """
    绘制速度信息面板 + 质量指示。

    布局：
      ┌─────────────────────────────────────────┐
      │ EGO VEHICLE SPEED           frame N    │
      │  XX.XX m/s        [OK] / [TURN]        │
      │  速度条图（近30秒）                      │
      │  ⚠️ 低有效像素（<50%）                  │
      └─────────────────────────────────────────┘
    """
    H, W = frame.shape[:2]
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 360, 140

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # 标题
    cv2.putText(frame, "EGO VEHICLE SPEED",
                (panel_x + 10, panel_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    # 速度数字
    if speed_ms < 0:
        speed_color = (0, 60, 255)
        speed_label = f"{speed_ms:+7.2f} m/s  [REVERSE]"
    else:
        if quality == "TURN":
            speed_color = (0, 220, 255)      # 黄色 = 转弯
        elif speed_ms < 16.7:
            speed_color = (0, 255, 80)        # 绿色 = 正常
        elif speed_ms < 27.8:
            speed_color = (0, 165, 255)      # 橙色 = 较快
        else:
            speed_color = (0, 60, 255)        # 红色 = 很快
        speed_label = f"{speed_ms:+7.2f} m/s"

    cv2.putText(frame, speed_label,
                (panel_x + 10, panel_y + 62),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, speed_color, 2)

    # 质量标签
    q_text = f"[{quality}]"
    q_color = (0, 220, 255) if quality == "TURN" else (0, 200, 100)
    cv2.putText(frame, q_text,
                (panel_x + 250, panel_y + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_color, 1)

    if quality == "TURN":
        cv2.putText(frame, "  (turning - use with caution)",
                    (panel_x + 10, panel_y + 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 220), 1)

    # 低有效像素警告
    if flow_valid_rate < 0.5:
        warn_text = f"  low valid pixels: {flow_valid_rate:.0%}"
        cv2.putText(frame, warn_text,
                    (panel_x + 10, panel_y + (86 if quality == "TURN" else 62)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 80, 255), 1)

    # 速度历史条形图（右侧）
    if len(speed_history) > 2:
        gx = W - 200
        gy = 10
        gw, gh = 190, 80
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

        history = list(speed_history)
        max_v = max(max(abs(s) for s in history), 1.0)
        pts = []
        for i, s in enumerate(history):
            px = gx + int(i / max(len(history) - 1, 1) * gw)
            py = gy + gh - int(abs(s) / max_v * gh)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0, 220, 120), 1)
        cv2.putText(frame, f"max {max_v:.1f} m/s",
                    (gx + 5, gy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    # 帧号
    cv2.putText(frame, f"frame {frame_idx}",
                (W - 110, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    return frame


# =============================================================================
# 主处理函数
# =============================================================================

def process_video_ego_speed(input_path, output_path, show_video=True,
                            fov_degrees=70.0, depth_frequency=5,
                            yolo_model='yolov8n.pt', yolo_conf=0.25,
                            model_size='small'):
    """
    Mode 6 主函数：自车速度估计（新版本）

    Args:
        input_path:        输入视频路径
        output_path:       输出视频路径
        show_video:        是否显示实时窗口
        fov_degrees:       水平视场角（度），手机约 70°
        depth_frequency:   每 N 帧重算一次深度
        yolo_model:        YOLO 模型名称（用于双帧交集掩码）
        yolo_conf:         YOLO 置信度阈值
        model_size:        Metric3D 模型大小 'small'/'large'
    """
    print("=" * 60)
    print("Mode 6: Ego Speed  |  RAFT + Metric3D v2 + YOLO 双帧掩码  (NEW)")
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print("\u26a0\ufe0f  FPS not detected, defaulting to 30.0")
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height}  {fps:.1f}fps  {total} frames")

    # 主点 y 坐标（cy ≈ H/2，对手机摄像头）
    cy = height / 2.0

    print("\n[1/3] Loading RAFT...")
    raft = RAFTOpticalFlow()
    print("✅ RAFT loaded")

    print("\n[2/3] Loading Metric3D v2...")
    depth_estimator = Metric3Dv2(model_size=model_size)
    if not depth_estimator.is_available():
        print("[ERROR] Metric3D v2 failed to load")
        cap.release()
        return False
    print("✅ Metric3D v2 loaded")

    intrinsics = depth_estimator.estimate_camera_intrinsics(width, height, fov_degrees=fov_degrees)
    fx, fy = intrinsics['fx'], intrinsics['fy']
    print(f"[Intrinsics] FOV={fov_degrees}°  fx={fx:.1f}  fy={fy:.1f}  cy={cy:.1f}")

    print("\n[3/3] Loading YOLOv8...")
    yolo_available = False
    try:
        from ultralytics import YOLO as YOLOModel
        yolo_path = model_config.get_model_path(yolo_model)
        yolo = YOLOModel(yolo_path)
        yolo_available = True
        print(f"✅ YOLOv8 loaded ({yolo_model})")
    except Exception as e:
        print(f"[WARN] YOLOv8 not available ({e})，双帧掩码将跳过")
        yolo = None

    speed_est = EgoSpeedEstimator(
        fx=fx, fy=fy, cy=cy, fps=fps,
        warmup_frames=30, warmup_alpha=0.5, steady_alpha=0.2,
        display_interval=15, display_delay=5,
    )

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.abspath(
        str(Path(output_path).with_name(
            Path(output_path).stem + '_' + run_ts + Path(output_path).suffix))
    )
    print(f"[Output] 视频将写入: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = get_video_writer(output_path, fps, width, height)
    frames_written = 0

    csv_rows = []
    frame_idx = 0
    prev_frame = None
    depth_cache = None

    print(f"\n[Processing] {total} frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 深度估计（每 N 帧）
            if frame_idx % depth_frequency == 1 or depth_cache is None:
                depth_cache = depth_estimator.estimate_depth(frame, intrinsics)

            speed_ms = 0.0
            quality = "OK"
            flow_valid_rate = 0.0

            if prev_frame is not None and depth_cache is not None:
                # YOLO 检测（仅动态目标）
                detections = []
                if yolo_available and yolo is not None:
                    results = yolo.predict(
                        prev_frame, conf=yolo_conf,
                        verbose=False, device='cpu'
                    )
                    if results and results[0].boxes is not None:
                        for det in results[0].boxes:
                            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                            cls_id = int(det.cls[0].cpu().numpy())
                            cls_name = yolo.names[cls_id] if cls_id < len(yolo.names) else ''
                            conf = float(det.conf[0].cpu().numpy())
                            detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'class_name': cls_name,
                                'confidence': conf,
                            })
                speed_est.set_detections(detections)

                # RAFT 光流
                flow = raft.compute_flow(prev_frame, frame,
                                         output_height=height, output_width=width)

                speed_ms = speed_est.estimate_speed(flow, depth_cache)
                quality = speed_est.last_quality  # OK 或 TURN
                flow_valid_rate = speed_est.last_flow_valid_rate
                # WARN 覆盖：有效像素不足时降级质量（转弯优先级更高）
                if quality != "TURN" and flow_valid_rate < 0.5:
                    quality = "WARN"

                # 首帧诊断
                if frame_idx == 1:
                    _print_first_frame_diag(flow, depth_cache, speed_ms, fy, fps)

            display_speed = speed_est.get_display_speed_ms()
            vis = _draw_speed_panel(
                frame.copy(), display_speed, quality, flow_valid_rate,
                speed_est.speed_history, frame_idx
            )

            if writer:
                frame_out = np.ascontiguousarray(vis)
                if _safe_cv2_write(writer, frame_out):
                    frames_written += 1

            if show_video:
                cv2.imshow("Mode 6 - Ego Speed (New)", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            csv_rows.append({
                'frame_idx':       frame_idx,
                'timestamp_s':    round(frame_idx / fps, 4),
                'ego_speed_ms':   round(speed_ms, 4),
                'quality_flag':   quality,
                'flow_valid_rate': round(flow_valid_rate, 4),
                'valid_pixels':   speed_est.last_valid_pixels,
                'total_pixels':   height * width,
            })

            prev_frame = frame.copy()
            frame_idx += 1

            if frame_idx > 0 and frame_idx % 50 == 0:
                print(f"  [{frame_idx}/{total}]  {speed_ms:.2f} m/s  [{quality}]  "
                      f"valid={flow_valid_rate:.0%}")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()

    if frame_idx == 0:
        print("[ERROR] 未读取到任何帧")
        return False

    # ── 写 CSV ──────────────────────────────────────────────────────────────
    _write_csv_outputs(output_path, csv_rows, fps)

    # ── 校验视频文件 ────────────────────────────────────────────────────────
    if frames_written == 0:
        print("[ERROR] 未成功写入任何视频帧，请检查 OpenCV VideoWriter / FFmpeg。")
        return False
    if not os.path.isfile(output_path):
        print(f"[ERROR] 预期视频文件不存在: {output_path}")
        return False
    sz = os.path.getsize(output_path)
    if sz < 1024:
        print(f"[ERROR] 输出视频异常过小 ({sz} bytes)，编码可能失败。")
        return False

    print(f"\n[OK] Done. {frame_idx} frames processed (video frames written: {frames_written})")
    return output_path


# =============================================================================
# CSV 输出辅助函数
# =============================================================================

def _write_csv_outputs(output_path, csv_rows, fps):
    """按帧级和秒级两个粒度写 CSV 文件"""
    if not csv_rows:
        return

    base_csv = str(Path(output_path).with_suffix(''))
    fps_safe = fps if fps > 0 else 30.0

    # ── 帧级 CSV ───────────────────────────────────────────────────────────
    frames_csv_path = base_csv + '_frames.csv'
    write_csv_with_header(
        frames_csv_path,
        fieldnames=list(csv_rows[0].keys()),
        rows=csv_rows,
        header_lines=[
            "mode: 6 | algorithm: EgoSpeed v2 (full-image dy + Metric3D + dual-frame YOLO mask + flow mask)",
            "core formula: forward_speed = dy * Z * fps / (y - cy)",
            "unit: ego_speed_ms = m/s",
            "⚠️  This mode estimates SELF vehicle speed, not external object speed",
            "quality_flag: OK/WARN/TURN = normal/low_valid_pixels/turning (WARN: flow_valid_rate < 0.5; TURN: |dx_median| > 2)",
        ]
    )
    print(f"[CSV] Frames: {frames_csv_path} ({len(csv_rows)} rows)")

    # ── 秒级 CSV ────────────────────────────────────────────────────────────
    second_groups = defaultdict(list)
    for row in csv_rows:
        second_key = int(row['frame_idx'] / fps_safe)
        second_groups[second_key].append(row)

    stats_rows = []
    cumulative_disp = 0.0
    for sec_idx in sorted(second_groups.keys()):
        sec_rows = second_groups[sec_idx]
        sec_speeds = [r['ego_speed_ms'] for r in sec_rows]
        duration_s = len(sec_rows) / fps_safe
        avg_spd = sum(sec_speeds) / len(sec_speeds)
        disp = avg_spd * duration_s
        cumulative_disp += disp
        stats_rows.append({
            'second':                      sec_idx,
            'start_frame':                 sec_rows[0]['frame_idx'],
            'end_frame':                    sec_rows[-1]['frame_idx'],
            'avg_speed_ms':                round(avg_spd, 3),
            'max_speed_ms':                round(max(sec_speeds), 3),
            'min_speed_ms':                round(min(sec_speeds), 3),
            'displacement_m':              round(disp, 2),
            'cumulative_displacement_m':   round(cumulative_disp, 2),
        })

    # SUMMARY 行
    all_speeds = [r['ego_speed_ms'] for r in csv_rows]
    total_duration_s = len(csv_rows) / fps_safe
    overall_avg = cumulative_disp / total_duration_s if total_duration_s > 0 else 0.0
    stats_rows.append({
        'second':                      'SUMMARY',
        'start_frame':                 csv_rows[0]['frame_idx'],
        'end_frame':                   csv_rows[-1]['frame_idx'],
        'avg_speed_ms':                round(overall_avg, 3),
        'max_speed_ms':                round(max(all_speeds), 3),
        'min_speed_ms':                round(min(all_speeds), 3),
        'displacement_m':              round(cumulative_disp, 2),
        'cumulative_displacement_m':   round(cumulative_disp, 2),
    })

    stats_csv_path = base_csv + '_stats.csv'
    write_csv_with_header(
        stats_csv_path,
        fieldnames=list(stats_rows[0].keys()),
        rows=stats_rows,
        header_lines=[
            "mode: 6 | algorithm: EgoSpeed v2",
            "unit: avg_speed_ms = m/s, max/min in m/s; displacement in meters",
            "⚠️  SUMMARY row: overall statistics across all frames",
        ]
    )
    n_secs = len(stats_rows) - 1
    print(f"[CSV] Stats: {stats_csv_path}  ({n_secs}s + summary  |  total disp {cumulative_disp:.1f} m)")


# =============================================================================
# 首帧诊断
# =============================================================================

def _print_first_frame_diag(flow, depth_cache, speed_ms, fy, fps):
    """打印首帧诊断信息，帮助判断光流是否正常"""
    H, W = flow.shape[:2]
    cy = H / 2.0

    # 全图有效区域（中景，y > cy 即地平线以下）
    y_rel_diag = np.arange(H, dtype=float).reshape(-1, 1) - cy
    valid = (depth_cache > 1.0) & (depth_cache < 100.0) & \
            (np.abs(y_rel_diag) > 20) & (y_rel_diag > 0)
    valid_idx = np.where(valid)
    if len(valid_idx[0]) < 5:
        print("\n[RAFT DIAG] 首帧：有效像素过少，跳过诊断")
        return

    n_sample = min(400, len(valid_idx[0]))
    sel = np.linspace(0, len(valid_idx[0]) - 1, n_sample, dtype=int)
    rows, cols = valid_idx[0][sel], valid_idx[1][sel]

    dy_vals = flow[rows, cols, 1]
    Z_vals = depth_cache[rows, cols]
    dx_vals = flow[rows, cols, 0]
    y_minus_cy = rows - cy

    # 新公式估算
    speeds_new = dy_vals * Z_vals * fps / y_minus_cy
    med_speed_new = float(np.median(speeds_new))

    print(f"\n[RAFT DIAG] 首帧 raw 指标:")
    print(f"  帧尺寸: {H}x{W}")
    print(f"  有效像素: {len(valid_idx[0])}")
    print(f"  dy 中位数: {float(np.median(dy_vals)):.3f} pix/frame")
    print(f"  dx 中位数: {float(np.median(dx_vals)):.3f} pix/frame  "
          f"({'⚠️ TURN' if abs(float(np.median(dx_vals))) > 2 else '正常'})")
    print(f"  Z 中位数: {float(np.median(Z_vals)):.2f} m")
    print(f"  |y-cy| 中位数: {float(np.median(np.abs(y_minus_cy))):.1f} px")
    print(f"  → 新公式估算速度: {med_speed_new:.1f} m/s")
    print(f"  → 最终 EMA 速度:   {speed_ms:.1f} m/s")


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Mode 6: Ego Speed Estimation (New)')
    parser.add_argument('--input', '-i', required=True, help='输入视频文件')
    parser.add_argument('--output', '-o', default='output/output_ego_speed.mp4',
                        help='输出视频文件')
    parser.add_argument('--no-display', action='store_true', help='不显示实时窗口')
    parser.add_argument('--fov', type=float, default=70.0,
                        help='水平视场角（度），手机约 70°')
    parser.add_argument('--depth-freq', type=int, default=5,
                        help='每 N 帧重算一次深度（默认 5）')
    parser.add_argument('--yolo-model', default='yolov8n.pt',
                        help='YOLO 模型名称（用于双帧掩码）')
    parser.add_argument('--yolo-conf', type=float, default=0.25,
                        help='YOLO 置信度阈值')
    parser.add_argument('--metric3d-size', default='small',
                        choices=['small', 'large'],
                        help='Metric3D 模型大小')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] 文件不存在: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or 'output', exist_ok=True)

    process_video_ego_speed(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_display,
        fov_degrees=args.fov,
        depth_frequency=args.depth_freq,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        model_size=args.metric3d_size,
    )
