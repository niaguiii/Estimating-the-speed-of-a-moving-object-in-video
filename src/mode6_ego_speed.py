"""
Mode 6: Ego-Vehicle Speed Estimation
路面光流 + Metric3D绝对深度 → 自车速度（无需YOLO检测目标）

算法:
1. RAFT 计算帧间光流（像素位移/帧）
2. Metric3D 估计路面绝对深度（米）
3. 路面底部区域采样
4. 速度 = median(pixel_flow * depth / focal_length) * FPS
"""
import os
import sys
import cv2
import csv
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.model_config as model_config
from src.optical_flow_raft import RAFTOpticalFlow
from src.depth_estimation_metric3d import Metric3Dv2


class EgoSpeedEstimator:
    """自车速度估计器（路面光流 + 绝对深度）"""

    def __init__(self, fx, fy, fps, road_region_ratio=0.4,
                 n_samples=500, max_depth=80.0, min_depth=0.5,
                 warmup_frames=30, warmup_alpha=0.5, steady_alpha=0.2,
                 display_interval=15, display_delay=5):
        self.fx = fx
        self.fy = fy
        self.fps = fps
        self.road_region_ratio = road_region_ratio
        self.n_samples = n_samples
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.warmup_frames = warmup_frames
        self.warmup_alpha = warmup_alpha
        self.steady_alpha = steady_alpha
        self.display_interval = display_interval
        self.display_delay = display_delay

        self.current_speed_ms = 0.0
        self.speed_history = deque(maxlen=120)
        self.frame_count = 0
        self.display_speed_ms = 0.0
        self._display_counter = display_interval - 1  # first update fires on frame 1

        self.last_road_avg_depth = 0.0
        self.last_road_valid_pixels = 0

        print(f"[EgoSpeedEstimator] FPS={fps:.1f}  warmup={warmup_frames}f "
              f"\u03b1={warmup_alpha}\u2192{steady_alpha}  display_interval={display_interval}f")

    def estimate_speed(self, flow, depth_map):
        H, W = flow.shape[:2]
        road_y = int(H * (1.0 - self.road_region_ratio))
        road_flow  = flow[road_y:, :]
        road_depth = depth_map[road_y:, :]

        valid = (road_depth > self.min_depth) & (road_depth < self.max_depth)
        self.last_road_valid_pixels = int(valid.sum())
        valid_idx = np.argwhere(valid)

        if len(valid_idx) < 20:
            self.frame_count += 1
            return self.current_speed_ms

        n = min(self.n_samples, len(valid_idx))
        chosen = valid_idx[np.random.choice(len(valid_idx), n, replace=False)]
        rows, cols = chosen[:, 0], chosen[:, 1]

        dy = road_flow[rows, cols, 1]   # 竖直分量：前进>0，倒车<0；转弯不影响 dy
        z  = road_depth[rows, cols]

        self.last_road_avg_depth = float(np.mean(z))

        # 仅用前向分量（dy）；忽略 dx 避免转弯时横向光流虚增速度
        forward_speeds = (dy * z / self.fy) * self.fps  # 有符号：前进正，倒车负

        valid_s = np.abs(forward_speeds) < 55.6  # abs < 200 km/h
        if valid_s.sum() < 5:
            self.frame_count += 1
            return self.current_speed_ms

        raw_speed = float(np.median(forward_speeds[valid_s]))

        # Two-phase EMA: fast warmup → slow steady
        alpha = self.warmup_alpha if self.frame_count < self.warmup_frames else self.steady_alpha

        # Outlier clamp (only after warmup to avoid suppressing initial climb)
        if self.frame_count >= self.warmup_frames and abs(self.current_speed_ms) > 0.1:
            cap_val = abs(self.current_speed_ms) * 3.0
            raw_speed = max(-cap_val, min(raw_speed, cap_val))

        self.current_speed_ms = alpha * raw_speed + (1.0 - alpha) * self.current_speed_ms
        self.speed_history.append(self.current_speed_ms)
        self.frame_count += 1

        # Update display speed every display_interval frames
        self._display_counter += 1
        if self._display_counter >= self.display_interval:
            self.display_speed_ms = self.current_speed_ms
            self._display_counter = 0

        return self.current_speed_ms

    def get_display_speed_ms(self):
        """\u7a33\u5b9a\u663e\u793a\u901f\u5ea6\uff1a\u51b7\u542f\u52a8\u6291\u5236 + \u6bcfN\u5e27\u66f4\u65b0\u4e00\u6b21\uff0c\u51cf\u5c11\u6570\u5b57\u8df3\u52a8"""
        if self.frame_count < self.display_delay:
            return 0.0
        return self.display_speed_ms


def _draw_panel(frame, speed_ms, speed_history, frame_idx, road_y):
    H, W = frame.shape[:2]
    speed_kmh = speed_ms * 3.6
    abs_kmh   = abs(speed_kmh)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (330, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "EGO VEHICLE SPEED", (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    color = (0, 255, 80) if abs_kmh < 60 else (0, 165, 255) if abs_kmh < 100 else (0, 60, 255)
    cv2.putText(frame, f"{speed_kmh:+6.1f} km/h", (20, 95),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, color, 2)
    cv2.putText(frame, f"{speed_ms:+.2f} m/s", (20, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    if len(speed_history) > 2:
        gx, gy, gw, gh = W - 215, 10, 200, 90
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (gx, gy), (gx + gw, gy + gh), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0, frame)
        history = list(speed_history)
        max_v = max(max(abs(s) * 3.6 for s in history), 10.0)
        pts = []
        for i, s in enumerate(history):
            px = gx + int(i / max(len(history) - 1, 1) * gw)
            py = gy + gh - int(abs(s) * 3.6 / max_v * gh)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0, 220, 120), 1)
        cv2.putText(frame, f"max {max_v:.0f} km/h", (gx + 5, gy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

    cv2.line(frame, (0, road_y), (W, road_y), (0, 220, 220), 1)
    cv2.putText(frame, "road region", (8, road_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 220), 1)
    cv2.putText(frame, f"frame {frame_idx}", (W - 110, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
    return frame


def process_video_ego_speed(input_path, output_path, show_video=True,
                             fov_degrees=70.0, depth_frequency=5,
                             road_region_ratio=0.4, model_size='small'):
    """
    Mode 6 主函数：自车速度估计

    Args:
        input_path: 输入视频
        output_path: 输出视频
        show_video: 是否显示实时窗口
        fov_degrees: 水平视场角（度），手机约70°
        depth_frequency: 每N帧重算一次深度
        road_region_ratio: 路面采样区域（图像底部比例）
        model_size: Metric3D大小 'small'/'large'
    """
    print("=" * 60)
    print("Mode 6: Ego Speed  |  RAFT + Metric3D v2  (No YOLO)")
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {input_path}")
        return False

    fps    = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print("\u26a0\ufe0f  FPS not detected from container, defaulting to 30.0")
        fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height}  {fps:.1f}fps  {total} frames")

    print("\n[1/2] Loading RAFT...")
    raft = RAFTOpticalFlow()
    print("✅ RAFT loaded")

    print("\n[2/2] Loading Metric3D v2...")
    depth_estimator = Metric3Dv2(model_size=model_size)
    if not depth_estimator.is_available():
        print("[ERROR] Metric3D v2 failed to load")
        cap.release()
        return False
    print("✅ Metric3D v2 loaded")

    intrinsics = depth_estimator.estimate_camera_intrinsics(width, height, fov_degrees=fov_degrees)
    fx, fy = intrinsics['fx'], intrinsics['fy']
    print(f"[Intrinsics] FOV={fov_degrees}°  fx={fx:.1f}  fy={fy:.1f}")

    speed_est = EgoSpeedEstimator(fx=fx, fy=fy, fps=fps,
                                   road_region_ratio=road_region_ratio)
    road_y = int(height * (1.0 - road_region_ratio))

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if output_path:
        _op = Path(output_path)
        output_path = str(_op.with_name(_op.stem + '_' + run_ts + _op.suffix))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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

            if frame_idx % depth_frequency == 0:
                depth_cache = depth_estimator.estimate_depth(frame, intrinsics)

            speed_ms = 0.0
            if prev_frame is not None and depth_cache is not None:
                flow = raft.compute_flow(prev_frame, frame)
                speed_ms = speed_est.estimate_speed(flow, depth_cache)

            display_speed = speed_est.get_display_speed_ms()
            vis = _draw_panel(frame.copy(), display_speed,
                              speed_est.speed_history, frame_idx, road_y)

            if writer:
                writer.write(vis)
            if show_video:
                cv2.imshow("Mode 6 - Ego Speed", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            csv_rows.append({
                'frame': frame_idx,
                'speed_ms': round(speed_ms, 4),
                'speed_kmh': round(speed_ms * 3.6, 4),
                'road_avg_depth_m': round(speed_est.last_road_avg_depth, 2),
                'road_valid_pixels': speed_est.last_road_valid_pixels,
            })

            prev_frame = frame.copy()
            frame_idx += 1

            if frame_idx > 0 and frame_idx % 50 == 0:
                if total > 0:
                    print(f"  [{frame_idx}/{total}]  {speed_ms * 3.6:.1f} km/h")
                else:
                    print(f"  [{frame_idx} frames]  {speed_ms * 3.6:.1f} km/h")

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_video:
            cv2.destroyAllWindows()

    if output_path and csv_rows:
        base_csv = str(Path(output_path).with_suffix(''))

        # ── CSV: 按秒汇总（m/s单位，含累计位移 + 汇总行）────────
        fps_safe  = fps if fps > 0 else 30.0
        fps_int   = max(1, int(round(fps_safe)))
        stats_rows = []
        cumulative_disp = 0.0

        for sec_start in range(0, len(csv_rows), fps_int):
            sec_rows   = csv_rows[sec_start:sec_start + fps_int]
            sec_speeds = [r['speed_ms'] for r in sec_rows]
            duration_s = len(sec_rows) / fps_safe
            avg_spd    = sum(sec_speeds) / len(sec_speeds)
            disp       = avg_spd * duration_s
            cumulative_disp += disp
            stats_rows.append({
                'second':                    sec_start // fps_int,
                'start_frame':               sec_rows[0]['frame'],
                'end_frame':                 sec_rows[-1]['frame'],
                'avg_speed_ms':              round(avg_spd, 3),
                'max_speed_ms':              round(max(sec_speeds), 3),
                'min_speed_ms':              round(min(sec_speeds), 3),
                'displacement_m':            round(disp, 2),
                'cumulative_displacement_m': round(cumulative_disp, 2),
            })

        # 汇总行
        total_duration_s = len(csv_rows) / fps_safe
        all_speeds = [r['speed_ms'] for r in csv_rows]
        overall_avg = cumulative_disp / total_duration_s if total_duration_s > 0 else 0.0
        stats_rows.append({
            'second':                    'SUMMARY',
            'start_frame':               csv_rows[0]['frame'],
            'end_frame':                 csv_rows[-1]['frame'],
            'avg_speed_ms':              round(overall_avg, 3),
            'max_speed_ms':              round(max(all_speeds), 3),
            'min_speed_ms':              round(min(all_speeds), 3),
            'displacement_m':            round(cumulative_disp, 2),
            'cumulative_displacement_m': round(cumulative_disp, 2),
        })

        stats_csv_path = base_csv + '_stats.csv'
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.DictWriter(f, fieldnames=stats_rows[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(stats_rows)
        n_secs = len(stats_rows) - 1
        print(f"[CSV] Stats: {stats_csv_path}  ({n_secs}s + summary  |  total disp {cumulative_disp:.1f} m)")

    print(f"\n[OK] Done. {frame_idx} frames processed")
    return True
