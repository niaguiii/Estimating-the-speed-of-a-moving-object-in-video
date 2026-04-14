# -*- coding: utf-8 -*-
"""
Mode 5: YOLOv8 + ByteTrack + RAFT + Metric3D v2.

This pipeline estimates object speed in moving-camera videos by combining
object detection and tracking, camera-motion compensation from RAFT optical
flow, and absolute metric depth from Metric3D v2.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.enhance_video import _safe_cv2_write, get_video_writer

# Import model_config as early as possible so cache and environment variables are set.
try:
    from . import model_config
except ImportError:
    import model_config

try:
    from .depth_estimation_metric3d import Metric3Dv2
    from .optical_flow_raft import RAFTOpticalFlow
except ImportError:
    from depth_estimation_metric3d import Metric3Dv2
    from optical_flow_raft import RAFTOpticalFlow


MOVABLE_CLASSES = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "train",
    "boat",
    "airplane",
    "dog",
    "cat",
    "horse",
    "bird",
    "cow",
    "sheep",
}


# =============================================================================
# CSV utilities
# =============================================================================

def write_csv_with_header(csv_path: str, fieldnames: list, rows: list,
                          header_lines: list | None = None):
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if header_lines:
            for line in header_lines:
                f.write(f"# {line}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(round(x1), 0, width))
    y1 = int(np.clip(round(y1), 0, height))
    x2 = int(np.clip(round(x2), 0, width))
    y2 = int(np.clip(round(y2), 0, height))
    return x1, y1, x2, y2


def _collect_movable_boxes(boxes, class_ids, confidences, names, conf_threshold):
    movable_boxes = []
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        class_name = names[class_id]
        if class_name in MOVABLE_CLASSES and float(conf) >= conf_threshold:
            movable_boxes.append(tuple(float(v) for v in box))
    return movable_boxes


def _estimate_camera_motion_masked(flow, current_boxes, previous_boxes, flow_min=0.01):
    """Estimate camera motion from static, observable flow pixels only."""
    valid = np.isfinite(flow[..., 0]) & np.isfinite(flow[..., 1])
    flow_mag = np.linalg.norm(flow, axis=2)
    valid &= flow_mag > flow_min

    height, width = flow.shape[:2]
    for box in list(previous_boxes) + list(current_boxes):
        x1, y1, x2, y2 = _clamp_box(box, width, height)
        if x2 > x1 and y2 > y1:
            valid[y1:y2, x1:x2] = False

    if int(valid.sum()) < 200:
        return float(np.median(flow[:, :, 0])), float(np.median(flow[:, :, 1]))

    return float(np.median(flow[..., 0][valid])), float(np.median(flow[..., 1][valid]))


class Metric3DSpeedEstimator:
    """Depth-aware 3D speed estimator for tracked objects."""

    def __init__(self, fps: float = 30.0, display_interval: int = 8, window_size: int = 7):
        """
        Args:
            fps: Video frame rate.
            display_interval: Update cadence for the rendered speed label.
            window_size: Number of frames used in the sliding baseline window.
        """
        self.fps = fps
        self.window_size = window_size
        self.position_window = {}
        self.speed_history = {}
        self.depth_history = {}
        self.track_frame_count = {}
        self.depth_alpha = 0.15
        self.ema_alpha = 0.4
        self.display_delay = max(3, window_size - 1)
        self.display_interval = display_interval
        self.display_speed_history = {}
        self.display_counter = {}
        self.last_valid_frame = {}

        print(
            f"[Metric3DEstimator] FPS={fps}, window={window_size}f "
            f"({window_size / fps * 1000:.0f}ms), display_interval={display_interval}f"
        )

    def calculate_3d_speed(self, track_id: int,
                           current_pos_2d: tuple,
                           current_depth: float,
                           camera_motion_2d: tuple,
                           intrinsics: dict,
                           frame_idx: int = 0) -> float:
        """
        Estimate depth-aware 3D speed in meters per second.

        Args:
            track_id: Tracker ID for the current object.
            current_pos_2d: Current box-center position in pixels.
            current_depth: Current Metric3D depth in meters.
            camera_motion_2d: Camera motion estimated from RAFT, in pixels/frame.
            intrinsics: Camera intrinsics dictionary with fx, fy, cx, cy.
            frame_idx: Current frame index.

        Returns:
            Smoothed 3D object speed in m/s.
        """
        cx, cy = current_pos_2d
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx_cam = intrinsics['cx']
        cy_cam = intrinsics['cy']

        if current_depth <= 0:
            return 0.0

        if track_id in self.depth_history:
            smooth_depth = (
                self.depth_alpha * current_depth
                + (1 - self.depth_alpha) * self.depth_history[track_id]
            )
        else:
            smooth_depth = current_depth
        self.depth_history[track_id] = smooth_depth

        comp_cx = cx - camera_motion_2d[0]
        comp_cy = cy - camera_motion_2d[1]
        x_world = (comp_cx - cx_cam) * smooth_depth / fx
        y_world = (comp_cy - cy_cam) * smooth_depth / fy
        z_world = smooth_depth

        if track_id in self.last_valid_frame:
            if frame_idx - self.last_valid_frame[track_id] > 1:
                if track_id in self.position_window:
                    self.position_window[track_id].clear()
                self.speed_history.pop(track_id, None)
                self.track_frame_count[track_id] = 0
                self.display_speed_history[track_id] = 0.0
                self.display_counter[track_id] = 0
        self.last_valid_frame[track_id] = frame_idx

        self.track_frame_count[track_id] = self.track_frame_count.get(track_id, 0) + 1
        frame_count = self.track_frame_count[track_id]

        if track_id not in self.position_window:
            self.position_window[track_id] = deque(maxlen=self.window_size)
        self.position_window[track_id].append((x_world, y_world, z_world))

        window = self.position_window[track_id]
        if len(window) >= 2:
            oldest = window[0]
            n_frames = len(window) - 1
            dx = x_world - oldest[0]
            dy = y_world - oldest[1]
            dz = z_world - oldest[2]
            distance_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            raw_speed = distance_3d * self.fps / n_frames

            if track_id in self.speed_history:
                speed_ms = (
                    self.ema_alpha * raw_speed
                    + (1 - self.ema_alpha) * self.speed_history[track_id]
                )
            else:
                speed_ms = raw_speed
            self.speed_history[track_id] = speed_ms
        else:
            speed_ms = 0.0

        if track_id not in self.display_counter:
            self.display_counter[track_id] = 0
            self.display_speed_history[track_id] = 0.0
        self.display_counter[track_id] += 1
        if self.display_counter[track_id] >= self.display_interval:
            self.display_speed_history[track_id] = speed_ms
            self.display_counter[track_id] = 0

        if frame_count <= self.display_delay:
            return 0.0
        return speed_ms

    def get_display_speed(self, track_id: int) -> float:
        """Return the debounced speed used for on-frame rendering."""
        if self.track_frame_count.get(track_id, 0) <= self.display_delay:
            return 0.0
        return self.display_speed_history.get(track_id, 0.0)


def process_video_metric3d(input_path: str, output_path: str,
                           show_video: bool = True,
                           conf_threshold: float = 0.25,
                           show_depth: bool = True,
                           depth_frequency: int = 5,
                           model_size: str = 'small',
                           fov_degrees: float = 60.0,
                           append_timestamp: bool = True):
    """
    Run Mode 5 with YOLOv8 + ByteTrack + RAFT + Metric3D v2.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        show_video: Whether to show the preview window.
        conf_threshold: Detection confidence threshold.
        show_depth: Whether to render the depth inset on the output video.
        depth_frequency: Recompute depth every N frames.
        model_size: Metric3D model size.
        fov_degrees: Horizontal field of view in degrees.
        append_timestamp: Whether to append a timestamp to the output stem.

    Returns:
        Absolute output video path on success, otherwise False.
    """
    print("=" * 60)
    print("Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8")
    print("=" * 60)

    print("\n[1/4] Loading YOLOv8...")
    try:
        yolo_path = model_config.get_model_path('yolov8n.pt')
        model = YOLO(yolo_path)
        _ = model.names
        print(f"[OK] YOLOv8 loaded from {yolo_path}")
    except Exception as e:
        print(f"[ERROR] YOLO model loading failed: {e}")
        return False

    print("\n[2/4] Loading RAFT...")
    raft = RAFTOpticalFlow(model_type='small', device='auto')
    if not raft.is_available():
        print("[ERROR] RAFT model not available, aborting.")
        return False
    print("[OK] RAFT loaded")

    print("\n[3/4] Loading Metric3D v2...")
    depth_estimator = Metric3Dv2(model_size=model_size, device='auto')
    if not depth_estimator.is_available():
        print("[ERROR] Metric3D v2 not available, aborting.")
        return False
    print("[OK] Metric3D v2 loaded")

    print("\n[4/4] Opening video...")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open: {input_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print("[WARN] FPS not detected from container, defaulting to 25.0")
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[OK] Video: {width}x{height} @ {fps}FPS, {total_frames} frames")

    intrinsics = depth_estimator.estimate_camera_intrinsics(
        width,
        height,
        fov_degrees=fov_degrees,
    )
    print(
        f"[Intrinsics] FOV={fov_degrees}deg  "
        f"fx={intrinsics['fx']:.1f}  fy={intrinsics['fy']:.1f}"
    )

    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path_obj = Path(output_path)
    if append_timestamp:
        output_path = os.path.abspath(
            str(output_path_obj.with_name(output_path_obj.stem + '_' + run_ts + output_path_obj.suffix))
        )
    else:
        output_path = os.path.abspath(str(output_path_obj))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[Output] {output_path}")

    out = get_video_writer(output_path, fps, width, height)
    if not out.isOpened():
        print("[ERROR] VideoWriter could not create the output video. Check FFmpeg / MP4 support.")
        cap.release()
        return False

    speed_estimator = Metric3DSpeedEstimator(fps=fps)

    print("\nProcessing video...")
    print("-" * 60)

    frame_idx = 0
    frames_written = 0
    prev_frame = None
    prev_movable_boxes = []
    track_positions = {}
    depth_map_cache = None
    depth_stride = max(int(depth_frequency), 1)
    csv_rows = []
    crops_dir = str(Path(output_path).with_suffix('')) + '_crops'
    os.makedirs(crops_dir, exist_ok=True)
    first_crop_paths = {}
    frame_fieldnames = [
        'frame', 'track_id', 'class_name', 'confidence',
        'cx', 'cy', 'x1', 'y1', 'x2', 'y2',
        'camera_dx', 'camera_dy', 'depth_meters', 'speed_ms'
    ]
    object_fieldnames = [
        'track_id', 'class_name', 'first_time_s', 'last_time_s', 'duration_s',
        'avg_speed_ms', 'max_speed_ms', 'min_speed_ms', 'avg_depth_m',
        'status', 'first_crop_path'
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.track(frame, conf=conf_threshold, persist=True, verbose=False)

        current_movable_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            class_ids_np = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences_np = results[0].boxes.conf.cpu().numpy()
            current_movable_boxes = _collect_movable_boxes(
                boxes_np,
                class_ids_np,
                confidences_np,
                model.names,
                conf_threshold,
            )

        camera_motion = (0.0, 0.0)
        if prev_frame is not None:
            flow = raft.compute_flow(prev_frame, frame, output_height=height, output_width=width)
            camera_motion = _estimate_camera_motion_masked(
                flow,
                current_boxes=current_movable_boxes,
                previous_boxes=prev_movable_boxes,
                flow_min=0.01,
            )

        if depth_map_cache is None or (frame_idx - 1) % depth_stride == 0:
            depth_map_cache = depth_estimator.estimate_depth(frame, intrinsics)

        prev_frame = frame.copy()
        prev_movable_boxes = current_movable_boxes

        annotated_frame = frame.copy()

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes))

            for box, class_id, conf, track_id in zip(boxes, class_ids, confidences, track_ids):
                x1, y1, x2, y2 = box.astype(int)
                class_name = model.names[class_id]

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                depth_meters = depth_estimator.get_object_depth(depth_map_cache, (x1, y1, x2, y2))

                prev_pos_2d = track_positions.get(track_id)
                speed = speed_estimator.calculate_3d_speed(
                    track_id=track_id,
                    current_pos_2d=(cx, cy),
                    current_depth=depth_meters,
                    camera_motion_2d=camera_motion,
                    intrinsics=intrinsics,
                    frame_idx=frame_idx,
                )

                if track_id not in track_positions:
                    pad = 8
                    crop = frame[max(0, y1 - pad):min(height, y2 + pad),
                                 max(0, x1 - pad):min(width, x2 + pad)]
                    crop_name = f"track_{track_id}_{class_name}.jpg"
                    crop_path = os.path.join(crops_dir, crop_name)
                    if crop.size > 0:
                        cv2.imwrite(crop_path, crop)
                    first_crop_paths[track_id] = f"crops/{crop_name}"

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
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'camera_dx': round(float(camera_motion[0]), 3),
                    'camera_dy': round(float(camera_motion[1]), 3),
                    'depth_meters': round(float(smooth_depth), 3),
                    'speed_ms': round(float(speed), 3),
                })

                if depth_meters > 0:
                    depth_color_val = min(255, int(depth_meters / 30.0 * 255))
                    color = (0, 255 - depth_color_val, depth_color_val)
                else:
                    color = (128, 128, 128)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                pixel_speed = 0.0
                if prev_pos_2d is not None:
                    prev_cx, prev_cy = prev_pos_2d
                    real_dx = (cx - prev_cx) - camera_motion[0]
                    real_dy = (cy - prev_cy) - camera_motion[1]
                    pixel_speed = np.sqrt(real_dx ** 2 + real_dy ** 2)

                if display_speed > 0:
                    label = (
                        f"ID{track_id} {class_name} (conf:{conf:.2f}) "
                        f"{pixel_speed:.1f}px/f | {display_speed:.1f}m/s | {smooth_depth:.1f}m"
                    )
                else:
                    label = f"ID{track_id} {class_name} (conf:{conf:.2f}) | {smooth_depth:.1f}m"

                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.6
                thickness = 2
                (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w + 6, y1), color, -1)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 3, y1 - 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 3, y1 - 5),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

        if show_depth and depth_map_cache is not None and (frame_idx - 1) % depth_stride == 0:
            depth_vis = depth_estimator.visualize_depth(depth_map_cache, max_depth=50.0)
            depth_small = cv2.resize(depth_vis, (width // 4, height // 4))
            annotated_frame[10:10 + height // 4, width - width // 4 - 10:width - 10] = depth_small
            cv2.putText(
                annotated_frame,
                'Metric3D (meters)',
                (width - width // 4 - 10, height // 4 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        panel_color = (0, 200, 200)
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_idx}/{total_frames} | Metric3D v2 + RAFT",
            (10, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            panel_color,
            2,
        )
        cv2.putText(
            annotated_frame,
            f"Camera: dx={camera_motion[0]:.1f} dy={camera_motion[1]:.1f}",
            (10, height - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        frame_out = np.ascontiguousarray(annotated_frame)
        if _safe_cv2_write(out, frame_out):
            frames_written += 1
        if show_video:
            cv2.imshow('Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_idx % 30 == 0:
            if total_frames > 0:
                print(f"Progress: {frame_idx / total_frames * 100:.1f}% ({frame_idx}/{total_frames})", end='\r')
            else:
                print(f"Progress: {frame_idx} frames processed", end='\r')

    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()

    base_csv = str(Path(output_path).with_suffix(''))
    if csv_rows:
        frames_csv_path = base_csv + '_frames.csv'
        write_csv_with_header(
            frames_csv_path,
            fieldnames=list(csv_rows[0].keys()),
            rows=csv_rows,
            header_lines=[
                'mode: 5 | algorithm: YOLOv8 + ByteTrack + RAFT + Metric3D v2 (absolute depth)',
                'unit: speed_ms = m/s; depth_meters = absolute depth from Metric3D v2 (meters)',
                'camera_dx/dy: masked camera motion (pixels/frame, movable objects and tiny flow excluded)',
                'speed: object speed after camera-motion compensation (m/s)',
            ],
        )
        print(f"[CSV] Per-frame: {frames_csv_path} ({len(csv_rows)} rows)")

        vehicle_data = defaultdict(list)
        for row in csv_rows:
            vehicle_data[row['track_id']].append(row)

        moving_threshold_ms = 0.5
        object_rows = []
        for tid, rows in sorted(vehicle_data.items()):
            moving = [row for row in rows if row['speed_ms'] > moving_threshold_ms]
            speeds_ms = [row['speed_ms'] for row in moving] if moving else []
            depths = [row['depth_meters'] for row in rows if row['depth_meters'] > 0]
            avg_speed = round(sum(speeds_ms) / len(speeds_ms), 3) if speeds_ms else None
            max_speed = round(max(speeds_ms), 3) if speeds_ms else None
            min_speed = round(min(speeds_ms), 3) if speeds_ms else None
            object_rows.append({
                'track_id': tid,
                'class_name': rows[0]['class_name'],
                'first_time_s': round((rows[0]['frame'] - 1) / fps, 2),
                'last_time_s': round((rows[-1]['frame'] - 1) / fps, 2),
                'duration_s': round((rows[-1]['frame'] - rows[0]['frame'] + 1) / fps, 2),
                'avg_speed_ms': avg_speed,
                'max_speed_ms': max_speed,
                'min_speed_ms': min_speed,
                'avg_depth_m': round(sum(depths) / len(depths), 2) if depths else 0.0,
                'status': 'moving' if speeds_ms and max(speeds_ms) > moving_threshold_ms else 'unknown',
                'first_crop_path': first_crop_paths.get(tid, ''),
            })

        objects_csv_path = base_csv + '_objects.csv'
        write_csv_with_header(
            objects_csv_path,
            fieldnames=list(object_rows[0].keys()),
            rows=object_rows,
            header_lines=[
                'mode: 5 | algorithm: YOLOv8 + ByteTrack + RAFT + Metric3D v2 (absolute depth)',
                'unit: speed_ms = m/s (object speed after camera-motion compensation)',
            ],
        )
        print(f"[CSV] Per-object: {objects_csv_path} ({len(object_rows)} objects)")
    else:
        frames_csv_path = base_csv + '_frames.csv'
        write_csv_with_header(
            frames_csv_path,
            fieldnames=frame_fieldnames,
            rows=[],
            header_lines=[
                'mode: 5 | algorithm: YOLOv8 + ByteTrack + RAFT + Metric3D v2 (absolute depth)',
                'unit: speed_ms = m/s; depth_meters = absolute depth from Metric3D v2 (meters)',
                'camera_dx/dy: masked camera motion (pixels/frame, movable objects and tiny flow excluded)',
                'speed: object speed after camera-motion compensation (m/s)',
            ],
        )
        objects_csv_path = base_csv + '_objects.csv'
        write_csv_with_header(
            objects_csv_path,
            fieldnames=object_fieldnames,
            rows=[],
            header_lines=[
                'mode: 5 | algorithm: YOLOv8 + ByteTrack + RAFT + Metric3D v2 (absolute depth)',
                'unit: speed_ms = m/s (object speed after camera-motion compensation)',
            ],
        )
        print(f"[CSV] Per-frame: {frames_csv_path} (0 rows)")
        print(f"[CSV] Per-object: {objects_csv_path} (0 objects)")

    if frame_idx == 0:
        print('[ERROR] No frames were read from the input video')
        return False

    if frames_written == 0:
        print('[ERROR] No video frames were written; check OpenCV VideoWriter / FFmpeg')
        return False
    if not os.path.isfile(output_path):
        print(f"[ERROR] Expected output video is missing: {output_path}")
        return False

    size_bytes = os.path.getsize(output_path)
    if size_bytes < 1024:
        print(
            f"[ERROR] Output video is suspiciously small ({size_bytes} bytes); encoding may have failed. "
            'Please check FFmpeg / VideoWriter availability and retry.'
        )
        return False

    print(f"\n{'=' * 60}")
    print('[OK] Phase 3 Metric3D processing complete')
    print(f"[Output] {output_path}")
    print(f"[Summary] Processed {frame_idx} frames (video frames written: {frames_written})")
    print(f"{'=' * 60}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3 Metric3D: RAFT + Metric3D v2 + YOLOv8')
    parser.add_argument('--input', type=str, default='input/test_video.mp4')
    parser.add_argument('--output', type=str, default='output/phase3_metric3d_output.mp4')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-depth', action='store_true')
    parser.add_argument(
        '--fov',
        type=float,
        default=60.0,
        help='Horizontal field of view in degrees',
    )
    parser.add_argument(
        '--depth-freq',
        type=int,
        default=5,
        help='Depth estimation frequency (every N frames)',
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='small',
        choices=['small', 'large', 'giant2'],
        help='Metric3D model size',
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output) or '.'
    os.makedirs(output_dir, exist_ok=True)

    success = process_video_metric3d(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_show,
        conf_threshold=args.conf,
        show_depth=not args.no_depth,
        depth_frequency=args.depth_freq,
        model_size=args.model_size,
        fov_degrees=args.fov,
    )

    if success:
        print('\n[OK] Done!')
    else:
        print('\n[ERROR] Failed!')
        sys.exit(1)

