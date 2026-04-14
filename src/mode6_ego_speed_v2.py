# -*- coding: utf-8 -*-
"""
Mode 6 v2: full-frame ego speed estimation.

Core idea:
1. Compute full-frame RAFT optical flow.
2. Remove YOLO-detected movable objects.
3. Remove low-observable pixels with tiny flow magnitude.
4. Remove invalid depth and pixels with |y-cy| too small.
5. Estimate per-pixel signed axial speed with:
      speed_i = dy_i * Z_i * fps / (y_i - cy)
6. Aggregate with median and smooth with EMA.
"""

import csv
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from src.enhance_video import _safe_cv2_write, get_video_writer

try:
    from . import model_config
except ImportError:
    import model_config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.depth_estimation_metric3d import Metric3Dv2
from src.optical_flow_raft import RAFTOpticalFlow


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


def write_csv_with_header(csv_path, fieldnames, rows, header_lines=None):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        if header_lines:
            for line in header_lines:
                handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _boxes_overlap(box_a, box_b, iou_thresh=0.3):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return False

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return union_area > 0 and (inter_area / union_area) >= iou_thresh


def _merge_boxes(box_a, box_b):
    return (
        min(box_a[0], box_b[0]),
        min(box_a[1], box_b[1]),
        max(box_a[2], box_b[2]),
        max(box_a[3], box_b[3]),
    )


def _clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = int(np.clip(round(x1), 0, width))
    y1 = int(np.clip(round(y1), 0, height))
    x2 = int(np.clip(round(x2), 0, width))
    y2 = int(np.clip(round(y2), 0, height))
    return x1, y1, x2, y2


def _quality_from_valid_rate(valid_rate, is_turning):
    if is_turning:
        return "TURN"
    if valid_rate >= 0.20:
        return "GOOD"
    if valid_rate >= 0.08:
        return "FAIR"
    return "POOR"


def _quality_color(quality):
    if quality == "GOOD":
        return (60, 220, 80)
    if quality == "FAIR":
        return (0, 210, 255)
    if quality == "POOR":
        return (0, 90, 255)
    return (0, 220, 255)


def _make_flow_visualization(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class EgoSpeedEstimator:
    """Full-frame signed ego-speed estimator."""

    def __init__(
        self,
        fx,
        fy,
        cx,
        cy,
        fps,
        n_samples=800,
        depth_min=1.0,
        depth_max=100.0,
        y_min_offset=20.0,
        flow_min=0.05,
        iou_thresh=0.3,
        box_area_thresh=500.0,
        box_conf_thresh=0.5,
        turn_threshold=2.0,
        warmup_frames=30,
        warmup_alpha=0.5,
        steady_alpha=0.2,
        display_interval=15,
        display_delay=5,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fps = fps
        self.n_samples = n_samples
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.y_min_offset = y_min_offset
        self.flow_min = flow_min
        self.iou_thresh = iou_thresh
        self.box_area_thresh = box_area_thresh
        self.box_conf_thresh = box_conf_thresh
        self.turn_threshold = turn_threshold
        self.warmup_frames = warmup_frames
        self.warmup_alpha = warmup_alpha
        self.steady_alpha = steady_alpha
        self.display_interval = display_interval
        self.display_delay = display_delay

        self.current_speed_ms = 0.0
        self.display_speed_ms = 0.0
        self.frame_count = 0
        self.speed_history = deque(maxlen=120)
        self._display_counter = display_interval - 1

        self.last_quality = "POOR"
        self.last_flow_valid_rate = 0.0
        self.last_valid_pixels = 0
        self.last_total_pixels = 0
        self.last_raw_speed = 0.0
        self.last_dx_median = 0.0
        self.last_valid_mask = None

        self._prev_detections = []
        self._curr_detections = []

        print(
            f"[EgoSpeedEstimator] fps={fps:.2f} cx={cx:.1f} cy={cy:.1f} "
            f"depth=[{depth_min:.1f},{depth_max:.1f}] "
            f"|y-cy|>{y_min_offset:.1f}, flow>{flow_min:.2f}"
        )

    def set_detections(self, detections):
        self._prev_detections = self._curr_detections
        self._curr_detections = [
            det
            for det in detections
            if det.get("class_name", "").lower() in MOVABLE_CLASSES
        ]

    def estimate_speed(self, flow, depth_map):
        height, width = flow.shape[:2]
        self.last_total_pixels = height * width

        yolo_mask = self._build_yolo_mask(height, width)
        flow_mag = np.linalg.norm(flow, axis=2)
        observability_mask = np.isfinite(flow_mag) & (flow_mag > self.flow_min)
        depth_mask = np.isfinite(depth_map) & (depth_map > self.depth_min) & (depth_map < self.depth_max)
        y_rel = np.arange(height, dtype=np.float32).reshape(-1, 1) - self.cy
        geometry_mask = np.abs(y_rel) > self.y_min_offset

        valid_mask = yolo_mask & observability_mask & depth_mask & geometry_mask
        self.last_valid_mask = valid_mask
        self.last_valid_pixels = int(valid_mask.sum())
        self.last_flow_valid_rate = (
            self.last_valid_pixels / float(self.last_total_pixels)
            if self.last_total_pixels
            else 0.0
        )

        if self.last_valid_pixels < 20:
            self.last_dx_median = 0.0
            self.last_raw_speed = self.current_speed_ms
            self.last_quality = _quality_from_valid_rate(self.last_flow_valid_rate, False)
            self._advance_frame()
            return self.current_speed_ms

        rows, cols = np.where(valid_mask)
        if len(rows) > self.n_samples:
            sample_idx = np.linspace(0, len(rows) - 1, self.n_samples, dtype=int)
            rows = rows[sample_idx]
            cols = cols[sample_idx]

        dy = flow[rows, cols, 1].astype(np.float32)
        dx = flow[rows, cols, 0].astype(np.float32)
        depth = depth_map[rows, cols].astype(np.float32)
        y_minus_cy = rows.astype(np.float32) - self.cy

        speeds = dy * depth * self.fps / y_minus_cy
        finite_mask = np.isfinite(speeds)
        speeds = speeds[finite_mask]
        dx = dx[finite_mask]

        if speeds.size == 0:
            self.last_dx_median = 0.0
            self.last_raw_speed = self.current_speed_ms
            self.last_quality = _quality_from_valid_rate(self.last_flow_valid_rate, False)
            self._advance_frame()
            return self.current_speed_ms

        self.last_dx_median = float(np.median(dx))
        is_turning = abs(self.last_dx_median) > self.turn_threshold
        raw_speed = float(np.median(speeds))
        self.last_raw_speed = raw_speed
        self.last_quality = _quality_from_valid_rate(self.last_flow_valid_rate, is_turning)

        alpha = self.warmup_alpha if self.frame_count < self.warmup_frames else self.steady_alpha
        if self.frame_count >= self.warmup_frames and abs(self.current_speed_ms) > 0.5:
            cap_val = max(abs(self.current_speed_ms) * 5.0, 40.0)
            raw_speed = float(np.clip(raw_speed, -cap_val, cap_val))

        self.current_speed_ms = alpha * raw_speed + (1.0 - alpha) * self.current_speed_ms
        self.speed_history.append(self.current_speed_ms)
        self._advance_frame()
        return self.current_speed_ms

    def get_display_speed_ms(self):
        if self.frame_count < self.display_delay:
            return 0.0
        return self.display_speed_ms

    def _build_yolo_mask(self, height, width):
        mask = np.ones((height, width), dtype=bool)
        if not self._prev_detections or not self._curr_detections:
            return mask

        for curr in self._curr_detections:
            if curr.get("confidence", 1.0) < self.box_conf_thresh:
                continue
            curr_box = curr["bbox"]
            for prev in self._prev_detections:
                if prev.get("confidence", 1.0) < self.box_conf_thresh:
                    continue
                if not _boxes_overlap(curr_box, prev["bbox"], self.iou_thresh):
                    continue
                x1, y1, x2, y2 = _clamp_box(_merge_boxes(curr_box, prev["bbox"]), width, height)
                if (x2 - x1) * (y2 - y1) < self.box_area_thresh:
                    continue
                mask[y1:y2, x1:x2] = False
                break
        return mask

    def _advance_frame(self):
        self.frame_count += 1
        self._display_counter += 1
        if self._display_counter >= self.display_interval:
            self.display_speed_ms = self.current_speed_ms
            self._display_counter = 0


def _draw_speed_panel(frame, speed_ms, quality, flow_valid_rate, speed_history, frame_idx):
    height, width = frame.shape[:2]
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 390, 152

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame,
        "EGO SPEED (SIGNED AXIAL)",
        (panel_x + 12, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    color = _quality_color(quality)
    cv2.putText(
        frame,
        f"{speed_ms:+7.2f} m/s",
        (panel_x + 12, panel_y + 64),
        cv2.FONT_HERSHEY_DUPLEX,
        1.35,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"[{quality}]",
        (panel_x + 255, panel_y + 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    valid_percent = flow_valid_rate * 100.0
    cv2.putText(
        frame,
        f"valid pixels: {valid_percent:5.1f}%",
        (panel_x + 12, panel_y + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    if quality in {"FAIR", "POOR", "TURN"}:
        hint_map = {
            "FAIR": "quality: usable but sparse valid pixels",
            "POOR": "quality: very sparse valid pixels",
            "TURN": "turn detected: speed is reference-only",
        }
        cv2.putText(
            frame,
            hint_map[quality],
            (panel_x + 12, panel_y + 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )

    if len(speed_history) > 2:
        chart_x = width - 210
        chart_y = 10
        chart_w, chart_h = 200, 84
        overlay_chart = frame.copy()
        cv2.rectangle(overlay_chart, (chart_x, chart_y), (chart_x + chart_w, chart_y + chart_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_chart, 0.55, frame, 0.45, 0, frame)

        history = list(speed_history)
        max_val = max(max(abs(val) for val in history), 1.0)
        mid_y = chart_y + chart_h // 2
        cv2.line(frame, (chart_x, mid_y), (chart_x + chart_w, mid_y), (90, 90, 90), 1)

        points = []
        for idx, value in enumerate(history):
            px = chart_x + int(idx / max(len(history) - 1, 1) * (chart_w - 1))
            py = mid_y - int((value / max_val) * (chart_h * 0.42))
            points.append((px, py))
        for idx in range(1, len(points)):
            cv2.line(frame, points[idx - 1], points[idx], (0, 220, 120), 1)

        cv2.putText(
            frame,
            f"history |max| {max_val:.1f} m/s",
            (chart_x + 8, chart_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        f"frame {frame_idx}",
        (width - 110, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )
    return frame


def _detect_movable_objects(yolo_model, frame, conf_threshold):
    detections = []
    if yolo_model is None:
        raise RuntimeError("YOLO model is required for Mode 6 v2 dynamic masking")

    try:
        results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"YOLO inference failed during Mode 6 v2 masking: {exc}") from exc

    if not results or results[0].boxes is None:
        return detections

    names = getattr(yolo_model, "names", {})
    for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        cls_id = int(det.cls[0].cpu().numpy())
        if isinstance(names, dict):
            class_name = names.get(cls_id, "")
        elif isinstance(names, list) and 0 <= cls_id < len(names):
            class_name = names[cls_id]
        else:
            class_name = ""
        detections.append(
            {
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "class_name": str(class_name),
                "confidence": float(det.conf[0].cpu().numpy()),
            }
        )
    return detections


def _ensure_diag_dir(output_path):
    diag_dir = Path(output_path).with_name(Path(output_path).stem + "_diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)
    return diag_dir


def _save_diagnostics(diag_dir, frame, flow, valid_mask, prefix="diagnostic"):
    overlay = frame.copy()
    green = np.zeros_like(frame)
    green[..., 1] = 255
    overlay = np.where(valid_mask[..., None], cv2.addWeighted(frame, 0.35, green, 0.65, 0), frame)

    binary = (valid_mask.astype(np.uint8) * 255)
    flow_viz = _make_flow_visualization(flow)

    cv2.imwrite(str(diag_dir / f"{prefix}_valid_mask_overlay.png"), overlay)
    cv2.imwrite(str(diag_dir / f"{prefix}_valid_mask_binary.png"), binary)
    cv2.imwrite(str(diag_dir / f"{prefix}_flow_visualization.png"), flow_viz)


def _print_first_frame_diag(estimator, depth_map):
    print("[Mode6 v2] first valid frame diagnostics")
    print(
        f"  valid_pixels: {estimator.last_valid_pixels}/{estimator.last_total_pixels} "
        f"({estimator.last_flow_valid_rate:.1%})"
    )
    print(f"  quality: {estimator.last_quality}")
    print(f"  dx_median: {estimator.last_dx_median:.3f} px/frame")
    print(f"  raw_speed: {estimator.last_raw_speed:.3f} m/s")
    if estimator.last_valid_mask is not None and estimator.last_valid_pixels > 0:
        valid_depth = depth_map[estimator.last_valid_mask]
        if valid_depth.size:
            print(f"  median_depth: {float(np.median(valid_depth)):.2f} m")


def _print_diag_snapshot(snapshot):
    print(f"[Mode6 v2] best diagnostic frame: {snapshot['frame_idx']}")
    print(
        f"  valid_pixels: {snapshot['valid_pixels']}/{snapshot['total_pixels']} "
        f"({snapshot['valid_rate']:.1%})"
    )
    print(f"  quality: {snapshot['quality']}")
    print(f"  dx_median: {snapshot['dx_median']:.3f} px/frame")
    print(f"  raw_speed: {snapshot['raw_speed']:.3f} m/s")
    print(f"  median_depth: {snapshot['median_depth']:.2f} m")


def _write_csv_outputs(output_path, csv_rows, fps):
    if not csv_rows:
        return

    base_csv = str(Path(output_path).with_suffix(""))
    fps_safe = fps if fps > 0 else 30.0

    frames_csv_path = base_csv + "_frames.csv"
    write_csv_with_header(
        frames_csv_path,
        fieldnames=list(csv_rows[0].keys()),
        rows=csv_rows,
        header_lines=[
            "mode: 6 | algorithm: EgoSpeed full-frame v2",
            "formula: signed_speed = dy * Z * fps / (y - cy)",
            "filters: YOLO movable objects + low-observable flow + invalid depth + degenerate |y-cy|",
            "quality_flag: GOOD/FAIR/POOR/TURN",
            "unit: ego_speed_ms = m/s",
        ],
    )
    print(f"[CSV] Frames: {frames_csv_path} ({len(csv_rows)} rows)")

    second_groups = defaultdict(list)
    for row in csv_rows:
        second_groups[int(row["frame_idx"] / fps_safe)].append(row)

    stats_rows = []
    cumulative_disp = 0.0
    quality_order = {"GOOD": 0, "FAIR": 1, "POOR": 2, "TURN": 3}
    for second in sorted(second_groups):
        sec_rows = second_groups[second]
        sec_speeds = [row["ego_speed_ms"] for row in sec_rows]
        duration_s = len(sec_rows) / fps_safe
        avg_speed = float(np.mean(sec_speeds))
        displacement = avg_speed * duration_s
        cumulative_disp += displacement
        dominant_quality = max(
            [row["quality_flag"] for row in sec_rows],
            key=lambda name: (sum(row["quality_flag"] == name for row in sec_rows), quality_order[name]),
        )
        stats_rows.append(
            {
                "second": second,
                "start_frame": sec_rows[0]["frame_idx"],
                "end_frame": sec_rows[-1]["frame_idx"],
                "avg_speed_ms": round(avg_speed, 3),
                "max_speed_ms": round(max(sec_speeds), 3),
                "min_speed_ms": round(min(sec_speeds), 3),
                "displacement_m": round(displacement, 3),
                "cumulative_displacement_m": round(cumulative_disp, 3),
                "dominant_quality": dominant_quality,
                "avg_valid_pixel_percent": round(
                    float(np.mean([row["valid_pixel_percent"] for row in sec_rows])),
                    2,
                ),
            }
        )

    all_speeds = [row["ego_speed_ms"] for row in csv_rows]
    total_duration_s = len(csv_rows) / fps_safe
    overall_avg = cumulative_disp / total_duration_s if total_duration_s > 0 else 0.0
    stats_rows.append(
        {
            "second": "SUMMARY",
            "start_frame": csv_rows[0]["frame_idx"],
            "end_frame": csv_rows[-1]["frame_idx"],
            "avg_speed_ms": round(overall_avg, 3),
            "max_speed_ms": round(max(all_speeds), 3),
            "min_speed_ms": round(min(all_speeds), 3),
            "displacement_m": round(cumulative_disp, 3),
            "cumulative_displacement_m": round(cumulative_disp, 3),
            "dominant_quality": "-",
            "avg_valid_pixel_percent": round(
                float(np.mean([row["valid_pixel_percent"] for row in csv_rows])),
                2,
            ),
        }
    )

    stats_csv_path = base_csv + "_stats.csv"
    write_csv_with_header(
        stats_csv_path,
        fieldnames=list(stats_rows[0].keys()),
        rows=stats_rows,
        header_lines=[
            "mode: 6 | second-level ego-speed summary",
            "signed displacement is accumulated from second-level average speed",
        ],
    )
    print(f"[CSV] Stats: {stats_csv_path}")


def process_video_ego_speed(
    input_path,
    output_path,
    show_video=True,
    fov_degrees=70.0,
    depth_frequency=5,
    diag_stride=20,
    flow_resolution_mode="native",
    yolo_model="yolov8n.pt",
    yolo_conf=0.25,
    conf_threshold=None,
    model_size="small",
    append_timestamp=True,
):
    """Run Mode 6 v2 and return output_path on success."""
    if conf_threshold is not None:
        yolo_conf = conf_threshold

    print("=" * 60)
    print("Mode 6 v2: Full-frame Ego Speed")
    print("=" * 60)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open input: {input_path}")
        return False

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
        print("[WARN] FPS not detected, using 30.0")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height}  {fps:.2f} fps  {total_frames} frames")

    print(f"[1/3] Loading RAFT... (mode={flow_resolution_mode})")
    raft = RAFTOpticalFlow(resolution_mode=flow_resolution_mode)
    if not raft.is_available():
        cap.release()
        print("[ERROR] RAFT failed to load")
        return False

    print("[2/3] Loading Metric3D...")
    depth_estimator = Metric3Dv2(model_size=model_size)
    if not depth_estimator.is_available():
        cap.release()
        print("[ERROR] Metric3D failed to load")
        return False

    intrinsics = depth_estimator.estimate_camera_intrinsics(width, height, fov_degrees=fov_degrees)
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    print(f"[Intrinsics] fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    print("[3/3] Loading YOLOv8...")
    yolo = None
    try:
        from ultralytics import YOLO as YOLOModel

        yolo_path = model_config.get_model_path(yolo_model)
        yolo = YOLOModel(yolo_path)
        print(f"[YOLO] loaded: {yolo_model}")
    except Exception as exc:
        cap.release()
        print(f"[ERROR] YOLO unavailable, Mode 6 v2 requires dynamic masking: {exc}")
        return False

    estimator = EgoSpeedEstimator(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        fps=fps,
    )

    _op = Path(output_path)
    if append_timestamp:
        from datetime import datetime

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.abspath(
            str(_op.with_name(_op.stem + "_" + run_ts + _op.suffix))
        )
    else:
        output_path = os.path.abspath(str(_op))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = get_video_writer(output_path, fps, width, height)
    if writer is None:
        cap.release()
        print("[ERROR] VideoWriter init failed")
        return False

    diag_dir = _ensure_diag_dir(output_path)
    csv_rows = []
    frames_written = 0
    frame_idx = 0
    prev_frame = None
    depth_cache = None
    diag_stride_safe = max(int(diag_stride), 1)
    diag_saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if depth_cache is None or frame_idx % max(depth_frequency, 1) == 0:
                depth_cache = depth_estimator.estimate_depth(frame, intrinsics)

            speed_ms = estimator.current_speed_ms
            quality = estimator.last_quality
            valid_rate = estimator.last_flow_valid_rate
            detections = _detect_movable_objects(yolo, frame, yolo_conf)
            estimator.set_detections(detections)

            if prev_frame is not None and depth_cache is not None:
                flow = raft.compute_flow(prev_frame, frame, output_height=height, output_width=width)
                speed_ms = estimator.estimate_speed(flow, depth_cache)
                quality = estimator.last_quality
                valid_rate = estimator.last_flow_valid_rate

                if estimator.last_valid_mask is not None and frame_idx % diag_stride_safe == 0:
                    diag_prefix = f"frame_{frame_idx:06d}"
                    _save_diagnostics(
                        diag_dir,
                        frame,
                        flow,
                        estimator.last_valid_mask,
                        prefix=diag_prefix,
                    )
                    diag_saved_count += 1

            display_speed = estimator.get_display_speed_ms()
            vis = _draw_speed_panel(
                frame.copy(),
                display_speed,
                quality,
                valid_rate,
                estimator.speed_history,
                frame_idx,
            )

            if _safe_cv2_write(writer, np.ascontiguousarray(vis)):
                frames_written += 1

            if show_video:
                cv2.imshow("Mode 6 v2 - Ego Speed", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            csv_rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp_s": round(frame_idx / fps, 4),
                    "ego_speed_ms": round(float(speed_ms), 4),
                    "quality_flag": quality,
                    "flow_valid_rate": round(float(valid_rate), 6),
                    "valid_pixel_percent": round(float(valid_rate * 100.0), 3),
                    "valid_pixels": estimator.last_valid_pixels,
                    "total_pixels": estimator.last_total_pixels,
                    "dx_median": round(float(estimator.last_dx_median), 6),
                    "raw_speed_ms": round(float(estimator.last_raw_speed), 6),
                }
            )

            prev_frame = frame.copy()
            frame_idx += 1

            if frame_idx > 0 and frame_idx % 50 == 0:
                print(
                    f"[{frame_idx}/{total_frames}] "
                    f"{speed_ms:+.2f} m/s [{quality}] valid={valid_rate:.1%}"
                )
    finally:
        cap.release()
        writer.release()
        if show_video:
            cv2.destroyAllWindows()

    if frame_idx == 0:
        print("[ERROR] No frames processed")
        return False
    if frames_written == 0:
        print("[ERROR] No frames written")
        return False
    if not os.path.isfile(output_path):
        print(f"[ERROR] Output video missing: {output_path}")
        return False

    _write_csv_outputs(output_path, csv_rows, fps)
    print(f"[Diag] Saved {diag_saved_count} diagnostic snapshots to {diag_dir}")
    print(f"[OK] Done. frames={frame_idx}, written={frames_written}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mode 6 v2: full-frame ego speed")
    parser.add_argument("--input", "-i", required=True, help="Input video")
    parser.add_argument("--output", "-o", default="output/output_ego_speed.mp4", help="Output video")
    parser.add_argument("--no-display", action="store_true", help="Disable live preview")
    parser.add_argument("--fov", type=float, default=70.0, help="Horizontal FOV in degrees")
    parser.add_argument("--depth-freq", type=int, default=5, help="Depth refresh interval")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model name")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument(
        "--metric3d-size",
        default="small",
        choices=["small", "large"],
        help="Metric3D model size",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] File not found: {args.input}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or "output", exist_ok=True)
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
