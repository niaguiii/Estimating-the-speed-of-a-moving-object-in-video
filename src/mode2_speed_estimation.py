# -*- coding: utf-8 -*-
"""
YOLOv8 + ByteTrack + Speed Estimation (Simplified Version)
Phase 2 Implementation - Object Size Based Speed Estimation

Features:
- ByteTrack high-precision tracking
- Auto speed estimation based on object size
- Real-time speed display (km/h)
- Speed statistics panel
- No manual calibration required

Note: This is a simplified version without optical flow.
      Assumes relatively stationary camera.
      Full version with RAFT optical flow coming in Phase 3.
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


# Standard object sizes in meters (width, height)
# Used for automatic pixel-to-meter calibration
OBJECT_REAL_SIZES = {
    # Vehicles
    'car': {'width': 1.8, 'height': 1.5},
    'truck': {'width': 2.5, 'height': 3.0},
    'bus': {'width': 2.5, 'height': 3.2},
    'motorcycle': {'width': 0.8, 'height': 1.2},
    'bicycle': {'width': 0.6, 'height': 1.0},
    
    # People
    'person': {'width': 0.5, 'height': 1.7},
    
    # Animals
    'dog': {'width': 0.5, 'height': 0.6},
    'cat': {'width': 0.4, 'height': 0.3},
    'horse': {'width': 2.0, 'height': 1.6},
    
    # Other
    'traffic light': {'width': 0.3, 'height': 1.0},
    'stop sign': {'width': 0.6, 'height': 0.6},
}


class SpeedEstimator:
    """Speed estimation based on object size"""
    
    def __init__(self, fps=30):
        self.fps = fps
        self.calibration_cache = {}  # Cache pixel/meter ratio per track
    
    def estimate_pixels_per_meter(self, class_name, bbox_width, bbox_height):
        """
        Estimate pixels per meter based on detected object size
        
        Args:
            class_name: Object class name
            bbox_width: Bounding box width in pixels
            bbox_height: Bounding box height in pixels
        
        Returns:
            pixels_per_meter: Estimated conversion ratio
        """
        if class_name not in OBJECT_REAL_SIZES:
            # Default estimation for unknown objects
            return None
        
        real_size = OBJECT_REAL_SIZES[class_name]
        
        # Use width for horizontal objects, height for vertical
        # Average both for better estimation
        px_per_m_width = bbox_width / real_size['width']
        px_per_m_height = bbox_height / real_size['height']
        
        # Use the larger value (more reliable)
        return max(px_per_m_width, px_per_m_height)
    
    def calculate_speed_kmh(self, pixel_velocity, pixels_per_meter):
        """
        Convert pixel velocity to km/h
        
        Args:
            pixel_velocity: Speed in pixels/frame
            pixels_per_meter: Calibration ratio
        
        Returns:
            speed_kmh: Speed in km/h
        """
        if pixels_per_meter is None or pixels_per_meter <= 0:
            return None
        
        # pixels/frame -> meters/second -> km/h
        meters_per_frame = pixel_velocity / pixels_per_meter
        meters_per_second = meters_per_frame * self.fps
        km_per_hour = meters_per_second * 3.6
        
        return km_per_hour


class YOLOv8SpeedDetector:
    """YOLOv8 + ByteTrack + Speed Estimation"""
    
    def __init__(self, model_name='yolov8n.pt', fps=30):
        self.model_name = model_name  # 模型名称（如'yolov8n.pt'）
        self.model = None
        self.classes = []
        self.fps = fps
        
        # Track history
        self.track_history = {}
        self.frame_count = 0
        
        # Speed estimator
        self.speed_estimator = SpeedEstimator(fps)
        
        # Speed records for statistics
        self.speed_records = {}  # {track_id: [speed_kmh, ...]}
        
        # Speed smoothing cache - KEY for stable display
        self.smoothed_speeds = {}  # {track_id: smoothed_speed_kmh}
        self.smooth_factor = 0.3  # EMA factor (lower = smoother, 0.2-0.4 recommended)
        
        self.setup_model()
    
    def setup_model(self):
        """Setup YOLOv8 model"""
        try:
            # 使用model_config获取正确路径
            yolo_path = model_config.get_model_path(self.model_name)
            print(f"Loading YOLOv8 model from: {yolo_path}")
            
            # 直接加载（环境变量已设置，会自动下载到models/）
            self.model = YOLO(yolo_path)
            
            self.classes = list(self.model.names.values())
            print(f"[OK] YOLOv8 model loaded")
            print(f"[OK] ByteTrack tracker ready")
            print(f"[OK] Speed estimation enabled (object-size based)")
            
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            sys.exit(1)
    
    def track_frame(self, frame, conf_threshold=0.25, tracker='bytetrack'):
        """Detect, track, and estimate speed"""
        self.frame_count += 1
        
        results = self.model.track(
            frame, 
            persist=True,
            conf=conf_threshold,
            iou=0.5,
            tracker=f"{tracker}.yaml",
            verbose=False
        )
        
        tracks = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                    
                    track_id = int(boxes.id[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_name = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
                    
                    # Estimate pixel/meter ratio
                    px_per_m = self.speed_estimator.estimate_pixels_per_meter(class_name, w, h)
                    
                    track = {
                        'id': track_id,
                        'bbox': [x, y, w, h],
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'pixels_per_meter': px_per_m,
                        'speed_kmh': None
                    }
                    
                    # Update history first
                    self._update_track_history(track)
                    
                    # Calculate speed with smoothing
                    vx, vy = self.get_pixel_velocity(track_id, num_frames=10)  # Use more frames
                    pixel_speed = np.sqrt(vx**2 + vy**2)
                    
                    raw_speed = None
                    if px_per_m is not None and pixel_speed > 0.3:  # Lower threshold
                        raw_speed = self.speed_estimator.calculate_speed_kmh(pixel_speed, px_per_m)
                    
                    # Apply speed smoothing (EMA)
                    smoothed_speed = self._smooth_speed(track_id, raw_speed)
                    track['speed_kmh'] = smoothed_speed
                    
                    # Record speed for statistics
                    if smoothed_speed is not None and smoothed_speed < 200:
                        if track_id not in self.speed_records:
                            self.speed_records[track_id] = []
                        self.speed_records[track_id].append(smoothed_speed)
                    
                    tracks.append(track)
        
        return tracks
    
    def _update_track_history(self, track):
        """Update track history"""
        track_id = track['id']
        bbox = track['bbox']
        x, y, w, h = bbox
        center = (x + w // 2, y + h // 2)
        
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
    
    def _smooth_speed(self, track_id, raw_speed):
        """
        Apply Exponential Moving Average (EMA) smoothing to speed
        This prevents flickering/jumping speed display
        """
        if track_id not in self.smoothed_speeds:
            # First time seeing this track
            if raw_speed is not None:
                self.smoothed_speeds[track_id] = raw_speed
                return raw_speed
            return None
        
        prev_speed = self.smoothed_speeds[track_id]
        
        if raw_speed is None:
            # No new speed calculated, keep previous (with slight decay)
            # This prevents sudden disappearance of speed
            return prev_speed
        
        # Apply EMA: new_smooth = alpha * raw + (1-alpha) * prev
        smoothed = self.smooth_factor * raw_speed + (1 - self.smooth_factor) * prev_speed
        self.smoothed_speeds[track_id] = smoothed
        
        return smoothed
    
    def get_pixel_velocity(self, track_id, num_frames=10):
        """
        Calculate pixel velocity using more frames for stability
        """
        history = self.track_history.get(track_id, [])
        
        if len(history) < 3:  # Need at least 3 frames
            return (0, 0)
        
        recent = history[-min(num_frames, len(history)):]
        
        if len(recent) < 3:
            return (0, 0)
        
        # Use weighted average of velocities for smoother result
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent)):
            dx = recent[i]['center'][0] - recent[i-1]['center'][0]
            dy = recent[i]['center'][1] - recent[i-1]['center'][1]
            dt = recent[i]['frame'] - recent[i-1]['frame']
            if dt > 0:
                velocities_x.append(dx / dt)
                velocities_y.append(dy / dt)
        
        if not velocities_x:
            return (0, 0)
        
        # Use median to filter outliers (more robust than mean)
        vx = np.median(velocities_x)
        vy = np.median(velocities_y)
        
        return (vx, vy)
    
    def get_track_history(self, track_id):
        """Get track history"""
        return self.track_history.get(track_id, [])
    
    def get_average_speed(self, track_id):
        """Get average speed for a track"""
        speeds = self.speed_records.get(track_id, [])
        if len(speeds) > 0:
            return np.mean(speeds)
        return None
    
    def get_speed_statistics(self, tracks):
        """Get speed statistics for current frame"""
        speeds = [t['speed_kmh'] for t in tracks if t['speed_kmh'] is not None]
        
        if len(speeds) == 0:
            return {'max': 0, 'avg': 0, 'count': 0}
        
        return {
            'max': max(speeds),
            'avg': np.mean(speeds),
            'count': len(speeds)
        }


def process_video(input_path, output_path=None, show_video=True, conf_threshold=0.25, tracker='bytetrack'):
    """Process video with speed estimation"""
    print("=" * 60)
    print("YOLOv8 + ByteTrack + Speed Estimation")
    print("=" * 60)
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"[CONFIG] Confidence: {conf_threshold}, Tracker: {tracker}")
    print("=" * 60)
    
    # Initialize detector with correct FPS
    detector = YOLOv8SpeedDetector('yolov8n.pt', fps=fps)
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 128, 255),
        (255, 128, 128), (128, 255, 128), (128, 128, 255)
    ]
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            tracks = detector.track_frame(frame, conf_threshold, tracker)
            stats = detector.get_speed_statistics(tracks)
            
            # Debug output
            if frame_count % 100 == 0 or frame_count == 1:
                print(f"Frame {frame_count}/{total_frames}: {len(tracks)} objects")
                for t in tracks:
                    if t['speed_kmh'] is not None:
                        print(f"  ID{t['id']} {t['class_name']}: {t['speed_kmh']:.1f} km/h")
            
            # Draw annotations
            annotated_frame = frame.copy()
            
            for track in tracks:
                x, y, w, h = track['bbox']
                track_id = track['id']
                class_name = track['class_name']
                confidence = track['confidence']
                speed_kmh = track['speed_kmh']
                
                color = colors[track_id % len(colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                # Create label with speed
                if speed_kmh is not None:
                    label = f"ID{track_id} {class_name} {speed_kmh:.1f}km/h"
                else:
                    label = f"ID{track_id} {class_name}"
                
                # Draw label background
                font_scale = 0.6
                thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                cv2.rectangle(annotated_frame, (x, y - label_size[1] - 10),
                            (x + label_size[0] + 6, y), color, -1)
                cv2.putText(annotated_frame, label, (x + 3, y - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Draw trajectory
                history = detector.get_track_history(track_id)
                if len(history) > 1:
                    points = [h['center'] for h in history[-30:]]
                    for i in range(1, len(points)):
                        # Gradient color for trajectory
                        alpha = i / len(points)
                        line_color = tuple(int(c * alpha) for c in color)
                        cv2.line(annotated_frame, points[i-1], points[i], line_color, 2)
            
            # Draw statistics panel
            panel_y = 30
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            panel_y += 30
            cv2.putText(annotated_frame, f"Objects: {len(tracks)} | Tracker: ByteTrack",
                       (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Speed statistics panel
            if stats['count'] > 0:
                panel_y += 30
                cv2.putText(annotated_frame, f"Max Speed: {stats['max']:.1f} km/h",
                           (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                panel_y += 30
                cv2.putText(annotated_frame, f"Avg Speed: {stats['avg']:.1f} km/h",
                           (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Mode indicator
            cv2.putText(annotated_frame, "Mode: Object-Size Estimation",
                       (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
            
            if show_video:
                cv2.imshow('YOLOv8 + ByteTrack + Speed', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if out:
                out.write(annotated_frame)
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}%")
    
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    print(f"\n[OK] Processing complete, {frame_count} frames processed")
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='YOLOv8 + ByteTrack + Speed Estimation')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', default='output/output_speed.mp4', help='Output video file path')
    parser.add_argument('--no-display', action='store_true', help='Do not display video window')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--tracker', '-t', choices=['bytetrack', 'botsort'], default='bytetrack',
                        help='Tracker type')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else 'output', exist_ok=True)
    
    success = process_video(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_display,
        conf_threshold=args.conf,
        tracker=args.tracker
    )
    
    if success:
        print("\n" + "=" * 60)
        print("[DONE] Speed Estimation Complete!")
        print("=" * 60)
        print("[OK] ByteTrack high-precision tracking")
        print("[OK] Object-size based speed estimation")
        print("[OK] Real-time speed display (km/h)")
        print(f"[OK] Output: {args.output}")
        print("=" * 60)
        print("\n[NOTE] This is simplified version (assumes stationary camera)")
        print("[NEXT] Full version with RAFT optical flow for moving camera")


if __name__ == "__main__":
    main()
