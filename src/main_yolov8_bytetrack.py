# -*- coding: utf-8 -*-
"""
YOLOv8 + ByteTrack - Ultralytics Native Integration
Phase 2 Core Implementation

Features:
- Built-in ByteTrack/BoT-SORT tracking
- Kalman filter motion prediction
- High precision tracking (80-90%)
- Track history for speed calculation
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


class YOLOv8ByteTrackDetector:
    """YOLOv8 + ByteTrack Integrated Detector/Tracker"""
    
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize detector"""
        self.model_name = f"models/{model_name}"
        self.model = None
        self.classes = []
        
        # Track history storage (for speed calculation)
        self.track_history = {}
        self.frame_count = 0
        
        self.setup_model()
    
    def setup_model(self):
        """Setup YOLOv8 model"""
        try:
            # 使用model_config获取正确路径
            yolo_path = model_config.get_model_path('yolov8n.pt')
            print(f"Loading YOLOv8 model from: {yolo_path}")
            
            # 直接加载（环境变量已设置，会自动下载到models/）
            self.model = YOLO(yolo_path)
            
            self.classes = list(self.model.names.values())
            print(f"[OK] YOLOv8 model loaded")
            print(f"[INFO] {len(self.classes)} object classes available")
            print(f"[READY] ByteTrack tracker ready")
            
        except Exception as e:
            print(f"[ERROR] Model loading failed: {e}")
            sys.exit(1)
    
    def track_frame(self, frame, conf_threshold=0.25, tracker='bytetrack'):
        """
        Detect and track objects in a single frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold
            tracker: Tracker type ('bytetrack' or 'botsort')
        
        Returns:
            tracks: List of tracking results
        """
        self.frame_count += 1
        
        # Use ultralytics built-in tracking
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
                    
                    track = {
                        'id': track_id,
                        'bbox': [x, y, w, h],
                        'class_name': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'disappeared': 0
                    }
                    tracks.append(track)
                    self._update_track_history(track)
        
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
    
    def get_pixel_velocity(self, track_id, num_frames=5):
        """
        Calculate pixel velocity (for real speed calculation later)
        
        Returns:
            (vx, vy): Pixel velocity in pixels/frame
        """
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
    
    def get_track_history(self, track_id):
        """Get track history for specific ID"""
        return self.track_history.get(track_id, [])


def process_video(input_path, output_path=None, show_video=True, conf_threshold=0.25, tracker='bytetrack'):
    """
    Process video with YOLOv8 + ByteTrack
    """
    print("=" * 60)
    print("YOLOv8 + ByteTrack Video Processing")
    print("=" * 60)
    
    detector = YOLOv8ByteTrackDetector('yolov8n.pt')
    
    print(f"[CONFIG] Confidence threshold: {conf_threshold}")
    print(f"[CONFIG] Tracker: {tracker}")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 60)
    print("[VIDEO INFO]")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f}s")
    print("=" * 60)
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
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
            
            if frame_count % 100 == 0 or frame_count == 1:
                print(f"Frame {frame_count}: Tracking {len(tracks)} objects")
                for track in tracks:
                    vx, vy = detector.get_pixel_velocity(track['id'])
                    speed_px = np.sqrt(vx**2 + vy**2)
                    print(f"  ID{track['id']}: {track['class_name']} "
                          f"(conf: {track['confidence']:.2f}, px_speed: {speed_px:.1f} px/frame)")
            
            annotated_frame = frame.copy()
            
            for track in tracks:
                x, y, w, h = track['bbox']
                track_id = track['id']
                class_name = track['class_name']
                confidence = track['confidence']
                
                color = colors[track_id % len(colors)]
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                vx, vy = detector.get_pixel_velocity(track_id)
                speed_px = np.sqrt(vx**2 + vy**2)
                
                label = f"ID{track_id} {class_name} {confidence:.2f}"
                if speed_px > 0.5:
                    label += f" ({speed_px:.1f}px/f)"
                
                font_scale = 0.5
                thickness = 1
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                cv2.rectangle(annotated_frame, (x, y - label_size[1] - 8),
                            (x + label_size[0] + 4, y), color, -1)
                cv2.putText(annotated_frame, label, (x + 2, y - 4),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Draw trajectory
                history = detector.get_track_history(track_id)
                if len(history) > 1:
                    points = [h['center'] for h in history[-20:]]
                    for i in range(1, len(points)):
                        cv2.line(annotated_frame, points[i-1], points[i], color, 2)
            
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len(tracks)} | Tracker: {tracker.upper()}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if show_video:
                cv2.imshow('YOLOv8 + ByteTrack', annotated_frame)
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
    parser = argparse.ArgumentParser(description='YOLOv8 + ByteTrack Video Object Tracking')
    parser.add_argument('--input', '-i', required=True, help='Input video file path')
    parser.add_argument('--output', '-o', default='output_bytetrack.mp4', help='Output video file path')
    parser.add_argument('--no-display', action='store_true', help='Do not display video window')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='Confidence threshold (default 0.25)')
    parser.add_argument('--tracker', '-t', choices=['bytetrack', 'botsort'], default='bytetrack',
                        help='Tracker type: bytetrack or botsort (default: bytetrack)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return
    
    success = process_video(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_display,
        conf_threshold=args.conf,
        tracker=args.tracker
    )
    
    if success:
        print("\n" + "=" * 60)
        print("[DONE] YOLOv8 + ByteTrack processing complete!")
        print("=" * 60)
        print("[OK] Using ultralytics built-in ByteTrack")
        print("[OK] Kalman filter motion prediction")
        print("[OK] Track history recorded (ready for speed calculation)")
        print(f"[OK] Output file: {args.output}")
        print("=" * 60)
        print("\n[NEXT] Integrate RAFT optical flow + Metric Depth for real speed calculation")


if __name__ == "__main__":
    main()
