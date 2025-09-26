"""
YOLOv8原生版本 - 使用ultralytics库
更准确的检测结果和更好的性能
"""
import cv2
import numpy as np
import os
import argparse
import sys
from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_name='yolov8n.pt'):
        """初始化YOLOv8检测器"""
        self.model_name = f"models/{model_name}"  # 从models文件夹加载
        self.model = None
        self.setup_model()
        
    def setup_model(self):
        """设置YOLOv8模型"""
        try:
            # 确保models目录存在
            os.makedirs('models', exist_ok=True)
            
            print(f"正在加载YOLOv8模型: {self.model_name}")
            
            # 如果模型文件不存在，会自动下载到指定位置
            if not os.path.exists(self.model_name):
                print(f"模型文件不存在，将自动下载到: {self.model_name}")
                # 先加载到当前目录，然后移动到models文件夹
                temp_model = YOLO('yolov8n.pt')
                temp_model.save(self.model_name)
                self.model = YOLO(self.model_name)
            else:
                self.model = YOLO(self.model_name)
                
            print("✅ YOLOv8原生模型加载成功")
            
            # 获取类别名称
            self.classes = list(self.model.names.values())
            print(f"📝 加载了 {len(self.classes)} 个物体类别")
            
        except Exception as e:
            print(f"❌ YOLOv8模型加载失败: {e}")
            print("尝试从默认位置加载...")
            try:
                self.model = YOLO('yolov8n.pt')
                self.classes = list(self.model.names.values())
                print("✅ 使用默认位置的模型加载成功")
            except Exception as e2:
                print(f"❌ 完全加载失败: {e2}")
                sys.exit(1)
    
    def detect_objects(self, frame, conf_threshold=0.25):
        """针对单帧图像检测物体"""
        try:
            # 使用YOLOv8进行检测
            results = self.model(frame, verbose=False, conf=conf_threshold, iou=0.4)
            
            detections = []
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes
                        
                        # 提取检测结果
                        for i in range(len(boxes)):
                            # 获取边界框坐标 (xyxy格式)
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            
                            # 转换为xywh格式
                            x = int(x1)
                            y = int(y1)
                            w = int(x2 - x1)
                            h = int(y2 - y1)
                            
                            # 获取置信度和类别ID
                            confidence = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            
                            # 使用传入的置信度阈值
                            if confidence >= conf_threshold:
                                class_name = self.classes[class_id] if class_id < len(self.classes) else 'unknown'
                                
                                detections.append({
                                    'bbox': [x, y, w, h],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'class_name': class_name
                                })
            
            return detections
            
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
            return []

class SimpleTracker:
    def __init__(self):
        """简单的物体追踪器"""
        self.tracks = []      # 当前追踪列表
        self.next_id = 1    # 下一个ID生成器
        self.max_disappeared = 30 # 最大容忍消失帧数
    
    #会调用YOLO检测到的detection
    def update(self, detections):
        """更新追踪"""
        if len(self.tracks) == 0:
            # 情况1: 首次检测，初始化追踪
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],  # 最后已知位置
                    'class_name': detection['class_name'],  # 类别
                    'disappeared': 0 #消失计数器，用来处理物体暂时消失的情况
                })
                self.next_id += 1
        else:
            # 情况2: 现有追踪，进行最近邻匹配
            self.match_detections(detections)
        
        # 移除消失太久的追踪
        self.tracks = [track for track in self.tracks 
                    if track['disappeared'] < self.max_disappeared] 
        
        return self.tracks
    
    def match_detections(self, detections):
        """匹配检测结果到追踪"""
        # 步骤1: 所有追踪的消失计数+1
        for track in self.tracks:
            track['disappeared'] += 1
        
        # 步骤2: 为每个新检测找最佳匹配
        for detection in detections:
            # 找到最近的追踪
            best_match = None
            min_distance = float('inf')
            det_center = self.get_center(detection['bbox'])
            
            # 计算与所有追踪的距离
            for track in self.tracks:
                if track['disappeared'] < self.max_disappeared:
                    track_center = self.get_center(track['bbox'])
                    distance = self.calculate_distance(det_center, track_center)
                    
                    # 找最近且距离<100像素的追踪
                    if distance < min_distance and distance < 100:  # 距离阈值
                        min_distance = distance
                        best_match = track
            
            if best_match:
                # 找到匹配：更新追踪
                best_match['bbox'] = detection['bbox']
                best_match['class_name'] = detection['class_name']
                best_match['disappeared'] = 0 # 找到了，充值消失计数器
            else:
                # 没找到匹配：创建新追踪
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'disappeared': 0
                })
                self.next_id += 1
    
    def get_center(self, bbox):
        """获取边界框中心点"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def calculate_distance(self, point1, point2):
        """计算两点距离"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_video(input_path, output_path=None, show_video=True, conf_threshold=0.25):
    """处理视频"""
    print("正在初始化YOLOv8检测器...")
    detector = YOLOv8Detector('yolov8n.pt')  # 使用nano版本，速度快
    tracker = SimpleTracker()
    
    print(f"使用置信度阈值: {conf_threshold}")
    if conf_threshold < 0.1:
        print("⚠️  注意：使用低置信度阈值，可能检测到较多误报")
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ 无法打开视频文件: {input_path}")
        return False
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("=" * 50)
    print("视频信息")
    print("=" * 50)
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps} FPS")
    print(f"总帧数: {total_frames}")
    print(f"时长: {total_frames/fps:.2f} 秒")
    print("=" * 50)
    
    # 设置输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    #逐帧处理的主循环
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测物体
            detections = detector.detect_objects(frame, conf_threshold)
            
            # 更新追踪
            tracks = tracker.update(detections)
            
            # 调试信息：详细检测和追踪结果
            if frame_count % 100 == 0 or frame_count == 1:
                active_tracks = [t for t in tracks if t['disappeared'] == 0]
                print(f"第{frame_count}帧检测到 {len(detections)} 个物体，追踪到 {len(active_tracks)} 个物体")
                
                # 调试：打印原始检测结果
                raw_results = detector.model(frame, verbose=False, conf=0.01, iou=0.5)  # 极低阈值用于调试
                if raw_results and len(raw_results) > 0:
                    for result in raw_results:
                        if result.boxes is not None:
                            print(f"  原始检测数量: {len(result.boxes)}")
                            if len(result.boxes) > 0:
                                max_conf = float(result.boxes.conf.max())
                                print(f"  最高置信度: {max_conf:.4f}")
                
                for i, det in enumerate(detections):
                    print(f"  检测{i+1}: {det['class_name']} (置信度: {det['confidence']:.3f})")
                for i, track in enumerate(active_tracks):
                    print(f"  追踪{i+1}: ID{track['id']} {track['class_name']}")
            
            # 绘制结果
            annotated_frame = frame.copy()
            
            # 绘制检测结果，显示类别和置信度
            for i, detection in enumerate(detections):
                x, y, w, h = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # 生成更好看的颜色（基于类别，但确保不会太暗）
                color_seed = hash(class_name) % 255
                np.random.seed(color_seed)
                # 确保颜色明亮且对比度高
                base_colors = [
                    (255, 0, 0), (0, 255, 0), (0, 0, 255),     # 红绿蓝
                    (255, 255, 0), (255, 0, 255), (0, 255, 255), # 青品黄
                    (255, 128, 0), (255, 0, 128), (128, 255, 0), # 橙色系
                    (0, 128, 255), (128, 0, 255), (255, 128, 128) # 其他
                ]
                color = base_colors[color_seed % len(base_colors)]
                
                # 绘制边界框（加粗一点）
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 3)
                
                # 绘制类别和置信度标签
                label = f"{class_name} {confidence:.3f}"
                font_scale = 0.6  # 稍微大一点的字体
                thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # 绘制标签背景（稍微大一点的背景）
                bg_x1 = x
                bg_y1 = y - label_size[1] - 12
                bg_x2 = x + label_size[0] + 8
                bg_y2 = y - 2
                
                # 确保背景在图像范围内
                bg_y1 = max(0, bg_y1)
                
                # 绘制半透明背景
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # 绘制边框
                cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
                
                # 绘制文字（使用白色确保可见性）
                text_color = (255, 255, 255)  # 纯白色文字
                cv2.putText(annotated_frame, label, (x + 4, y - 6), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            
            # 绘制追踪ID（更美观的方式）
            for track in tracks:
                if track['disappeared'] == 0:  # 只绘制当前帧检测到的
                    x, y, w, h = track['bbox']
                    track_id = track['id']
                    class_name = track['class_name']
                    
                    # 生成ID专用颜色（基于ID号）
                    id_color_seed = track_id % 6
                    id_colors = [
                        (255, 255, 0),   # 黄色
                        (0, 255, 255),   # 青色
                        (255, 0, 255),   # 品红
                        (255, 128, 0),   # 橙色
                        (128, 255, 0),   # 草绿
                        (0, 128, 255)    # 天蓝
                    ]
                    id_color = id_colors[id_color_seed]
                    
                    # 在右上角显示追踪ID
                    id_label = f"ID{track_id}"
                    id_font_scale = 0.5
                    id_thickness = 2
                    id_label_size = cv2.getTextSize(id_label, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, id_thickness)[0]
                    
                    # ID标签位置（右上角）
                    id_x = x + w - id_label_size[0] - 8
                    id_y = y + id_label_size[1] + 8
                    
                    # 绘制ID背景
                    id_bg_x1 = id_x - 4
                    id_bg_y1 = id_y - id_label_size[1] - 4
                    id_bg_x2 = id_x + id_label_size[0] + 4
                    id_bg_y2 = id_y + 4
                    
                    # 半透明ID背景
                    id_overlay = annotated_frame.copy()
                    cv2.rectangle(id_overlay, (id_bg_x1, id_bg_y1), (id_bg_x2, id_bg_y2), id_color, -1)
                    cv2.addWeighted(id_overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
                    
                    # ID边框
                    cv2.rectangle(annotated_frame, (id_bg_x1, id_bg_y1), (id_bg_x2, id_bg_y2), id_color, 1)
                    
                    # 绘制ID文字（黑色文字在亮色背景上）
                    cv2.putText(annotated_frame, id_label, (id_x, id_y - 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (0, 0, 0), id_thickness)
            
            # 添加信息
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len([t for t in tracks if t['disappeared'] == 0])}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示视频
            if show_video:
                cv2.imshow('YOLOv8 Object Detection & Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 保存输出
            if output_path:
                out.write(annotated_frame)
            
            # 显示进度
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"处理进度: {progress:.1f}%")
    
    finally:
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    print(f"✅ 处理完成，共处理 {frame_count} 帧")
    return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频中移动物体速度估计 - YOLOv8原生版本')
    parser.add_argument('--input', '-i', required=True, help='输入视频文件路径')
    parser.add_argument('--output', '-o', default='output_yolov8_native.mp4', help='输出视频文件路径')
    parser.add_argument('--no-display', action='store_true', help='不显示视频窗口')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='YOLOv8模型 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)')
    parser.add_argument('--conf', '-c', type=float, default=0.25, help='置信度阈值 (0.01-0.9, 默认0.25)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    print("开始处理视频...")
    print(f"使用模型: {args.model}")
    
    success = process_video(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_display,
        conf_threshold=args.conf
    )
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 YOLOv8原生版本处理完成！")
        print("✅ 使用ultralytics原生YOLOv8")
        print("✅ 更准确的物体检测")
        print("✅ 更丰富的模型选择")
        print(f"✅ 输出文件: {args.output}")
        print("=" * 50)

if __name__ == "__main__":
    main()
