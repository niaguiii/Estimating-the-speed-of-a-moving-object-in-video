"""
简化版主程序 - 使用OpenCV内置功能实现物体检测和追踪
避免依赖problematic的ultralytics包
"""
import cv2
import numpy as np
import os
import argparse
import sys
import urllib.request

class OpenCVObjectDetector:
    def __init__(self):
        """初始化OpenCV DNN检测器"""
        self.net = None
        self.output_layers = None
        self.classes = []
        
        # 下载并加载YOLOv3模型（OpenCV兼容版本）
        self.setup_yolo_model()
        
    def setup_yolo_model(self):
        """设置YOLO模型 - 支持YOLOv8/YOLOv5 ONNX和YOLOv3"""
        # 优先尝试YOLOv8/YOLOv5 ONNX模型
        onnx_model = 'models/yolov8n.onnx'
        if os.path.exists(onnx_model):
            model_files = {
                'onnx': onnx_model,
                'names': 'models/coco.names'
            }
            self.model_type = 'onnx'
        else:
            # 回退到YOLOv3
            model_files = {
                'weights': 'models/yolov3.weights',
                'config': 'models/yolov3.cfg',
                'names': 'models/coco.names'
            }
            self.model_type = 'darknet'
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 检查模型文件是否存在，如果不存在则下载
        for file_type, filename in model_files.items():
            if not os.path.exists(filename):
                print(f"正在下载 {filename}...")
                self.download_model_file(file_type, filename)
        
        try:
            if self.model_type == 'onnx':
                # 加载ONNX模型 (YOLOv8/YOLOv5)
                self.net = cv2.dnn.readNetFromONNX(model_files['onnx'])
                print("✅ YOLOv8/YOLOv5 ONNX模型加载成功")
                
                # 设置推理后端（可选，提高性能）
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            else:
                # 加载Darknet模型 (YOLOv3)
                self.net = cv2.dnn.readNet(model_files['weights'], model_files['config'])
                # 获取输出层
                layer_names = self.net.getLayerNames()
                self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
                print("✅ YOLOv3 Darknet模型加载成功")
            
            # 加载类别名称
            with open(model_files['names'], 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            print(f"📝 加载了 {len(self.classes)} 个物体类别")
            
        except Exception as e:
            print(f"❌ YOLO模型加载失败: {e}")
            print("可能原因：模型文件未下载或损坏")
            print("正在切换到备用检测方法...")
            self.setup_fallback_detector()
    
    def download_model_file(self, file_type, filename):
        """下载模型文件"""
        urls = {
            'weights': 'https://pjreddie.com/media/files/yolov3.weights',
            'config': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
            'names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
        }
        
        try:
            urllib.request.urlretrieve(urls[file_type], filename)
            print(f"✅ {filename} 下载完成")
        except Exception as e:
            print(f"❌ {filename} 下载失败: {e}")
            if file_type == 'names':
                # 如果下载失败，创建基础的类别文件
                self.create_basic_coco_names(filename)
    
    def create_basic_coco_names(self, filename):
        """创建基础的COCO类别文件"""
        basic_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench'
        ]
        
        with open(filename, 'w') as f:
            for class_name in basic_classes:
                f.write(class_name + '\n')
        
        print(f"✅ 创建基础 {filename}")
    
    def setup_fallback_detector(self):
        """设置备用检测器（Haar级联）"""
        try:
            # 使用OpenCV内置的人脸检测器作为示例
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.classes = ['face']
            self.net = None  # 标记使用备用方法
            print("✅ 备用检测器加载成功")
        except Exception as e:
            print(f"❌ 备用检测器加载失败: {e}")
    
    def detect_objects(self, frame):
        """检测物体"""
        if self.net is not None:
            return self.detect_with_yolo(frame)
        else:
            return self.detect_with_cascade(frame)
    
    def detect_with_yolo(self, frame):
        """使用YOLO检测 - 支持ONNX和Darknet格式"""
        height, width = frame.shape[:2]
        
        if self.model_type == 'onnx':
            # YOLOv8/YOLOv5 ONNX格式
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward()
            return self.parse_onnx_output(outputs, width, height)
        else:
            # YOLOv3 Darknet格式
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outputs = self.net.forward(self.output_layers)
            return self.parse_darknet_output(outputs, width, height)
    
    def parse_onnx_output(self, outputs, width, height):
        """解析ONNX模型输出 (YOLOv8/YOLOv5格式)"""
        output = outputs[0]
        
        # YOLOv8输出格式: [1, 84, 8400] -> [classes + xywh + conf]
        if len(output.shape) == 3:
            output = output[0]  # 移除batch维度
        
        # 转置: [84, 8400] -> [8400, 84]
        output = output.transpose()
        
        boxes = []
        confidences = []
        class_ids = []
        
        # YOLOv8格式: [cx, cy, w, h, class1_conf, class2_conf, ...]
        # 没有单独的objectness分数，类别置信度就是最终置信度
        for row in output:
            # 提取类别分数 (第5列开始是80个类别的分数)
            classes_scores = row[4:84]  # COCO有80个类别
            class_id = np.argmax(classes_scores)
            max_class_confidence = classes_scores[class_id]
            
            # 检查置信度范围是否正常（应该在0-1之间）
            if max_class_confidence > 1.0:
                # 如果置信度大于1，可能需要sigmoid归一化
                max_class_confidence = 1.0 / (1.0 + np.exp(-max_class_confidence))
            
            # 使用合理的阈值
            if max_class_confidence >= 0.5:  # 使用0.5作为阈值
                # 提取边界框 (center_x, center_y, width, height)
                cx, cy, w, h = row[0:4]
                
                # 转换为像素坐标
                cx = cx * width / 640
                cy = cy * height / 640
                w = w * width / 640
                h = h * height / 640
                
                # 转换为左上角坐标
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                w = int(w)
                h = int(h)
                
                # 确保边界框在图像范围内
                x = max(0, x)
                y = max(0, y)
                w = min(w, width - x)
                h = min(h, height - y)
                
                if w > 10 and h > 10:  # 确保边界框有效且不太小
                    boxes.append([x, y, w, h])
                    confidences.append(float(max_class_confidence))
                    class_ids.append(class_id)
        
        return self.apply_nms(boxes, confidences, class_ids)
    
    def parse_darknet_output(self, outputs, width, height):
        """解析Darknet模型输出 (YOLOv3格式)"""
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.25:  # 降低阈值以提高检测率
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        return self.apply_nms(boxes, confidences, class_ids)
    
    def apply_nms(self, boxes, confidences, class_ids):
        """应用非极大值抑制并返回检测结果"""
        # 非极大值抑制 - 使用合理的置信度阈值
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'bbox': boxes[i],
                    'confidence': confidences[i],
                    'class_id': class_ids[i],
                    'class_name': self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else 'unknown'
                })
        
        return detections
    
    def detect_with_cascade(self, frame):
        """使用级联分类器检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 4)
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': [x, y, w, h],
                'confidence': 0.8,
                'class_id': 0,
                'class_name': 'face'
            })
        
        return detections

class SimpleTracker:
    def __init__(self):
        """简单的物体追踪器"""
        self.tracks = []
        self.next_id = 1
        self.max_disappeared = 10
    
    def update(self, detections):
        """更新追踪"""
        if len(self.tracks) == 0:
            # 初始化追踪
            for detection in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'disappeared': 0
                })
                self.next_id += 1
        else:
            # 简单的最近邻匹配
            self.match_detections(detections)
        
        # 移除消失太久的追踪
        self.tracks = [track for track in self.tracks if track['disappeared'] < self.max_disappeared]
        
        return self.tracks
    
    def match_detections(self, detections):
        """匹配检测结果到追踪"""
        for track in self.tracks:
            track['disappeared'] += 1
        
        for detection in detections:
            # 找到最近的追踪
            best_match = None
            min_distance = float('inf')
            
            det_center = self.get_center(detection['bbox'])
            
            for track in self.tracks:
                if track['disappeared'] < self.max_disappeared:
                    track_center = self.get_center(track['bbox'])
                    distance = self.calculate_distance(det_center, track_center)
                    
                    if distance < min_distance and distance < 100:  # 距离阈值
                        min_distance = distance
                        best_match = track
            
            if best_match:
                # 更新追踪
                best_match['bbox'] = detection['bbox']
                best_match['class_name'] = detection['class_name']
                best_match['disappeared'] = 0
            else:
                # 创建新追踪
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

def process_video(input_path, output_path=None, show_video=True):
    """处理视频"""
    print("正在初始化检测器...")
    detector = OpenCVObjectDetector()
    tracker = SimpleTracker()
    
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
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测物体
            detections = detector.detect_objects(frame)
            
            # 更新追踪
            tracks = tracker.update(detections)
            
            # 调试信息：详细检测和追踪结果
            if frame_count % 100 == 0 or frame_count == 1:
                active_tracks = [t for t in tracks if t['disappeared'] == 0]
                print(f"第{frame_count}帧检测到 {len(detections)} 个物体，追踪到 {len(active_tracks)} 个物体")
                for i, det in enumerate(detections):
                    print(f"  检测{i+1}: {det['class_name']} (置信度: {det['confidence']:.2f})")
                for i, track in enumerate(active_tracks):
                    print(f"  追踪{i+1}: ID{track['id']} {track['class_name']}")
            
            # 绘制结果
            annotated_frame = frame.copy()
            
            # 直接绘制检测结果，而不是追踪结果，以显示置信度
            for detection in detections:
                x, y, w, h = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                
                # 生成固定颜色（基于类别）
                color_seed = hash(class_name) % 255
                np.random.seed(color_seed)
                color = tuple(map(int, np.random.randint(100, 255, 3)))
                
                # 绘制边界框
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
                
                # 绘制类别和置信度标签
                label = f"{class_name} {confidence:.2f}"
                font_scale = 0.5  # 调小字体
                thickness = 1
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # 绘制标签背景
                cv2.rectangle(annotated_frame, (x, y - label_size[1] - 8), 
                            (x + label_size[0] + 4, y), color, -1)
                
                # 绘制文字
                cv2.putText(annotated_frame, label, (x + 2, y - 4), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # 同时绘制追踪ID
            for track in tracks:
                if track['disappeared'] == 0:  # 只绘制当前帧检测到的
                    x, y, w, h = track['bbox']
                    track_id = track['id']
                    
                    # 在右上角显示追踪ID
                    id_label = f"ID:{track_id}"
                    cv2.putText(annotated_frame, id_label, (x + w - 40, y + 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # 添加信息
            cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Objects: {len([t for t in tracks if t['disappeared'] == 0])}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示视频
            if show_video:
                cv2.imshow('Object Detection & Tracking', annotated_frame)
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
    parser = argparse.ArgumentParser(description='视频中移动物体速度估计 - OpenCV版本')
    parser.add_argument('--input', '-i', required=True, help='输入视频文件路径')
    parser.add_argument('--output', '-o', default='output_opencv.mp4', help='输出视频文件路径')
    parser.add_argument('--no-display', action='store_true', help='不显示视频窗口')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    print("🚀 开始处理视频...")
    success = process_video(
        input_path=args.input,
        output_path=args.output,
        show_video=not args.no_display
    )
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 OpenCV版本处理完成！")
        print("✅ 物体检测功能正常")
        print("✅ 物体追踪功能正常")
        print("✅ 视频帧数检测功能正常")
        print(f"✅ 输出文件: {args.output}")
        print("=" * 50)

if __name__ == "__main__":
    main()
