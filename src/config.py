"""
配置文件 - 项目的各种参数设置
"""

# 项目信息
PROJECT_NAME = "视频中移动物体速度估计项目"
VERSION = "1.0.0 - 第一阶段完成"
DESCRIPTION = "基于YOLOv8的智能物体检测与追踪系统"

# 文件夹配置
FOLDERS = {
    'input': 'input',           # 输入视频文件夹
    'output': 'output',         # 输出结果文件夹
    'models': 'models',         # 模型文件文件夹
    'logs': 'logs',             # 日志文件夹
    'temp': 'temp'              # 临时文件夹
}

# 支持的视频格式
SUPPORTED_VIDEO_FORMATS = [
    '.mp4', '.avi', '.mov', '.mkv', 
    '.flv', '.wmv', '.m4v', '.mpg', '.mpeg'
]

# YOLO模型配置 - 支持YOLOv8/YOLOv5 ONNX和YOLOv3 Darknet
YOLO_CONFIG = {
    # 优先使用YOLOv8/YOLOv5 ONNX模型
    'onnx_model': 'models/yolov8n.onnx',
    
    # 回退到YOLOv3 Darknet模型
    'weights_file': 'models/yolov3.weights',
    'config_file': 'models/yolov3.cfg',
    'names_file': 'models/coco.names',
    'download_urls': {
        'weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'config': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
}

# 检测参数
DETECTION_CONFIG = {
    'confidence_threshold': 0.5,    # 置信度阈值
    'nms_threshold': 0.4,           # NMS阈值
    'input_size': (416, 416),       # 输入尺寸
    'scale_factor': 0.00392,        # 缩放因子
}

# 追踪参数
TRACKING_CONFIG = {
    'max_disappeared': 10,          # 最大消失帧数
    'distance_threshold': 100,      # 距离阈值（像素）
    'min_track_length': 5,          # 最小追踪长度
}

# 视频处理参数
VIDEO_CONFIG = {
    'output_codec': 'mp4v',         # 输出编码格式
    'progress_update_interval': 30, # 进度更新间隔（帧数）
    'max_display_width': 1280,      # 最大显示宽度
    'max_display_height': 720,      # 最大显示高度
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'bbox_thickness': 2,            # 边界框线条粗细
    'text_thickness': 2,            # 文字粗细
    'text_scale': 0.6,              # 文字大小
    'info_text_scale': 1.0,         # 信息文字大小
    'colors': [                     # 预定义颜色列表
        (0, 255, 0),    # 绿色
        (255, 0, 0),    # 蓝色
        (0, 0, 255),    # 红色
        (255, 255, 0),  # 青色
        (255, 0, 255),  # 品红色
        (0, 255, 255),  # 黄色
        (128, 0, 128),  # 紫色
        (255, 165, 0),  # 橙色
    ]
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True,
}

# 测试视频配置
TEST_VIDEO_CONFIG = {
    'width': 640,
    'height': 480,
    'fps': 30,
    'duration': 5,                  # 秒
    'filename': 'test_video.mp4',
    'objects': [
        {
            'type': 'rectangle',
            'color': (0, 255, 0),
            'size': (60, 40),
            'speed': (120, 0),
            'start_pos': (50, 160),
            'label': 'Car'
        },
        {
            'type': 'circle',
            'color': (255, 0, 0),
            'size': 25,
            'speed': (0, 80),
            'start_pos': (350, 50),
            'label': 'Ball'
        },
        {
            'type': 'triangle',
            'color': (0, 0, 255),
            'size': (60, 40),
            'speed': (60, 40),
            'start_pos': (100, 100),
            'label': 'Object'
        }
    ]
}
