# -*- coding: utf-8 -*-
"""
统一模型配置模块
所有AI模型的下载路径统一管理
必须在导入任何深度学习库之前导入此模块
"""
import os
import sys

# ⚠️ 关键：必须在导入torch/ultralytics/transformers之前设置环境变量！

# 获取项目根目录（无论从哪里运行都正确）
def get_project_root():
    """获取项目根目录"""
    cwd = os.getcwd()

    # 检查是否在src目录
    if os.path.basename(cwd) == 'src':
        return os.path.dirname(cwd)

    # 检查是否在scripts目录
    if 'scripts' in cwd:
        # 找到scripts的位置，返回其父目录
        parts = cwd.split(os.sep)
        if 'scripts' in parts:
            idx = parts.index('scripts')
            return os.sep.join(parts[:idx])

    # 检查是否在项目根目录（包含models文件夹）
    if os.path.exists(os.path.join(cwd, 'models')):
        return cwd

    # 默认返回当前目录
    return cwd


# 设置项目根目录和模型目录
PROJECT_ROOT = get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 设置所有模型缓存目录（必须在导入深度学习库之前）
# PyTorch / RAFT 模型
os.environ['TORCH_HOME'] = MODEL_DIR
os.environ['TORCH_HUB'] = os.path.join(MODEL_DIR, 'hub')

# Hugging Face / Depth Anything V2 模型
os.environ['HF_HOME'] = MODEL_DIR
os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR
os.environ['HF_HUB_CACHE'] = MODEL_DIR

# Ultralytics / YOLOv8 模型
os.environ['YOLO_CONFIG_DIR'] = MODEL_DIR
# ultralytics会自动使用TORCH_HOME下载模型

# 打印配置信息（调试用）
_DEBUG = os.environ.get('MODEL_DEBUG', '0') == '1'
if _DEBUG:
    print(f"[ModelConfig] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[ModelConfig] MODEL_DIR: {MODEL_DIR}")
    print(f"[ModelConfig] TORCH_HOME: {os.environ['TORCH_HOME']}")
    print(f"[ModelConfig] HF_HOME: {os.environ['HF_HOME']}")
    print(f"[ModelConfig] YOLO_CONFIG_DIR: {os.environ['YOLO_CONFIG_DIR']}")


def get_model_path(model_name):
    """
    获取模型文件的完整路径
    
    Args:
        model_name: 模型文件名，如 'yolov8n.pt', 'yolov8s.pt'
    
    Returns:
        完整的模型路径
    """
    return os.path.join(MODEL_DIR, model_name)


def download_yolov8(model_name='yolov8n.pt'):
    """
    下载YOLOv8模型到models目录
    
    Args:
        model_name: 模型名称，如 'yolov8n.pt', 'yolov8s.pt'
    
    Returns:
        模型路径
    """
    model_path = get_model_path(model_name)
    
    if os.path.exists(model_path):
        print(f"[YOLOv8] Model exists: {model_path}")
        return model_path
    
    print(f"[YOLOv8] Downloading {model_name} to models/...")
    
    try:
        # ultralytics会自动下载到TORCH_HOME
        from ultralytics import YOLO
        model = YOLO(model_name)  # 会下载到models/目录
        
        # 如果模型被下载到其他位置，移动到models/
        default_locations = [
            model_name,  # 当前目录
            os.path.join(os.path.expanduser('~'), '.cache', 'ultralytics', model_name),
        ]
        
        for loc in default_locations:
            if os.path.exists(loc) and not os.path.samefile(os.path.dirname(loc), MODEL_DIR):
                import shutil
                print(f"[YOLOv8] Moving {loc} -> {model_path}")
                shutil.move(loc, model_path)
                break
        
        if os.path.exists(model_path):
            print(f"[YOLOv8] ✅ Model downloaded: {model_path}")
        else:
            print(f"[YOLOv8] ⚠️  Model location may vary, check models/ directory")
        
        return model_path
        
    except Exception as e:
        print(f"[YOLOv8] ❌ Download failed: {e}")
        return None


# 模型文件列表
MODELS = {
    'yolov8n': 'yolov8n.pt',      # YOLOv8 Nano (~6MB)
    'yolov8s': 'yolov8s.pt',      # YOLOv8 Small (~22MB)
    'yolov8m': 'yolov8m.pt',      # YOLOv8 Medium (~52MB)
    'yolov8l': 'yolov8l.pt',      # YOLOv8 Large (~87MB)
    'yolov8x': 'yolov8x.pt',      # YOLOv8 XLarge (~137MB)
    'coco_names': 'coco.names',    # COCO类别名称
}


if __name__ == "__main__":
    """测试模型配置"""
    print("=" * 70)
    print("模型配置测试")
    print("=" * 70)
    print(f"\n项目根目录: {PROJECT_ROOT}")
    print(f"模型目录: {MODEL_DIR}")
    print(f"目录是否存在: {os.path.exists(MODEL_DIR)}")
    
    print("\n环境变量:")
    print(f"  TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"  YOLO_CONFIG_DIR: {os.environ.get('YOLO_CONFIG_DIR')}")
    
    print("\n模型路径:")
    for key, filename in MODELS.items():
        path = get_model_path(filename)
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"  {exists} {key}: {path}")
    
    print("\n" + "=" * 70)
    print("配置正确！")
    print("=" * 70)
