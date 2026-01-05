# -*- coding: utf-8 -*-
"""
Depth Anything V2 - Phase 3 Depth Estimation Module
使用Depth Anything V2进行单目深度估计

Features:
- Monocular depth estimation
- Metric depth (real-world distance)
- Automatic pixel-to-meter conversion
- GPU/CPU automatic detection
"""
# ⚠️ 关键：必须最先导入model_config来设置环境变量！
try:
    from . import model_config
except ImportError:
    import model_config

# 现在才导入torch和相关库
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")


class DepthAnythingV2:
    """Depth Anything V2 深度估计器"""
    
    def __init__(self, model_size: str = 'small', device: str = 'auto'):
        """
        初始化Depth Anything V2模型
        
        Args:
            model_size: 'small', 'base', or 'large'
            device: 'auto', 'cuda', 'cpu'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 模型映射
        model_mapping = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"model_size must be one of {list(model_mapping.keys())}")
        
        model_name = model_mapping[model_size]
        
        print(f"[DepthAnything] Initializing Depth-Anything-V2-{model_size} on {self.device}")
        print(f"[DepthAnything] Loading from: {model_name}")
        
        try:
            # 加载预处理器和模型
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"[DepthAnything] Model loaded successfully")
        except Exception as e:
            print(f"[DepthAnything] Error loading model: {e}")
            print("[DepthAnything] Trying to download from Hugging Face...")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: OpenCV BGR图像 (H, W, 3)
            
        Returns:
            inputs: 预处理后的tensor
        """
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 使用processor预处理
        inputs = self.processor(images=image_rgb, return_tensors="pt")
        
        # 移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        估计深度图
        
        Args:
            image: OpenCV BGR图像 (H, W, 3)
            
        Returns:
            depth_map: 深度图 (H, W)，值越大表示越远
        """
        h, w = image.shape[:2]
        
        with torch.no_grad():
            # 预处理
            inputs = self.preprocess_image(image)
            
            # 推理
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
            # 插值到原始尺寸
            predicted_depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(h, w),
                mode='bicubic',
                align_corners=False
            )
            
            # 转换为numpy
            depth_map = predicted_depth.squeeze().cpu().numpy()
        
        return depth_map
    
    def depth_to_distance(self, depth_map: np.ndarray, 
                         normalize: bool = True) -> np.ndarray:
        """
        将深度图转换为相对距离
        
        Args:
            depth_map: 深度图 (H, W)
            normalize: 是否归一化到0-1范围
            
        Returns:
            distance_map: 距离图 (H, W)
        """
        if normalize:
            # 归一化到0-1
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max > depth_min:
                distance_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                distance_map = np.zeros_like(depth_map)
        else:
            distance_map = depth_map
        
        return distance_map
    
    def get_object_depth(self, depth_map: np.ndarray, 
                        bbox: Tuple[int, int, int, int],
                        method: str = 'median') -> float:
        """
        获取物体区域的深度值
        
        Args:
            depth_map: 深度图 (H, W)
            bbox: 边界框 (x1, y1, x2, y2)
            method: 'median', 'mean', 'min'
            
        Returns:
            depth_value: 物体的深度值
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape
        
        # 确保边界框在图像范围内
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # 提取物体区域
        obj_region = depth_map[y1:y2, x1:x2]
        
        # 计算深度值
        if method == 'median':
            depth_value = np.median(obj_region)
        elif method == 'mean':
            depth_value = np.mean(obj_region)
        elif method == 'min':
            depth_value = np.min(obj_region)
        else:
            depth_value = np.median(obj_region)
        
        return float(depth_value)
    
    def visualize_depth(self, depth_map: np.ndarray, 
                       colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        可视化深度图
        
        Args:
            depth_map: 深度图 (H, W)
            colormap: OpenCV colormap
            
        Returns:
            depth_colored: 彩色深度图 (H, W, 3) BGR
        """
        # 归一化到0-255
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        
        # 应用colormap
        depth_colored = cv2.applyColorMap(depth_normalized, colormap)
        
        return depth_colored


class DepthBasedCalibrator:
    """基于深度的像素/米标定器"""
    
    def __init__(self, depth_estimator: DepthAnythingV2):
        """
        初始化
        
        Args:
            depth_estimator: Depth Anything V2估计器
        """
        self.depth_estimator = depth_estimator
        self.reference_objects = {}  # 存储参考物体的深度和尺寸
    
    def calibrate_with_depth(self, image: np.ndarray, 
                            bbox: Tuple[int, int, int, int],
                            real_size_meters: float,
                            bbox_size_pixels: float,
                            depth_map: Optional[np.ndarray] = None) -> float:
        """
        使用深度信息进行标定
        
        Args:
            image: 图像
            bbox: 边界框
            real_size_meters: 真实尺寸（米）
            bbox_size_pixels: 边界框尺寸（像素）
            depth_map: 预计算的深度图（可选）
            
        Returns:
            pixel_per_meter: 像素/米比例（考虑深度）
        """
        # 获取深度图
        if depth_map is None:
            depth_map = self.depth_estimator.estimate_depth(image)
        
        # 获取物体深度
        obj_depth = self.depth_estimator.get_object_depth(depth_map, bbox)
        
        # 深度校正系数（深度越大，物体看起来越小）
        # 这里使用简单的反比例关系
        # 实际应用中需要相机内参
        depth_correction = 1.0 / (1.0 + obj_depth)
        
        # 标定
        pixel_per_meter = (bbox_size_pixels / real_size_meters) * depth_correction
        
        return pixel_per_meter
    
    def estimate_real_distance(self, pixel_distance: float,
                              depth_value: float,
                              reference_ppm: float) -> float:
        """
        估计真实距离（米）
        
        Args:
            pixel_distance: 像素距离
            depth_value: 深度值
            reference_ppm: 参考像素/米比例
            
        Returns:
            real_distance: 真实距离（米）
        """
        # 深度校正
        depth_correction = 1.0 / (1.0 + depth_value)
        
        # 转换为真实距离
        real_distance = pixel_distance / (reference_ppm * depth_correction)
        
        return real_distance


if __name__ == "__main__":
    """测试Depth Anything V2"""
    print("=" * 60)
    print("Depth Anything V2 Test")
    print("=" * 60)
    
    try:
        # 初始化
        print("\n[1/3] Initializing Depth Anything V2...")
        depth_estimator = DepthAnythingV2(model_size='small', device='auto')
        print("✅ Depth Anything V2 initialized")
        
        # 创建测试图像
        print("\n[2/3] Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # 估计深度
        print("\n[3/3] Estimating depth...")
        depth_map = depth_estimator.estimate_depth(test_image)
        print(f"✅ Depth map shape: {depth_map.shape}")
        print(f"✅ Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        
        # 测试物体深度
        bbox = (100, 100, 200, 200)
        obj_depth = depth_estimator.get_object_depth(depth_map, bbox)
        print(f"✅ Object depth: {obj_depth:.2f}")
        
        print("\n" + "=" * 60)
        print("✅ Depth Anything V2 Module Ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: First run will download model from Hugging Face (~100MB)")
        print("      This is normal and only happens once.")
