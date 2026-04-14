# -*- coding: utf-8 -*-
"""
Metric3D v2 - 绝对深度估计模块
使用Metric3D v2进行度量深度估计（输出真实距离，单位：米）

Features:
- Metric depth estimation (直接输出米数)
- Zero-shot prediction
- GPU/CPU automatic detection
- Camera intrinsics support
"""
# ⚠️ 关键：必须最先导入model_config来设置环境变量！
try:
    from . import model_config
except ImportError:
    import model_config

# 现在才导入torch和相关库
import math
import types
import torch
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class Metric3Dv2:
    """Metric3D v2 度量深度估计器（绝对深度，单位：米）"""
    
    def __init__(self, model_size: str = 'small', device: str = 'auto'):
        """
        初始化Metric3D v2模型
        
        Args:
            model_size: 'small', 'large', 'giant2'
            device: 'auto', 'cuda', 'cpu'
        """
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 模型映射
        model_mapping = {
            'small': 'metric3d_vit_small',
            'large': 'metric3d_vit_large',
            'giant2': 'metric3d_vit_giant2'
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"model_size must be one of {list(model_mapping.keys())}")
        
        model_name = model_mapping[model_size]
        
        print(f"[Metric3D] Initializing Metric3D-v2-{model_size} on {self.device}")
        print(f"[Metric3D] Loading model: {model_name}")
        
        try:
            # 从PyTorch Hub加载模型
            self.model = torch.hub.load(
                'yvanyin/metric3d', 
                model_name, 
                pretrain=True,
                trust_repo=True
            )
            self._patch_cpu_compatibility()
            self.model = self.model.to(self.device)
            self.model.eval()
            self._available = True
            print(f"[Metric3D] Model loaded successfully")
            
        except Exception as e:
            print(f"[Metric3D] Error loading model: {e}")
            print("[Metric3D] Trying to download from GitHub...")
            self._available = False
            self.model = None
            raise
        
        self.camera_intrinsics = None
        self.default_focal_length = 1000.0  # 默认焦距
    
    def is_available(self) -> bool:
        """检查模型是否已成功加载"""
        return getattr(self, '_available', False)
        
    def _patch_cpu_compatibility(self):
        """Patch upstream Metric3D modules that hardcode CUDA-only tensors."""
        if self.model is None or self.device.type != 'cpu':
            return

        patched = 0

        for module in self.model.modules():
            if hasattr(module, 'get_bins'):
                def _get_bins(this, bins_num, _device=self.device):
                    depth_bins_vec = torch.linspace(
                        math.log(this.min_val),
                        math.log(this.max_val),
                        bins_num,
                        device=_device,
                    )
                    return torch.exp(depth_bins_vec)

                module.get_bins = types.MethodType(_get_bins, module)
                patched += 1

            if hasattr(module, 'create_mesh_grid'):
                def _create_mesh_grid(
                    this,
                    height,
                    width,
                    batch,
                    device=None,
                    set_buffer=True,
                    _device=self.device,
                ):
                    actual_device = device if device is not None else _device
                    if isinstance(actual_device, str):
                        actual_device = torch.device(actual_device)
                    y, x = torch.meshgrid(
                        [
                            torch.arange(0, height, dtype=torch.float32, device=actual_device),
                            torch.arange(0, width, dtype=torch.float32, device=actual_device),
                        ],
                        indexing='ij',
                    )
                    meshgrid = torch.stack((x, y))
                    return meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)

                module.create_mesh_grid = types.MethodType(_create_mesh_grid, module)
                patched += 1

        if patched:
            print(f"[Metric3D] Applied {patched} CPU compatibility patch(es)")

    def set_camera_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """
        设置相机内参
        
        Args:
            fx: 焦距x（像素）
            fy: 焦距y（像素）
            cx: 主点x坐标
            cy: 主点y坐标
        """
        self.camera_intrinsics = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
        print(f"[Metric3D] Camera intrinsics set: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    def estimate_camera_intrinsics(self, image_width: int, image_height: int, 
                                   fov_degrees: float = 60.0) -> Dict[str, float]:
        """
        估算相机内参（如果没有实际内参）
        
        Args:
            image_width: 图像宽度
            image_height: 图像高度
            fov_degrees: 水平视场角（度），默认60度
            
        Returns:
            intrinsics: 相机内参字典
        """
        # 根据视场角估算焦距
        # fx = width / (2 * tan(fov/2))
        import math
        fov_rad = math.radians(fov_degrees)
        fx = image_width / (2 * math.tan(fov_rad / 2))
        fy = fx  # 假设像素是正方形
        
        cx = image_width / 2
        cy = image_height / 2
        
        intrinsics = {
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy
        }
        
        print(f"[Metric3D] Estimated intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        print(f"[Metric3D] (Based on FOV={fov_degrees}°)")
        
        return intrinsics
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: OpenCV BGR图像 (H, W, 3)
            
        Returns:
            rgb_tensor: 预处理后的RGB tensor
        """
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0, 1]
        image_rgb = image_rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        
        # 移到设备
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def estimate_depth(self, image: np.ndarray, 
                      intrinsics: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        估计深度图（绝对深度，单位：米）
        
        Args:
            image: OpenCV BGR图像 (H, W, 3)
            intrinsics: 相机内参 {'fx', 'fy', 'cx', 'cy'}，可选
            
        Returns:
            depth_map: 深度图 (H, W)，单位：米
        """
        h, w = image.shape[:2]
        
        # 如果没有提供内参，使用默认或估算
        if intrinsics is None:
            if self.camera_intrinsics is None:
                # 自动估算
                intrinsics = self.estimate_camera_intrinsics(w, h)
                self.camera_intrinsics = intrinsics
            else:
                intrinsics = self.camera_intrinsics
        
        with torch.no_grad():
            # 预处理
            rgb_tensor = self.preprocess_image(image)
            
            # 构建输入字典
            input_dict = {
                'input': rgb_tensor,
                'intrinsic': torch.tensor([
                    [intrinsics['fx'], 0, intrinsics['cx']],
                    [0, intrinsics['fy'], intrinsics['cy']],
                    [0, 0, 1]
                ]).unsqueeze(0).to(self.device).float()
            }
            
            # 推理
            pred_depth, confidence, output_dict = self.model.inference(input_dict)
            
            # 转换为numpy（单位：米）
            depth_map = pred_depth.squeeze().cpu().numpy()
            
            # 调整到原始尺寸
            if depth_map.shape != (h, w):
                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return depth_map
    
    def get_object_depth(self, depth_map: np.ndarray, 
                        bbox: Tuple[int, int, int, int],
                        method: str = 'median') -> float:
        """
        获取物体区域的深度值（米）
        
        Args:
            depth_map: 深度图 (H, W)，单位：米
            bbox: 边界框 (x1, y1, x2, y2)
            method: 'median', 'mean', 'min'
            
        Returns:
            depth_value: 物体的深度值（米）
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
                       max_depth: float = 50.0,
                       colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
        """
        可视化深度图
        
        Args:
            depth_map: 深度图 (H, W)，单位：米
            max_depth: 最大深度（米），用于归一化
            colormap: OpenCV colormap
            
        Returns:
            depth_colored: 彩色深度图 (H, W, 3) BGR
        """
        # 归一化到0-255
        depth_normalized = np.clip(depth_map / max_depth, 0, 1)
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # 应用colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)
        
        return depth_colored
    
    def pixel_to_3d(self, x_pixel: float, y_pixel: float, depth: float,
                    intrinsics: Optional[Dict[str, float]] = None) -> Tuple[float, float, float]:
        """
        将2D像素坐标+深度转换为3D世界坐标
        
        Args:
            x_pixel: 像素x坐标
            y_pixel: 像素y坐标
            depth: 深度值（米）
            intrinsics: 相机内参，可选
            
        Returns:
            (X, Y, Z): 3D坐标（米）
        """
        if intrinsics is None:
            intrinsics = self.camera_intrinsics
        
        if intrinsics is None:
            raise ValueError("Camera intrinsics not set!")
        
        # 反投影公式
        X = (x_pixel - intrinsics['cx']) * depth / intrinsics['fx']
        Y = (y_pixel - intrinsics['cy']) * depth / intrinsics['fy']
        Z = depth
        
        return (X, Y, Z)


if __name__ == "__main__":
    """测试Metric3D v2"""
    print("=" * 60)
    print("Metric3D v2 Test")
    print("=" * 60)
    
    try:
        # 初始化
        print("\n[1/3] Initializing Metric3D v2...")
        depth_estimator = Metric3Dv2(model_size='small', device='auto')
        print("✅ Metric3D v2 initialized")
        
        # 创建测试图像
        print("\n[2/3] Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✅ Test image created")
        
        # 估计深度
        print("\n[3/3] Estimating depth...")
        depth_map = depth_estimator.estimate_depth(test_image)
        print(f"✅ Depth map shape: {depth_map.shape}")
        print(f"✅ Depth range: [{depth_map.min():.2f}m, {depth_map.max():.2f}m]")
        
        # 测试物体深度
        bbox = (100, 100, 200, 200)
        obj_depth = depth_estimator.get_object_depth(depth_map, bbox)
        print(f"✅ Object depth: {obj_depth:.2f} meters")
        
        # 测试3D转换
        x, y, z = depth_estimator.pixel_to_3d(320, 240, obj_depth)
        print(f"✅ 3D position: X={x:.2f}m, Y={y:.2f}m, Z={z:.2f}m")
        
        print("\n" + "=" * 60)
        print("✅ Metric3D v2 Module Ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: First run will download model from GitHub (~500MB)")
        print("      This is normal and only happens once.")
