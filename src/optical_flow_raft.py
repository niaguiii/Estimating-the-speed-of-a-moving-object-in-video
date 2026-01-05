# -*- coding: utf-8 -*-
"""
RAFT Optical Flow - Phase 3 Core Module
使用PyTorch内置RAFT模型进行光流估计，实现摄像头运动分离

Features:
- RAFT optical flow computation
- Camera motion estimation
- Object real motion separation
- GPU/CPU automatic detection
"""
# ⚠️ 关键：必须最先导入model_config来设置环境变量！
try:
    from . import model_config
except ImportError:
    import model_config

# 现在才导入torch和相关库
import torch
import torchvision.models.optical_flow as flow_models
import cv2
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RAFTOpticalFlow:
    """RAFT光流估计器 - 摄像头运动分离"""
    
    def __init__(self, model_type: str = 'small', device: str = 'auto'):
        """
        初始化RAFT模型
        
        Args:
            model_type: 'small' or 'large' (small更快，large更准确)
            device: 'auto', 'cuda', 'cpu'
        """
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[RAFT] Initializing RAFT_{model_type.upper()} on {self.device}")
        
        # 加载RAFT模型
        if model_type == 'small':
            weights = flow_models.Raft_Small_Weights.DEFAULT
            self.model = flow_models.raft_small(weights=weights, progress=True)
        else:
            weights = flow_models.Raft_Large_Weights.DEFAULT
            self.model = flow_models.raft_large(weights=weights, progress=True)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"[RAFT] Model loaded successfully")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        预处理帧：BGR -> RGB -> Tensor
        
        Args:
            frame: OpenCV BGR图像 (H, W, 3)
            
        Returns:
            tensor: (1, 3, H, W) tensor
        """
        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Numpy -> Tensor (H, W, C) -> (C, H, W)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float()
        
        # 添加batch维度 (1, 3, H, W)
        frame_tensor = frame_tensor.unsqueeze(0)
        
        return frame_tensor.to(self.device)
    
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        计算两帧之间的光流
        
        Args:
            frame1: 前一帧 (H, W, 3) BGR
            frame2: 当前帧 (H, W, 3) BGR
            
        Returns:
            flow: 光流场 (H, W, 2) 其中flow[:,:,0]是x方向，flow[:,:,1]是y方向
        """
        with torch.no_grad():
            # 预处理
            img1 = self.preprocess_frame(frame1)
            img2 = self.preprocess_frame(frame2)
            
            # RAFT推理
            flow_predictions = self.model(img1, img2)
            
            # 获取最终预测 (取最后一个迭代结果)
            flow = flow_predictions[-1][0].cpu().numpy()
            
            # (2, H, W) -> (H, W, 2)
            flow = np.transpose(flow, (1, 2, 0))
            
        return flow
    
    def estimate_camera_motion(self, flow: np.ndarray, 
                               method: str = 'median') -> Tuple[float, float]:
        """
        估计摄像头的全局运动
        
        Args:
            flow: 光流场 (H, W, 2)
            method: 'median' (中位数) 或 'mean' (平均值)
            
        Returns:
            (dx, dy): 摄像头在x和y方向的运动
        """
        if method == 'median':
            # 使用中位数（对异常值更鲁棒）
            camera_dx = np.median(flow[:, :, 0])
            camera_dy = np.median(flow[:, :, 1])
        else:
            # 使用平均值
            camera_dx = np.mean(flow[:, :, 0])
            camera_dy = np.mean(flow[:, :, 1])
        
        return camera_dx, camera_dy
    
    def separate_object_motion(self, flow: np.ndarray, bbox: Tuple[int, int, int, int],
                               camera_motion: Tuple[float, float]) -> Tuple[float, float]:
        """
        分离物体的真实运动（去除摄像头运动）
        
        Args:
            flow: 光流场 (H, W, 2)
            bbox: 物体边界框 (x1, y1, x2, y2)
            camera_motion: 摄像头运动 (dx, dy)
            
        Returns:
            (obj_dx, obj_dy): 物体真实运动向量
        """
        x1, y1, x2, y2 = bbox
        
        # 确保边界框在图像范围内
        h, w = flow.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return 0.0, 0.0
        
        # 提取物体区域的光流
        obj_flow = flow[y1:y2, x1:x2]
        
        # 物体的表观运动（包含摄像头运动）
        obj_apparent_dx = np.median(obj_flow[:, :, 0])
        obj_apparent_dy = np.median(obj_flow[:, :, 1])
        
        # 真实运动 = 表观运动 - 摄像头运动
        camera_dx, camera_dy = camera_motion
        obj_real_dx = obj_apparent_dx - camera_dx
        obj_real_dy = obj_apparent_dy - camera_dy
        
        return obj_real_dx, obj_real_dy
    
    def visualize_flow(self, flow: np.ndarray, max_magnitude: float = 10.0) -> np.ndarray:
        """
        可视化光流场（HSV颜色编码）
        
        Args:
            flow: 光流场 (H, W, 2)
            max_magnitude: 最大幅度（用于归一化）
            
        Returns:
            flow_img: BGR图像 (H, W, 3)
        """
        h, w = flow.shape[:2]
        
        # 创建HSV图像
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 计算光流的幅度和角度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 角度映射到色调 (0-180)
        hsv[..., 0] = ang * 180 / np.pi / 2
        
        # 幅度映射到亮度 (归一化到0-255)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # 饱和度固定为最大
        hsv[..., 1] = 255
        
        # HSV -> BGR
        flow_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_img
    
    def draw_flow_arrows(self, frame: np.ndarray, flow: np.ndarray, 
                        step: int = 16, scale: float = 3.0) -> np.ndarray:
        """
        在图像上绘制光流箭头
        
        Args:
            frame: 原始帧
            flow: 光流场
            step: 采样步长（每隔几个像素画一个箭头）
            scale: 箭头长度缩放
            
        Returns:
            frame_with_arrows: 带箭头的图像
        """
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # 采样点
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        
        # 对应的光流
        fx, fy = flow[y, x].T
        
        # 绘制箭头
        for i in range(len(x)):
            x_start, y_start = x[i], y[i]
            x_end = int(x_start + fx[i] * scale)
            y_end = int(y_start + fy[i] * scale)
            
            # 只画显著运动
            if abs(fx[i]) > 0.5 or abs(fy[i]) > 0.5:
                cv2.arrowedLine(result, (x_start, y_start), (x_end, y_end),
                               (0, 255, 0), 1, tipLength=0.3)
        
        return result


class CameraMotionCompensator:
    """摄像头运动补偿器 - 用于速度估计"""
    
    def __init__(self):
        self.raft = RAFTOpticalFlow(model_type='small')  # 使用small模型更快
        self.prev_frame = None
        self.camera_motion_history = []
        self.max_history = 30  # 保留30帧历史用于平滑
        
    def update(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        更新当前帧，计算摄像头运动
        
        Args:
            frame: 当前帧 (H, W, 3) BGR
            
        Returns:
            camera_motion: (dx, dy) 或 None（第一帧）
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return None
        
        # 计算光流
        flow = self.raft.compute_flow(self.prev_frame, frame)
        
        # 估计摄像头运动
        camera_motion = self.raft.estimate_camera_motion(flow, method='median')
        
        # 保存历史
        self.camera_motion_history.append(camera_motion)
        if len(self.camera_motion_history) > self.max_history:
            self.camera_motion_history.pop(0)
        
        # 更新前一帧
        self.prev_frame = frame.copy()
        
        return camera_motion
    
    def get_smoothed_camera_motion(self) -> Tuple[float, float]:
        """获取平滑后的摄像头运动（使用历史平均）"""
        if not self.camera_motion_history:
            return 0.0, 0.0
        
        dx_list = [m[0] for m in self.camera_motion_history]
        dy_list = [m[1] for m in self.camera_motion_history]
        
        return np.mean(dx_list), np.mean(dy_list)
    
    def compensate_object_motion(self, apparent_motion: Tuple[float, float]) -> Tuple[float, float]:
        """
        补偿物体的表观运动，得到真实运动
        
        Args:
            apparent_motion: 物体的表观运动 (dx, dy)
            
        Returns:
            real_motion: 物体的真实运动 (dx, dy)
        """
        camera_dx, camera_dy = self.get_smoothed_camera_motion()
        
        real_dx = apparent_motion[0] - camera_dx
        real_dy = apparent_motion[1] - camera_dy
        
        return real_dx, real_dy


if __name__ == "__main__":
    """测试RAFT光流"""
    print("=" * 60)
    print("RAFT Optical Flow Test")
    print("=" * 60)
    
    # 初始化RAFT
    raft = RAFTOpticalFlow(model_type='small', device='auto')
    
    # 创建测试图像
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\nComputing optical flow...")
    flow = raft.compute_flow(frame1, frame2)
    print(f"✅ Flow shape: {flow.shape}")
    
    print("\nEstimating camera motion...")
    camera_motion = raft.estimate_camera_motion(flow)
    print(f"✅ Camera motion: dx={camera_motion[0]:.2f}, dy={camera_motion[1]:.2f}")
    
    print("\nSeparating object motion...")
    bbox = (100, 100, 200, 200)
    obj_motion = raft.separate_object_motion(flow, bbox, camera_motion)
    print(f"✅ Object real motion: dx={obj_motion[0]:.2f}, dy={obj_motion[1]:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ RAFT Optical Flow Module Ready!")
    print("=" * 60)
