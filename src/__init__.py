"""
核心程序模块

包含所有核心检测、追踪和速度估算功能
"""

__version__ = "2.0.0"
__all__ = [
    'model_config',
    'main_opencv',
    'main_yolov8_native',
    'mode1_detection_tracking',
    'mode2_speed_estimation',
    'mode3_raft_optical_flow',
    'mode4_depth_anything_v2',
    'mode5_metric3d_v2',
    'mode6_ego_speed_v2',
    'mode6_ego_speed',
    'optical_flow_raft',
    'depth_estimation',
    'depth_estimation_metric3d',
    'enhance_video',
    'quality_detector',
]
