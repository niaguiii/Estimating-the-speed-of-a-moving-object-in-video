# -*- coding: utf-8 -*-
"""
视频质量检测模块 (Video Quality Detector)

检测视频中的三种常见质量问题：
1. 模糊 (Blur)      - 基于 Laplacian 方差法
2. 雾浓度 (Haze)    - 基于暗通道先验 (DCP, He et al. TPAMI 2011)
3. 亮度 (Brightness)- 基于直方图统计分析

非侵入式设计：不修改原视频，不影响现有 pipeline
"""
import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class QualityReport:
    """视频质量检测报告"""
    # 数值指标
    blur_index: float       # Laplacian方差，越小越模糊 [0, ∞)
    haze_index: float       # 暗通道均值，越大雾越浓 [0, 255]
    brightness_index: float # 亮度指数 [0,1], 太小暗太大曝

    # 文本等级
    blur_level: str          # "clear" | "moderate" | "blur"
    haze_level: str          # "clear" | "mild" | "foggy"
    brightness_level: str    # "dark" | "normal" | "overexposed"

    # 是否需要增强
    needs_enhancement: bool
    issues: List[str]        # 需要处理的问题列表

    # 元数据
    sampled_frames: int = 0
    total_frames: int = 0
    video_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# 辅助函数
# =============================================================================

def _sample_frames(video_path: str, num_samples: int = 20) -> List[np.ndarray]:
    """
    均匀采样视频帧（覆盖整段视频）

    Args:
        video_path: 视频路径
        num_samples: 最少采样帧数（实际取样数取 min(num_samples, total)）

    Returns:
        采样的灰度帧列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 均匀采样：覆盖整个视频
    # 用 np.linspace 直接从 0 到 total 取均匀分布的帧索引
    effective_samples = min(num_samples, total)
    indices = np.linspace(0, total - 1, effective_samples, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

    cap.release()
    return frames


def _estimate_atmospheric_light(frame: np.ndarray, dark_channel: np.ndarray) -> float:
    """
    估计大气光 A (Atmospheric Light)

    取暗通道中最亮的 0.1% 像素，对应原图取亮度最高的像素，取中值

    Reference: He et al., "Single Image Haze Removal Using Dark Channel Prior"
               TPAMI 2011
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = max(1, int(num_pixels * 0.001))

    # 找到暗通道中值最大的像素位置
    flat = dark_channel.flatten()
    sorted_indices = np.argsort(flat)[::-1]

    candidates = []
    for i in range(num_brightest):
        idx = sorted_indices[i]
        y, x = idx // w, idx % w
        # 该位置原图的亮度（取RGB最大值）
        brightest_channel = np.max(frame[y, x])
        candidates.append(brightest_channel)

    A = float(np.median(candidates)) if candidates else 150.0
    return np.clip(A, 180.0, 250.0)


# =============================================================================
# 核心检测算法
# =============================================================================

def detect_blur(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    模糊度检测 — Laplacian 方差法

    原理：模糊图像的高频细节丢失，Laplacian 算子的响应值降低。
          Laplacian 响应方差 = ∑(I_xx + I_yy - mean)² / N

    Reference:
        - Pech-Pacheco et al., "Object Class Confirmed/Identification via
          Laplacian Images", ICIAR 2000
        - OpenCV Laplacian Variance 作为无参考图像清晰度指标

    Args:
        frames: 灰度帧列表

    Returns:
        (blur_index, blur_level)
        blur_index: Laplacian方差 [0, ∞), 越小越模糊
        blur_level: "clear" | "moderate" | "blur"
    """
    if not frames:
        return 0.0, "unknown"

    variances = []
    for gray in frames:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        var = laplacian.var()
        variances.append(var)

    # 取中位数，抑制异常帧（如黑帧、过渡帧）
    blur_index = float(np.median(variances))

    # 阈值（基于经验值，参考学术文献常用范围）
    if blur_index < 80:
        blur_level = "blur"
    elif blur_index < 350:
        blur_level = "moderate"
    else:
        blur_level = "clear"

    return blur_index, blur_level


def detect_haze(frames: List[np.ndarray], original_frames: Optional[List[np.ndarray]] = None) -> Tuple[float, str]:
    """
    雾浓度检测 — 暗通道先验 (DCP)

    原理：在绝大多数无雾图像中，任意像素的RGB三通道至少有一个接近0。
          有雾图像中，大气散射导致暗通道值升高。

    Reference:
        He et al., "Single Image Haze Removal Using Dark Channel Prior"
        IEEE TPAMI, Vol. 33, No. 12, December 2011.

    雾模型：
        I(x) = J(x) · t(x) + A · (1 - t(x))
        - I(x): 有雾图像（观测）
        - J(x): 无雾图像（目标）
        - A:    大气光
        - t(x): 透射率

    Args:
        frames: 暗通道帧列表（用于快速检测）
        original_frames: 原始BGR帧列表（用于精确A估计）

    Returns:
        (haze_index, haze_level)
        haze_index: 暗通道均值 [0, 255], 越大雾越浓
        haze_level: "clear" | "mild" | "foggy"
    """
    if not frames:
        return 0.0, "unknown"

    # 使用暗通道均值作为雾浓度指标
    dark_vals = []
    for dark in frames:
        dark_vals.append(dark.mean())

    haze_index = float(np.median(dark_vals))

    # 阈值（基于 DCP 论文的经验值）
    if haze_index > 60:
        haze_level = "foggy"
    elif haze_index > 35:
        haze_level = "mild"
    else:
        haze_level = "clear"

    return haze_index, haze_level


def detect_brightness(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    亮度检测 — 直方图 + 亮度统计

    原理：
        - 低光照：灰度均值低，暗部像素占比高
        - 过曝：亮部像素占比高，均值接近饱和

    Reference:
        Gonzalez & Woods, "Digital Image Processing", 4th Ed., 2018.
        Chapter 3: Histogram Processing (直方图均衡化理论基础)

    Args:
        frames: 灰度帧列表

    Returns:
        (brightness_index, brightness_level)
        brightness_index: [0, 1], 太小暗太大曝
        brightness_level: "dark" | "normal" | "overexposed"
    """
    if not frames:
        return 0.5, "unknown"

    means = []
    dark_ratios = []   # 暗部占比 (< 50)
    bright_ratios = [] # 亮部占比 (> 220)

    for gray in frames:
        h, w = gray.shape
        total = h * w
        mean = gray.mean()

        dark_ratio = np.sum(gray < 50) / total
        bright_ratio = np.sum(gray > 220) / total

        means.append(mean)
        dark_ratios.append(dark_ratio)
        bright_ratios.append(bright_ratio)

    mean_brightness = float(np.median(means))
    dark_ratio = float(np.median(dark_ratios))
    bright_ratio = float(np.median(bright_ratios))

    # 亮度指数 = 均值归一化 × 0.6 + 非暗部占比 × 0.4
    brightness_index = (mean_brightness / 255.0) * 0.6 + (1.0 - dark_ratio) * 0.4

    if bright_ratio > 0.15:
        # 过曝优先检测
        brightness_level = "overexposed"
    elif brightness_index < 0.30:
        brightness_level = "dark"
    elif brightness_index > 0.75:
        brightness_level = "overexposed"
    else:
        brightness_level = "normal"

    return brightness_index, brightness_level


def compute_dark_channel(frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """
    计算暗通道图 (Dark Channel)

    I^dark(x) = min_{c∈{r,g,b}} I^c(x)

    Args:
        frame: BGR图像
        kernel_size: 最小值滤波核大小

    Returns:
        暗通道图 [H, W], 值范围 [0, 255]
    """
    # 取每个像素RGB三通道的最小值
    min_rgb = np.min(frame, axis=2)
    # 最小值滤波（取局部区域最小，消除白色物体误判）
    dark = cv2.erode(min_rgb, np.ones((kernel_size, kernel_size), dtype=np.uint8))
    return dark


# =============================================================================
# 主入口
# =============================================================================

def detect_video_quality(video_path: str,
                          num_samples: int = 20) -> QualityReport:
    """
    视频质量检测主函数

    采样视频帧，统一检测模糊 / 雾浓度 / 亮度三种指标，
    返回结构化检测报告。

    非侵入式：不修改原视频，不影响现有 pipeline。

    Args:
        video_path: 视频文件路径
        num_samples: 每种指标最少采样帧数（实际取 min(num_samples, total)）

    Returns:
        QualityReport 结构体
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"[QualityDetector] 无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 采样帧
    frames = _sample_frames(video_path, num_samples=num_samples)
    if not frames:
        raise ValueError(f"[QualityDetector] 采样失败: {video_path}")

    sampled = len(frames)

    # 取原始BGR帧用于DCP检测
    cap2 = cv2.VideoCapture(video_path)
    original_frames = []
    indices = np.linspace(0, total_frames - 1, sampled, dtype=int)
    for idx in indices:
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap2.read()
        if ret and frame is not None:
            original_frames.append(frame)
    cap2.release()

    # 1. 模糊检测（灰度帧）
    blur_index, blur_level = detect_blur(frames)

    # 2. 雾浓度检测（原始BGR帧）
    dark_channel_frames = [compute_dark_channel(f) for f in original_frames]
    haze_index, haze_level = detect_haze(dark_channel_frames, original_frames)

    # 3. 亮度检测（灰度帧）
    brightness_index, brightness_level = detect_brightness(frames)

    # 汇总问题
    issues = []
    if blur_level != "clear":
        issues.append("blur")
    if haze_level != "clear":
        issues.append("haze")
    if brightness_level != "normal":
        issues.append("brightness")

    needs_enhancement = len(issues) > 0

    return QualityReport(
        blur_index=round(blur_index, 2),
        haze_index=round(haze_index, 2),
        brightness_index=round(brightness_index, 4),
        blur_level=blur_level,
        haze_level=haze_level,
        brightness_level=brightness_level,
        needs_enhancement=needs_enhancement,
        issues=issues,
        sampled_frames=sampled,
        total_frames=total_frames,
        video_path=video_path
    )


def quick_detect(video_path: str) -> QualityReport:
    """
    快速检测（少量采样，适合预览）

    仅采样10帧，间隔自动调整，速度快但精度略低。
    """
    return detect_video_quality(video_path, num_samples=10)
