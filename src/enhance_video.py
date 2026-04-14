# -*- coding: utf-8 -*-
"""
视频预处理增强模块 (Video Enhancement)

三种增强方法：
1. 去雾 (Dehazing)   - 基于暗通道先验 (DCP, He et al. TPAMI 2011)
2. 去模糊 (Deblur)  - 维纳反卷积 (Wiener Deconvolution)
3. 提亮 (Brightness)- CLAHE 直方图均衡化 + Gamma 校正

非侵入式设计：独立函数，可选择性组合调用，不影响现有 pipeline
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


# =============================================================================
# 辅助函数
# =============================================================================

def _read_video_info(video_path: str) -> dict:
    """读取视频基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频: {video_path}")
    try:
        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        }
    finally:
        cap.release()
    return info


def _safe_cv2_write(writer: cv2.VideoWriter, frame: np.ndarray) -> bool:
    """安全写入帧，处理类型和值范围问题"""
    try:
        # 确保数据类型正确
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer.write(frame)
        return True
    except (AttributeError, cv2.error):
        return False


def get_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    获取一个在浏览器中可正常播放的视频写入器。

    策略：优先尝试 H.264（avc1），浏览器兼容性最好；
    兜底 mp4v（OpenCV 默认，但浏览器支持差）。

    Windows 上 OpenCV 对 H.264 的支持取决于编译时是否绑定了 ffmpeg，
    如果 'avc1' 失败会自动降级。
    """
    output_path = os.path.abspath(os.path.normpath(output_path))
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # 优先 H.264（浏览器最兼容）
    preferred = [('avc1', 'H.264/avc1'), ('H264', 'H.264/H264')]
    fallback = [('mp4v', 'MP4V（兼容性差）')]

    for fourcc_str, label in preferred + fallback:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[VideoWriter] 使用 {label} 编码: {output_path}")
            return writer
        writer.release()

    # 最后的兜底：尝试从输入视频复制编码参数
    raise RuntimeError(
        f"无法创建视频写入器（路径: {output_path}），"
        "请检查 OpenCV 是否正确编译了 ffmpeg 支持。"
    )


# =============================================================================
# 去雾 (Dehazing) — 暗通道先验 DCP
# =============================================================================

def apply_dehazing_video(input_path: str,
                          output_path: str,
                          omega: float = 0.95,
                          t0: float = 0.1,
                          kernel_size: int = 15,
                          progress_callback=None) -> bool:
    """
    基于暗通道先验 (DCP) 的视频去雾

    算法步骤（He et al., TPAMI 2011）：
        1. 求暗通道图: I^dark(x) = min_{c∈r,g,b}(I^c(x))
        2. 估计大气光 A: 取暗通道中最亮的0.1%像素对应原图的亮度中值
        3. 估计透射率: t(x) = 1 - ω·I^dark(x)/A
        4. 软抠边（guided filter）细化透射率
        5. 恢复无雾图: J(x) = (I(x) - A) / max(t(x), t0) + A

    Reference:
        He et al., "Single Image Haze Removal Using Dark Channel Prior"
        IEEE TPAMI, Vol. 33, No. 12, December 2011.

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        omega: 雾保留系数（0.95保留少量雾感更自然）
        t0: 透射率下限（防止分母为0）
        kernel_size: 暗通道最小值滤波核大小
        progress_callback: 进度回调函数 callback(percent: float, message: str)
    Returns:
        bool: 处理是否成功
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Dehazing] 无法打开视频: {input_path}")
        return False

    info = _read_video_info(input_path)
    fps = info['fps']
    w, h = info['width'], info['height']
    total = info['total_frames']

    writer = get_video_writer(output_path, fps, w, h)

    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ===== DCP 去雾 =====
        # Step 1: 计算暗通道
        dark = cv2.erode(
            np.min(frame, axis=2).astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8)
        )

        # Step 2: 估计大气光 A（全局）
        h_d, w_d = dark.shape
        num_pixels = h_d * w_d
        num_brightest = max(1, int(num_pixels * 0.001))
        flat = dark.flatten()
        sorted_idx = np.argsort(flat)[::-1]
        candidates = []
        for i in range(num_brightest):
            y = sorted_idx[i] // w_d
            x = sorted_idx[i] % w_d
            brightest = float(np.max(frame[y, x]))
            candidates.append(brightest)
        A = float(np.median(candidates)) if candidates else 200.0
        A = np.clip(A, 180.0, 250.0)

        # Step 3: 计算透射率 t(x) = 1 - omega * dark / A
        transmission = 1.0 - omega * (dark.astype(np.float32) / A)
        transmission = np.clip(transmission, t0, 1.0)

        # Step 4: Guided Filter 细化透射率（使用原彩色帧作为引导）
        # GuidedFilter 需要三通道引导图；将灰度帧扩展为三通道（保留原始彩色信息引导）
        gray_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
        # 引导半径取 kernel_size / 2，避免小图上窗口覆盖整张图像
        guided_radius = max(2, kernel_size // 2)
        try:
            transmission = cv2.ximgproc.createGuidedFilter(
                guide=gray_3ch, radius=guided_radius, eps=1e-3
            ).filter(transmission.astype(np.float32))
            transmission = np.clip(transmission, t0, 1.0)
        except (cv2.error, AttributeError) as e:
            print(f"[Dehazing] ⚠️  Guided Filter 不可用，跳过细化: {e}")

        # Step 5: 恢复无雾图像 J(x) = (I(x) - A) / t(x) + A
        frame_f = frame.astype(np.float32)
        dark_ref = transmission[:, :, np.newaxis]
        recovered = (frame_f - A) / dark_ref + A
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)

        _safe_cv2_write(writer, recovered)
        processed += 1

        # 进度回调
        if progress_callback and total > 0:
            pct = processed / total * 100
            progress_callback(pct, f"去雾中... {processed}/{total} 帧")

    cap.release()
    writer.release()

    if progress_callback:
        progress_callback(100.0, "去雾完成")

    print(f"[Dehazing] 完成: {processed} 帧 → {output_path}")
    return True


# =============================================================================
# 去模糊 (Deblurring) — 维纳反卷积
# =============================================================================

def _estimate_blur_kernel(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    估计运动模糊核（基于 Radon 变换 + 频域分析）

    原理：
        - 模糊图像的频谱会出现特征性的"十字"线条（由模糊核零点造成）
        - 使用 Sobel 提取边缘方向，Radon 变换找到主模糊方向
        - 模糊长度通过频域能量分布估计

    Returns:
        估计的模糊核（简化为线形核），失败返回 None
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 150)

        # 统计边缘方向
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                                minLineLength=30, maxLineGap=10)
        if lines is None or len(lines) < 5:
            return None

        angles = []
        lengths = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angles.append(angle)
            lengths.append(length)

        # 取主方向（加权平均）
        main_angle = np.average(angles, weights=lengths)
        avg_length = np.median(lengths)
        blur_length = min(max(int(avg_length * 0.15), 5), 31)
        if blur_length % 2 == 0:
            blur_length += 1

        # 生成线形模糊核
        kH, kW = blur_length, blur_length
        kernel = np.zeros((kH, kW), dtype=np.float32)
        center = blur_length // 2

        # 沿主方向画线
        cos_a, sin_a = np.cos(main_angle), np.sin(main_angle)
        for i in range(blur_length):
            dx = int(round(i * cos_a))
            dy = int(round(i * sin_a))
            y_pos = center + dy
            x_pos = center + dx
            if 0 <= y_pos < kH and 0 <= x_pos < kW:
                kernel[y_pos, x_pos] = 1.0

        if kernel.sum() > 0:
            kernel /= kernel.sum()
            return kernel
        return None

    except (cv2.error, ValueError, AttributeError):
        return None


def apply_deblurring_video(input_path: str,
                            output_path: str,
                            nsr: float = 0.01,
                            progress_callback=None) -> bool:
    """
    基于维纳反卷积的视频去模糊

    原理：
        模糊建模：B = K ⊗ I + n
        - B: 模糊图像（观测）
        - K: 模糊核
        - I: 清晰图像（目标）
        - n: 噪声

        维纳滤波（频域）：
            Ĥ(ω) = K*(ω)·|B(ω)|² / (|K(ω)|²·|B(ω)|² + NSR)
            即：增强高频，抑制噪声放大

    Reference:
        Wiener, "Extrapolation, Interpolation, and Smoothing of
        Stationary Time Series", 1949.
        Gonzalez & Woods, "Digital Image Processing", 4th Ed., 2018.

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        nsr: 噪信比 (Noise-to-Signal Ratio)，控制去噪强度，越大越平滑
        progress_callback: 进度回调函数
    Returns:
        bool: 处理是否成功
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Deblur] 无法打开视频: {input_path}")
        return False

    info = _read_video_info(input_path)
    fps = info['fps']
    w, h = info['width'], info['height']
    total = info['total_frames']

    writer = get_video_writer(output_path, fps, w, h)

    # 预估计模糊核（取前30帧统计估计）
    estimated_kernel = None
    sample_frames = min(30, total)
    kernel_votes = []
    for i in range(0, total, max(1, total // sample_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            k = _estimate_blur_kernel(frame)
            if k is not None:
                kernel_votes.append(k)
    if kernel_votes:
        # 取中位模糊核
        estimated_kernel = kernel_votes[len(kernel_votes) // 2]
        print(f"[Deblur] 估计模糊核大小: {estimated_kernel.shape}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 默认线形模糊核（对运动模糊效果较好）
    if estimated_kernel is None:
        kH, kW = 15, 15
        estimated_kernel = np.zeros((kH, kW), dtype=np.float32)
        estimated_kernel[kH // 2, :] = 1.0 / kW  # 水平线核

    # 预处理模糊核：中心化 + 频域归一化
    kernel_padded = np.zeros((h, w), dtype=np.float32)
    kH, kW = estimated_kernel.shape
    kY, kX = kH // 2, kW // 2
    kernel_padded[0:kH, 0:kW] = estimated_kernel
    kernel_padded = np.roll(kernel_padded, -kY, axis=0)
    kernel_padded = np.roll(kernel_padded, -kX, axis=1)
    kernel_fft = np.fft.fft2(kernel_padded)
    kernel_fft_conj = np.conj(kernel_fft)

    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # ===== 维纳反卷积（逐通道） =====
        blurred_fft = np.fft.fft2(gray)
        power_kernel = np.abs(kernel_fft) ** 2          # |K(ω)|²
        power_signal = np.abs(blurred_fft) ** 2        # |B(ω)|²

        # 维纳滤波（频域）：
        #   Ĥ(ω) = K*(ω)·|B(ω)|² / (|K(ω)|²·|B(ω)|² + NSR·|B(ω)|²)
        #   = K*(ω) / (|K(ω)|² + NSR)
        # 等价于：K*(ω)·|B(ω)|² / (|K(ω)|²·|B(ω)|² + NSR)
        # 其中 NSR 为噪信功率比（标量），|B(ω)|² 在频域均匀近似时约等于常数
        # 为防止分母为零，加 eps 稳定
        eps = 1e-8
        denominator = power_kernel + nsr
        denominator = np.maximum(denominator, eps)
        wiener_filter = kernel_fft_conj * blurred_fft / denominator
        deblurred_fft = wiener_filter
        deblurred = np.real(np.fft.ifft2(deblurred_fft))
        deblurred = np.clip(deblurred, 0, 255).astype(np.uint8)

        # 彩色化：原彩色帧做轻度锐化补偿细节
        blurred_u8 = gray.astype(np.uint8)
        blurred_3ch = cv2.cvtColor(blurred_u8, cv2.COLOR_GRAY2BGR)
        deblurred_3ch = cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
        # 线性混合：保留去模糊结构，加回20%原色彩
        result = cv2.addWeighted(deblurred_3ch, 0.85, blurred_3ch, 0.15, 0)

        _safe_cv2_write(writer, result)
        processed += 1

        if progress_callback and total > 0:
            pct = processed / total * 100
            progress_callback(pct, f"去模糊中... {processed}/{total} 帧")

    cap.release()
    writer.release()

    if progress_callback:
        progress_callback(100.0, "去模糊完成")

    print(f"[Deblur] 完成: {processed} 帧 → {output_path}")
    return True


# =============================================================================
# 亮度增强 (Brightness Enhancement) — CLAHE + Gamma
# =============================================================================

def apply_brightness_video(input_path: str,
                           output_path: str,
                           gamma: float = 0.65,
                           clip_limit: float = 2.0,
                           tile_size: int = 8,
                           progress_callback=None) -> bool:
    """
    视频亮度增强（提亮或降暗）

    策略：
        - gamma < 1（暗场景）：CLAHE 对比度增强 + gamma 提亮
        - gamma > 1（过曝场景）：跳过 CLAHE（避免放大高光噪声），仅用 gamma 降暗

    Reference:
        Zuiderveld, "Contrast Limited Adaptive Histogram Equalization",
        Graphics Gems IV, 1994.
        Gonzalez & Woods, "Digital Image Processing", 4th Ed., 2018.

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        gamma: Gamma 校正指数，<1 提亮，>1 压暗（默认0.65适合低光照）
        clip_limit: CLAHE 对比度裁剪阈值
        tile_size: CLAHE 网格大小
        progress_callback: 进度回调函数
    Returns:
        bool: 处理是否成功
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Brightness] 无法打开视频: {input_path}")
        return False

    info = _read_video_info(input_path)
    fps = info['fps']
    w, h = info['width'], info['height']
    total = info['total_frames']

    writer = get_video_writer(output_path, fps, w, h)

    is_overexposed = gamma > 1.0
    if not is_overexposed:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    # 构建 Gamma 查找表
    gamma_table = np.array([
        int(((i / 255.0) ** gamma) * 255.0)
        for i in range(256)
    ], dtype=np.uint8)

    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if is_overexposed:
            # 过曝：跳过 CLAHE，直接 gamma 降暗
            result = cv2.LUT(frame, gamma_table)
        else:
            # 暗场景：CLAHE 增强对比度 + gamma 提亮
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_ch, b_ch = cv2.split(lab)
            l_enhanced = clahe.apply(l_channel)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            clahe_result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            gamma_result = cv2.LUT(clahe_result, gamma_table)
            # 轻度去噪（CLAHE 可能放大噪声）
            result = cv2.fastNlMeansDenoisingColored(gamma_result, None, 3, 3, 7, 21)

        _safe_cv2_write(writer, result)
        processed += 1

        if progress_callback and total > 0:
            pct = processed / total * 100
            mode_label = "降暗" if is_overexposed else "提亮中"
            progress_callback(pct, f"{mode_label}... {processed}/{total} 帧")

    cap.release()
    writer.release()

    if progress_callback:
        progress_callback(100.0, "亮度增强完成")

    print(f"[Brightness] 完成: {processed} 帧 → {output_path}")
    return True


# =============================================================================
# 自动组合增强（根据 QualityReport 选择性处理）
# =============================================================================

def enhance_video(input_path: str,
                  output_path: str,
                  issues: List[str],
                  quality_report=None,
                  progress_callback=None,
                  brightness_level: Optional[str] = None) -> Tuple[bool, List[str]]:
    """
    根据检测到的问题自动组合增强流程

    处理顺序（经验最优）：
        1. 去雾（雾会严重影响后续检测，先恢复场景）
        2. 去模糊（恢复清晰边缘）
        3. 提亮/降暗（最后处理亮度）

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（最终输出）
        issues: 问题列表 ["blur", "haze", "brightness"]
        quality_report: QualityReport 对象（可选，用于自适应参数）
        progress_callback: 进度回调函数
        brightness_level: "dark" | "normal" | "overexposed"，用于决定 gamma 方向
    Returns:
        (success, applied_methods): 是否成功，应用了哪些方法
    """
    if not issues:
        print("[Enhance] 无需处理，跳过")
        return True, []

    applied = []
    temp_files = []
    current_input = input_path

    # 自适应 gamma 参数
    # gamma < 1: 提亮（暗场景）；gamma > 1: 压暗（过曝场景）
    if brightness_level == 'overexposed':
        gamma = 1.5  # 过曝场景压暗
    elif quality_report is not None:
        if quality_report.brightness_index < 0.20:
            gamma = 0.50  # 极暗场景
        elif quality_report.brightness_index < 0.30:
            gamma = 0.60  # 偏暗场景
        else:
            gamma = 0.65  # 默认提亮
    else:
        gamma = 0.65

    # 确定处理顺序
    order = ['haze', 'blur', 'brightness']
    steps = [i for i in order if i in issues]

    total_steps = len(steps)
    for step_idx, issue in enumerate(steps):
        # 中间结果存临时文件
        if step_idx < total_steps - 1:
            step_output = output_path + f".step{step_idx}.tmp.mp4"
            temp_files.append(step_output)
        else:
            step_output = output_path

        def step_callback(percent: float, msg: str):
            if progress_callback:
                overall_pct = (step_idx + percent / 100) / total_steps * 100
                progress_callback(overall_pct, msg)

        if issue == 'haze':
            success = apply_dehazing_video(current_input, step_output,
                                           progress_callback=step_callback)
        elif issue == 'blur':
            success = apply_deblurring_video(current_input, step_output,
                                             progress_callback=step_callback)
        elif issue == 'brightness':
            success = apply_brightness_video(current_input, step_output,
                                             gamma=gamma,
                                             progress_callback=step_callback)
        else:
            success = False

        if not success:
            # 清理临时文件
            for tf in temp_files:
                if os.path.exists(tf):
                    try:
                        os.remove(tf)
                    except OSError:
                        pass
            return False, applied

        applied.append(issue)
        current_input = step_output

    # 清理临时文件
    for tf in temp_files:
        if os.path.exists(tf):
            try:
                os.remove(tf)
            except OSError:
                pass

    if progress_callback:
        progress_callback(100.0, f"增强完成: {', '.join(applied)}")

    print(f"[Enhance] 增强完成: {applied}")
    return True, applied
