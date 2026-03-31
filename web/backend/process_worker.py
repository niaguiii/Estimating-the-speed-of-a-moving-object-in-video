#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理Worker进程
独立进程，可以被强制终止
"""
import math
import os
import sys
from pathlib import Path

# 强制stdout无缓冲（确保实时输出）
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# 添加项目根目录到路径
# __file__ = .../web/backend/process_worker.py
# PROJECT_ROOT = .../web/backend -> .. -> web -> .. -> 项目根目录
SCRIPT_DIR = Path(__file__).parent.resolve()  # web/backend
WEB_DIR = SCRIPT_DIR.parent  # web
PROJECT_ROOT = WEB_DIR.parent  # 项目根目录
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

# ✅ 关键：切换工作目录到项目根目录！
# 这样model_config.py的get_project_root()才能正确找到models/目录
os.chdir(str(PROJECT_ROOT))
print(f"[Worker] 工作目录: {os.getcwd()}")
print(f"[Worker] 项目根目录: {PROJECT_ROOT}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Video Processing Worker')
    parser.add_argument('task_id', help='Task ID')
    parser.add_argument('input_path', help='Input video path')
    parser.add_argument('mode', type=int, help='Processing mode (1-6)')
    parser.add_argument('output_dir', help='Output directory')
    parser.add_argument('--focal-mm', type=float, default=None, help='Equivalent focal length in mm (Mode 5/6)')
    parser.add_argument('--depth-freq', type=int, default=None, help='Depth update frequency (Mode 5)')
    args = parser.parse_args()

    task_id = args.task_id
    input_path = args.input_path
    mode = args.mode
    output_dir = Path(args.output_dir)
    focal_mm = args.focal_mm
    depth_freq = args.depth_freq

    output_path = output_dir / f"{task_id}_output.mp4"

    try:
        print(f"[Worker] 当前工作目录: {os.getcwd()}")
        print(f"[Worker] models目录存在: {os.path.exists('models')}")

        # 根据模式选择处理函数
        if mode == 1:
            import mode1_detection_tracking
            process_func = mode1_detection_tracking.process_video
            kwargs = {}
        elif mode == 2:
            import mode2_speed_estimation
            process_func = mode2_speed_estimation.process_video
            kwargs = {}
        elif mode == 3:
            import mode3_raft_optical_flow
            process_func = mode3_raft_optical_flow.process_video_with_raft
            kwargs = {}
        elif mode == 4:
            import mode4_depth_anything_v2
            process_func = mode4_depth_anything_v2.process_video_phase3
            kwargs = {}
        elif mode == 5:
            import mode5_metric3d_v2
            process_func = mode5_metric3d_v2.process_video_metric3d
            fov = _calc_fov(focal_mm, default=60.0)
            depth_frequency = depth_freq if depth_freq else 5
            kwargs = {'fov_degrees': fov, 'depth_frequency': depth_frequency}
        elif mode == 6:
            import mode6_ego_speed
            process_func = mode6_ego_speed.process_video_ego_speed
            fov = _calc_fov(focal_mm, default=75.0)
            kwargs = {'fov_degrees': fov}
        else:
            raise ValueError(f"不支持的模式: {mode}. 请选择1-6")

        print(f"[Worker] 开始处理任务 {task_id}")
        success = process_func(
            input_path=input_path,
            output_path=str(output_path),
            show_video=False,
            conf_threshold=0.25,
            **kwargs
        )

        if success:
            print(f"✅ 任务 {task_id} 处理完成")
            sys.exit(0)
        else:
            print(f"❌ 任务 {task_id} 处理失败")
            sys.exit(1)

    except KeyboardInterrupt:
        print(f"⚠️ 任务 {task_id} 被中断")
        sys.exit(2)
    except Exception as e:
        print(f"❌ 任务 {task_id} 出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _calc_fov(focal_mm, default):
    """根据等效焦段计算水平FOV（度）"""
    if focal_mm is None or focal_mm <= 0:
        return default
    return 2.0 * math.degrees(math.atan(18.0 / focal_mm))

if __name__ == "__main__":
    main()
