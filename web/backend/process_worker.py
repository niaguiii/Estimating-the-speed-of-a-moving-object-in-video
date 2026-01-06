#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理Worker进程
独立进程，可以被强制终止
"""
import sys
import os
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
    if len(sys.argv) < 5:
        print("Usage: process_worker.py <task_id> <input_path> <mode> <output_dir>")
        sys.exit(1)
    
    task_id = sys.argv[1]
    input_path = sys.argv[2]
    mode = int(sys.argv[3])
    output_dir = Path(sys.argv[4])
    
    output_path = output_dir / f"{task_id}_output.mp4"
    
    try:
        # 确认工作目录正确
        print(f"[Worker] 当前工作目录: {os.getcwd()}")
        print(f"[Worker] models目录存在: {os.path.exists('models')}")
        
        # 根据模式选择处理函数
        if mode == 1:
            # Mode 1: Detection + Tracking
            import mode1_detection_tracking
            process_func = mode1_detection_tracking.process_video
        elif mode == 2:
            # Mode 2: Speed Estimation
            import mode2_speed_estimation
            process_func = mode2_speed_estimation.process_video
        elif mode == 3:
            # Mode 3: RAFT Optical Flow
            import mode3_raft_optical_flow
            process_func = mode3_raft_optical_flow.process_video_with_raft
        elif mode == 4:
            # Mode 4: Depth Anything V2
            import mode4_depth_anything_v2
            process_func = mode4_depth_anything_v2.process_video_phase3
        elif mode == 5:
            # Mode 5: Metric3D v2
            import mode5_metric3d_v2
            process_func = mode5_metric3d_v2.process_video_metric3d
        else:
            raise ValueError(f"不支持的模式: {mode}. 请选择1-5")
        
        # 处理视频（主程序会输出Frame进度到stdout）
        print(f"[Worker] 开始处理任务 {task_id}")
        success = process_func(
            input_path=input_path,
            output_path=str(output_path),
            show_video=False,
            conf_threshold=0.25
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

if __name__ == "__main__":
    main()
