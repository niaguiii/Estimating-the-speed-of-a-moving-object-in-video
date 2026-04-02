"""
视频中移动物体速度估计项目 - 主程序
统一入口

核心功能:
- YOLOv8 物体检测
- ByteTrack 高精度追踪
- 速度估算 (km/h)

六种处理模式:
1. Mode 1: 检测+追踪 (mode1_detection_tracking.py)
   - YOLOv8 + ByteTrack
   
2. Mode 2: 速度估算 (mode2_speed_estimation.py)
   - YOLOv8 + ByteTrack + 物体尺寸标定
   
3. Mode 3: RAFT光流 (mode3_raft_optical_flow.py)
   - YOLOv8 + RAFT + 移动摄像头支持
   
4. Mode 4: Depth Anything V2 (mode4_depth_anything_v2.py)
   - YOLOv8 + RAFT + Depth Anything V2 (相对深度，±10-15%)
   
5. Mode 5: Metric3D v2 (mode5_metric3d_v2.py) 
   - YOLOv8 + RAFT + Metric3D v2 (绝对深度，±2-5%，最新最好！)

6. Mode 6: 自车测速 (mode6_ego_speed.py)
   - RAFT + Metric3D v2，无需YOLO，路面光流测自车速度
"""
import os
import sys
import glob
import math

def get_input_videos():
    """获取input文件夹中的所有视频文件"""
    input_dir = "data/cli/input"
    if not os.path.exists(input_dir):
        return []
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        # 搜索小写和大写扩展名，但避免重复
        video_files.extend(glob.glob(os.path.join(input_dir, ext), recursive=False))
        video_files.extend(glob.glob(os.path.join(input_dir, ext.upper()), recursive=False))
    
    # 去重：将所有路径标准化后去重
    unique_videos = list(set(os.path.normpath(video) for video in video_files))
    return sorted(unique_videos)  # 排序保证输出顺序一致

def setup_directories():
    """创建必要的文件夹"""
    dirs = ['data/cli/input', 'data/cli/output', 'models', 'logs']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ 创建文件夹: {dir_name}/")

def show_menu():
    """显示主菜单"""
    print("=" * 60)
    print("  Video Speed Estimation System")
    print("  YOLOv8 + ByteTrack + Speed Estimation")
    print("=" * 60)
    print("Core Technologies:")
    print("  - YOLOv8: Object Detection")
    print("  - ByteTrack: High-Precision Tracking")
    print("  - Speed Estimation: Real-world Speed (km/h)")
    print("=" * 60)

def select_video():
    """选择要处理的视频"""
    video_files = get_input_videos()
    
    if not video_files:
        print("\n📁 data/cli/input/ 文件夹中没有找到视频文件")
        print("支持的格式: MP4, AVI, MOV, MKV, FLV, WMV")
        print("\n请将视频文件放入 data/cli/input/ 文件夹后重新运行程序")
        
        # 询问是否创建测试视频
        choice = input("\n是否创建测试视频？(y/n): ").lower().strip()
        if choice == 'y':
            create_test_video()
            return "data/cli/input/test_video.mp4"
        else:
            return None
    
    print(f"\n📹 找到 {len(video_files)} 个视频文件:")
    for i, video in enumerate(video_files, 1):
        filename = os.path.basename(video)
        print(f"  {i}. {filename}")
    
    if len(video_files) == 1:
        print(f"\n🎬 自动选择: {os.path.basename(video_files[0])}")
        return video_files[0]
    
    while True:
        try:
            choice = input(f"\n请选择视频 (1-{len(video_files)}, q退出): ").strip()
            if choice.lower() == 'q':
                return None
            
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(video_files):
                    selected = video_files[index]
                    print(f"✅ 已选择: {os.path.basename(selected)}")
                    return selected
                else:
                    print(f"❌ 无效选择，请输入 1-{len(video_files)} 之间的数字")
            else:
                print("❌ 请输入数字或 'q' 退出")
        except ValueError:
            print("❌ 输入格式错误，请重新输入")

def create_test_video():
    """创建测试视频"""
    try:
        import cv2
        import numpy as np
        
        print("🎬 正在创建测试视频...")
        
        # 确保input文件夹存在
        if not os.path.exists('data/cli/input'):
            os.makedirs('data/cli/input')
        
        output_path = 'data/cli/input/test_video.mp4'
        
        # 视频参数
        width, height = 640, 480
        fps = 30
        duration = 5  # 5秒
        total_frames = fps * duration
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            # 创建背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 添加网格背景
            for i in range(0, width, 50):
                cv2.line(frame, (i, 0), (i, height), (30, 30, 30), 1)
            for i in range(0, height, 50):
                cv2.line(frame, (0, i), (width, i), (30, 30, 30), 1)
            
            t = frame_num / fps
            
            # 移动物体1: 从左到右的汽车形状（更像真实汽车）
            x1 = int(50 + t * 80) % (width - 120)
            y1 = height // 2
            # 绘制汽车主体
            cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 + 40), (50, 50, 200), -1)
            cv2.rectangle(frame, (x1 + 20, y1 - 15), (x1 + 80, y1), (100, 100, 250), -1)
            # 车轮
            cv2.circle(frame, (x1 + 20, y1 + 40), 8, (0, 0, 0), -1)
            cv2.circle(frame, (x1 + 80, y1 + 40), 8, (0, 0, 0), -1)
            cv2.putText(frame, "Car", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 移动物体2: 从上到下的人形轮廓
            x2 = width // 3
            y2 = int(50 + t * 60) % (height - 100)
            # 人头
            cv2.circle(frame, (x2 + 15, y2 + 15), 12, (0, 150, 0), -1)
            # 身体
            cv2.rectangle(frame, (x2 + 5, y2 + 25), (x2 + 25, y2 + 60), (0, 200, 0), -1)
            # 手臂
            cv2.line(frame, (x2 + 5, y2 + 35), (x2 - 5, y2 + 45), (0, 200, 0), 3)
            cv2.line(frame, (x2 + 25, y2 + 35), (x2 + 35, y2 + 45), (0, 200, 0), 3)
            # 腿
            cv2.line(frame, (x2 + 10, y2 + 60), (x2 + 5, y2 + 85), (0, 200, 0), 3)
            cv2.line(frame, (x2 + 20, y2 + 60), (x2 + 25, y2 + 85), (0, 200, 0), 3)
            cv2.putText(frame, "Person", (x2-10, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 添加信息
            cv2.putText(frame, f"Test Video - Frame {frame_num + 1}/{total_frames}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time: {t:.2f}s", 
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            out.write(frame)
        
        out.release()
        print(f"✅ 测试视频创建完成: {output_path}")
        
    except Exception as e:
        print(f"❌ 测试视频创建失败: {e}")

def get_output_filename(input_path):
    """生成输出文件名"""
    if not os.path.exists('data/cli/output'):
        os.makedirs('data/cli/output')
    
    # 获取输入文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"data/cli/output/{base_name}_result.mp4"
    
    return output_name

def select_model_version():
    """选择处理模式"""
    print("\n" + "=" * 60)
    print("Select Processing Mode:")
    print("=" * 60)
    print("")
    print("  1. Detection + Tracking")
    print("     - YOLOv8 object detection")
    print("     - ByteTrack high-precision tracking")
    print("     - Trajectory visualization")
    print("")
    print("  2. Detection + Tracking + Speed Estimation")
    print("     - All features from mode 1")
    print("     - Speed estimation (km/h)")
    print("     - Assumes stationary camera")
    print("")
    print("  3. RAFT Optical Flow + Speed Estimation [Phase 3]")
    print("     - All features from mode 2")
    print("     - RAFT optical flow for camera motion separation")
    print("     - Supports moving camera scenarios")
    print("")
    print("  4. RAFT + Depth Anything V2 [Depth Estimation]")
    print("     - All features from mode 3")
    print("     - Depth Anything V2 for relative depth estimation")
    print("     - Depth-aware speed estimation")
    print("     - Good accuracy (±10-15%)")
    print("")
    print("  5. RAFT + Metric3D v2 [BEST - Absolute Depth!] ")
    print("     - All features from mode 4")
    print("     - Metric3D v2 for ABSOLUTE depth (meters)")
    print("     - No manual calibration needed")
    print("     - Highest accuracy (±2-5%)")
    print("     - Universal scene support")
    print("")
    print("  6. Ego-Vehicle Speed (mode6_ego_speed.py)")
    print("     - NO YOLO detection needed")
    print("     - RAFT + Metric3D v2 on road surface")
    print("     - Measures YOUR OWN vehicle speed")
    print("     - Ideal for dashcam footage")
    print("")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelect mode (1-6, default 5): ").strip()
            if not choice:
                choice = '5'  # Default: Metric3D v2 (最新最好)
                
            if choice == '1':
                return 'tracking'
            elif choice == '2':
                return 'speed'
            elif choice == '3':
                return 'raft'
            elif choice == '4':
                return 'phase3'
            elif choice == '5':
                return 'metric3d'
            elif choice == '6':
                return 'ego'
            else:
                print("[ERROR] Please enter 1-6")
        except ValueError:
            print("[ERROR] Invalid input")

def main():
    """
    主函数 - 支持六种处理模式
    
    Mode 1: mode1_detection_tracking.py    - 检测+追踪
    Mode 2: mode2_speed_estimation.py      - 速度估算
    Mode 3: mode3_raft_optical_flow.py     - RAFT光流
    Mode 4: mode4_depth_anything_v2.py     - Depth Anything V2
    Mode 5: mode5_metric3d_v2.py           - Metric3D v2 (推荐)
    Mode 6: mode6_ego_speed.py             - 自车测速
    """
    show_menu()
    
    # 设置目录结构
    setup_directories()
    
    # 选择视频
    selected_video = select_video()
    if not selected_video:
        print("\n👋 程序退出")
        return
    
    # 选择模型版本
    model_version = select_model_version()

    # ========== 预处理增强（自动检测 + 可选处理）==========
    # 导入检测模块（懒加载，失败不影响主流程）
    enhancement_options = []  # 默认不预处理
    try:
        from src.quality_detector import detect_video_quality, QualityReport
        print("\n" + "=" * 60)
        print("🔍 视频质量自动检测中...")
        print("=" * 60)
        report = detect_video_quality(selected_video)
        print(f"  清晰度: Laplacian方差 = {report.blur_index:.1f}  [{report.blur_level}]")
        print(f"  雾气:   暗通道均值 = {report.haze_index:.1f}  [{report.haze_level}]")
        print(f"  亮度:   亮度指数 = {report.brightness_index:.3f}  [{report.brightness_level}]")
        print(f"  采样帧数: {report.sampled_frames}/{report.total_frames}")

        if report.needs_enhancement:
            print(f"\n  ⚠️  检测到以下问题: {', '.join(report.issues)}")
            print("\n  预处理增强选项:")
            print("  0. 跳过预处理（直接处理视频，默认）")
            print("  1. 检测报告（仅显示，不处理）")
            print("  2. 自动处理（处理全部检测到的问题）")
            print("  3. 手动选择：")
            issue_labels = {
                'blur': '3a. 去模糊 (Wiener反卷积)',
                'haze': '3b. 去雾 (DCP暗通道先验)',
                'brightness': '3c. 提亮 (CLAHE+Gamma)',
            }
            for issue in report.issues:
                print(f"       {issue_labels.get(issue, issue)}")

            choice = input("\n  选择预处理选项 (0-3, 默认 0): ").strip()
            if choice == '1':
                print("\n  [检测模式] 仅显示报告，不执行预处理")
            elif choice == '2':
                enhancement_options = report.issues
                print(f"\n  [自动增强] 将处理: {', '.join(enhancement_options)}")
            elif choice == '3':
                selected = []
                valid_keys = {'a': 'blur', 'b': 'haze', 'c': 'brightness'}
                sub_choice = input("  输入选项 (如 ab / ac / abc，空格分隔: ").strip().lower()
                for ch in sub_choice:
                    if ch in valid_keys and valid_keys[ch] in report.issues:
                        selected.append(valid_keys[ch])
                enhancement_options = selected
                if enhancement_options:
                    print(f"  [手动选择] 将处理: {', '.join(enhancement_options)}")
                else:
                    print("  [无选择] 跳过预处理")
            else:
                print("  [跳过] 不执行预处理")
        else:
            print("\n  ✅ 视频质量良好，无需预处理")

    except ImportError:
        print("\n  [提示] 预处理模块不可用，跳过质量检测")
    except Exception as e:
        print(f"\n  [提示] 预处理检测失败，继续直接处理: {e}")

    # 生成输出文件名
    output_path = get_output_filename(selected_video)
    base_name = os.path.splitext(output_path)[0]
    
    # 根据版本添加标识
    version_suffix = {
        'tracking': '_mode1',
        'speed': '_mode2',
        'raft': '_mode3',
        'phase3': '_mode4',
        'metric3d': '_mode5',
        'ego': '_mode6'
    }
    output_path = f"{base_name}{version_suffix.get(model_version, '')}.mp4"
    
    # 模式名称映射
    mode_names = {
        'tracking': 'Mode 1: YOLOv8 + ByteTrack (Detection & Tracking)',
        'speed': 'Mode 2: YOLOv8 + ByteTrack + Speed',
        'raft': 'Mode 3: RAFT Optical Flow + Speed',
        'phase3': 'Mode 4: RAFT + Depth Anything V2 (Relative Depth)',
        'metric3d': 'Mode 5: RAFT + Metric3D v2 (Absolute Depth - BEST!)',
        'ego': 'Mode 6: Ego-Vehicle Speed (RAFT + Metric3D, No YOLO)'
    }
    
    print(f"\n" + "=" * 60)
    print(f"Starting video processing...")
    print(f"=" * 60)
    print(f"Input:  {selected_video}")
    print(f"Output: {output_path}")
    print(f"Mode:   {mode_names.get(model_version, model_version)}")
    print("=" * 60)
    
    # 询问是否显示实时窗口
    show_window = True
    choice = input("\nShow video window? (y/n, default y): ").lower().strip()
    if choice == 'n':
        show_window = False

    # ========== 执行预处理增强 ==========
    current_video = selected_video
    if enhancement_options:
        try:
            from src.quality_detector import detect_video_quality
            from src.enhance_video import enhance_video
            from src.quality_detector import QualityReport

            report = detect_video_quality(selected_video)
            # 中间增强文件路径（临时）
            enhanced_path = selected_video.rsplit('.', 1)
            enhanced_path = f"{enhanced_path[0]}_enhanced.{enhanced_path[1]}"

            print(f"\n{'=' * 60}")
            print(f"🔧 正在执行预处理增强...")
            print(f"{'=' * 60}")

            def progress_cb(pct, msg):
                print(f"\r  进度: {pct:.0f}%  {msg}", end='', flush=True)

            success, applied = enhance_video(
                input_path=selected_video,
                output_path=enhanced_path,
                issues=enhancement_options,
                quality_report=report,
                progress_callback=progress_cb,
                brightness_level=report.brightness_level
            )
            print()  # 换行

            if success and os.path.exists(enhanced_path):
                current_video = enhanced_path
                print(f"  ✅ 增强完成，已应用: {', '.join(applied)}")
            else:
                print(f"  ⚠️ 增强失败，将使用原始视频")
        except ImportError:
            print("\n  [提示] 预处理模块不可用，跳过增强")
        except Exception as e:
            print(f"\n  [提示] 预处理增强失败: {e}")

    # 处理视频
    try:
        # 根据版本导入对应模块
        if model_version == 'metric3d':
            # Mode 5: Metric3D v2 (绝对深度)
            # 询问相机焦段 → 自动换算水平视角
            print("\n[Mode 5] 请输入相机等效全画幅焦段（mm）")
            print("  常见焦段参考: 14mm(超广) 24mm(广角) 35mm(标准广) 50mm(标准) 85mm(人像) 135mm(中长)")
            print("  手机广角端通常约等效 24-28mm，固定监控约 35-50mm")
            _focal_input = input("  焦段 (默认 50): ").strip()
            try:
                _focal_mm = float(_focal_input) if _focal_input else 50.0
                if _focal_mm <= 0:
                    raise ValueError
            except ValueError:
                print("  [警告] 输入无效，使用默认值 50mm")
                _focal_mm = 50.0
            # 全画幅传感器宽度 36mm → 水平 FOV
            fov_degrees = 2.0 * math.degrees(math.atan(18.0 / _focal_mm))
            print(f"  ✅ {_focal_mm:.0f}mm  →  水平FOV ≈ {fov_degrees:.1f}°")

            print("\n[Mode 5] 深度更新频率（每N帧重算一次深度，越小越精确但越慢）")
            print("  推荐: GPU可用→3~5, 仅CPU→10, 极快速物体→3")
            _depth_input = input("  深度频率 (默认 5): ").strip()
            try:
                depth_freq = int(_depth_input) if _depth_input else 5
                if depth_freq <= 0:
                    raise ValueError
            except ValueError:
                print("  [警告] 输入无效，使用默认值 5")
                depth_freq = 5
            print(f"  ✅ 每 {depth_freq} 帧更新一次深度")

            from src.mode5_metric3d_v2 import process_video_metric3d
            success = process_video_metric3d(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25,
                show_depth=True,
                depth_frequency=depth_freq,
                model_size='small',  # 可选：'small', 'large', 'giant2'
                fov_degrees=fov_degrees
            )
        elif model_version == 'phase3':
            # Mode 4: Depth Anything V2 (相对深度)
            from src.mode4_depth_anything_v2 import process_video_phase3
            success = process_video_phase3(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25,
                show_depth=True,
                depth_frequency=10
            )
        elif model_version == 'raft':
            # Mode 3: RAFT Optical Flow
            from src.mode3_raft_optical_flow import process_video_with_raft
            success = process_video_with_raft(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25,
                show_flow=False
            )
        elif model_version == 'speed':
            # Mode 2: Speed Estimation
            from src.mode2_speed_estimation import process_video
            success = process_video(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25
            )
        elif model_version == 'ego':
            # Mode 6: Ego-Vehicle Speed (no YOLO)
            # 询问相机焦段 → 自动换算水平视角
            print("\n[Mode 6] 请输入相机等效全画幅焦段（mm）")
            print("  常见焦段参考: 14mm(超广) 24mm(广角) 35mm(标准广) 50mm(标准) 85mm(人像) 135mm(中长)")
            print("  行车记录仪/手持行走通常约等效 14-24mm，手机广角端约 24-28mm")
            _focal_input = input("  焦段 (默认 24): ").strip()
            try:
                _focal_mm = float(_focal_input) if _focal_input else 24.0
                if _focal_mm <= 0:
                    raise ValueError
            except ValueError:
                print("  [警告] 输入无效，使用默认值 24mm")
                _focal_mm = 24.0
            fov_degrees = 2.0 * math.degrees(math.atan(18.0 / _focal_mm))
            print(f"  ✅ {_focal_mm:.0f}mm  →  水平FOV ≈ {fov_degrees:.1f}°")

            from src.mode6_ego_speed import process_video_ego_speed
            success = process_video_ego_speed(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                fov_degrees=fov_degrees,
                depth_frequency=10,
                road_region_ratio=0.4,
                model_size='small'
            )
        else:  # tracking
            # Mode 1: Detection + Tracking
            from src.mode1_detection_tracking import process_video
            success = process_video(
                input_path=current_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25
            )
        
        if success:
            print("\n" + "=" * 60)
            print("[DONE] Processing Complete!")
            print("=" * 60)
            print(f"[OK] Output saved: {output_path}")
            print(f"[OK] Mode: {mode_names.get(model_version, model_version)}")

            if enhancement_options:
                print(f"[OK] Pre-enhancement: {', '.join(enhancement_options)}")

            if model_version == 'ego':
                print("[OK] Ego speed: RAFT + Metric3D v2 road surface sampling")
                print("[OK] No YOLO detection needed - measures your own vehicle speed!")
            elif model_version == 'metric3d':
                print("[OK] RAFT + Metric3D v2 + 3D Speed estimation enabled")
                print("[OK] Absolute depth (meters) - No calibration needed!")
            elif model_version == 'phase3':
                print("[OK] RAFT + Depth Anything V2 + Speed estimation enabled")
            elif model_version == 'raft':
                print("[OK] RAFT optical flow + Speed estimation enabled")
            elif model_version == 'speed':
                print("[OK] Speed estimation enabled")
            else:
                print("[OK] ByteTrack tracking enabled")
            
            print("=" * 60)
            
            # 询问是否打开输出文件夹
            choice = input("\nOpen output folder? (y/n): ").lower().strip()
            if choice == 'y':
                try:
                    import subprocess
                    subprocess.run(['explorer', 'data\\cli\\output'], check=True)
                except:
                    print("Please open data/cli/output/ folder manually")
        else:
            print("\n[ERROR] Processing failed")
            
    except KeyboardInterrupt:
        print("\n\n[STOP] User interrupted")
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram exit")
    
    input("\nPress Enter to close...")