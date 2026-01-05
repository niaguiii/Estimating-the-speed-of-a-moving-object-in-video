"""
视频中移动物体速度估计项目 - 主程序
统一入口

核心功能:
- YOLOv8 物体检测
- ByteTrack 高精度追踪
- 速度估算 (km/h)

模式:
1. 检测追踪模式 - YOLOv8 + ByteTrack
2. 速度估算模式 - YOLOv8 + ByteTrack + Speed Estimation
"""
import os
import sys
import glob

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
    print("  4. RAFT + Depth Anything V2 [Phase 3 Complete - BEST!] 🔥")
    print("     - All features from mode 3")
    print("     - Depth Anything V2 for metric depth estimation")
    print("     - Depth-aware speed estimation")
    print("     - Most accurate results")
    print("")
    print("=" * 60)
    
    while True:
        try:
            choice = input("\nSelect mode (1-4, default 4): ").strip()
            if not choice:
                choice = '4'  # Default: Phase 3 Complete
                
            if choice == '1':
                return 'tracking'
            elif choice == '2':
                return 'speed'
            elif choice == '3':
                return 'raft'
            elif choice == '4':
                return 'phase3'
            else:
                print("[ERROR] Please enter 1-4")
        except ValueError:
            print("[ERROR] Invalid input")

def main():
    """主函数"""
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
    
    # 生成输出文件名
    output_path = get_output_filename(selected_video)
    base_name = os.path.splitext(output_path)[0]
    
    # 根据版本添加标识
    version_suffix = {
        'tracking': '_tracking',
        'speed': '_speed',
        'raft': '_raft',
        'phase3': '_phase3'
    }
    output_path = f"{base_name}{version_suffix.get(model_version, '')}.mp4"
    
    # 模式名称映射
    mode_names = {
        'tracking': 'YOLOv8 + ByteTrack (Detection & Tracking)',
        'speed': 'YOLOv8 + ByteTrack + Speed (Full Features)',
        'raft': 'RAFT Optical Flow + Speed Estimation (Phase 3)',
        'phase3': 'RAFT + Depth Anything V2 (Phase 3 Complete)'
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
    
    # 处理视频
    try:
        # 根据版本导入对应模块
        if model_version == 'phase3':
            from src.main_phase3_complete import process_video_phase3
            # 调用Phase 3完整处理
            success = process_video_phase3(
                input_path=selected_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25,
                show_depth=True,
                depth_frequency=10
            )
        elif model_version == 'raft':
            from src.main_yolov8_raft import process_video_with_raft
            # 调用RAFT处理函数
            success = process_video_with_raft(
                input_path=selected_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25,
                show_flow=False
            )
        elif model_version == 'speed':
            from src.main_yolov8_speed import process_video
            success = process_video(
                input_path=selected_video,
                output_path=output_path,
                show_video=show_window,
                conf_threshold=0.25
            )
        else:  # tracking
            from src.main_yolov8_bytetrack import process_video
            success = process_video(
                input_path=selected_video,
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
            
            if model_version == 'phase3':
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