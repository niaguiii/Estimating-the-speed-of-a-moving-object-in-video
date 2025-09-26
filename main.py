"""
视频中移动物体速度估计项目 - 主程序
简化版启动器，自动处理输入输出
支持ONNX版本和YOLOv8原生版本
"""
import os
import sys
import glob

def get_input_videos():
    """获取input文件夹中的所有视频文件"""
    input_dir = "input"
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
    dirs = ['input', 'output', 'models', 'logs']
    
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"✅ 创建文件夹: {dir_name}/")

def show_menu():
    """显示主菜单"""
    print("=" * 60)
    print("🎯 视频中移动物体速度估计项目 - 第一阶段")
    print("=" * 60)
    print("功能:")
    print("  ✅ 物体检测 (YOLOv8 ONNX / ultralytics原生)")
    print("  ✅ 物体追踪 (ID分配和轨迹跟踪)")
    print("  ✅ 视频分析 (帧数、帧率统计)")
    print("=" * 60)

def select_video():
    """选择要处理的视频"""
    video_files = get_input_videos()
    
    if not video_files:
        print("\n📁 input/ 文件夹中没有找到视频文件")
        print("支持的格式: MP4, AVI, MOV, MKV, FLV, WMV")
        print("\n请将视频文件放入 input/ 文件夹后重新运行程序")
        
        # 询问是否创建测试视频
        choice = input("\n是否创建测试视频？(y/n): ").lower().strip()
        if choice == 'y':
            create_test_video()
            return "input/test_video.mp4"
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
        if not os.path.exists('input'):
            os.makedirs('input')
        
        output_path = 'input/test_video.mp4'
        
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
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # 获取输入文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"output/{base_name}_result.mp4"
    
    return output_name

def select_model_version():
    """选择模型版本"""
    print("\n🤖 选择检测模型版本:")
    print("  1. ONNX版本 (轻量级，兼容性好)")
    print("  2. YOLOv8原生版本 (ultralytics，更准确)")
    
    while True:
        try:
            choice = input("\n请选择模型版本 (1-2, 默认2): ").strip()
            if not choice:
                choice = '2'  # 默认使用原生版本
                
            if choice == '1':
                return 'onnx'
            elif choice == '2':
                return 'native'
            else:
                print("❌ 请输入 1 或 2")
        except ValueError:
            print("❌ 输入格式错误，请重新输入")

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
    if model_version == 'native':
        # 为原生版本添加标识
        base_name = os.path.splitext(output_path)[0]
        output_path = f"{base_name}_yolov8native.mp4"
    
    print(f"\n🚀 开始处理视频...")
    print(f"📥 输入: {selected_video}")
    print(f"📤 输出: {output_path}")
    print(f"🤖 模型: {'YOLOv8原生版本' if model_version == 'native' else 'ONNX版本'}")
    
    # 询问是否显示实时窗口
    show_window = True
    choice = input("\n是否显示处理窗口？(y/n, 默认y): ").lower().strip()
    if choice == 'n':
        show_window = False
    
    # 处理视频
    try:
        if model_version == 'native':
            # 使用YOLOv8原生版本
            from main_yolov8_native import process_video
        else:
            # 使用ONNX版本
            from main_opencv import process_video
        
        if model_version == 'native':
            # 原生版本支持置信度参数
            success = process_video(
                input_path=selected_video,
                output_path=output_path,
                show_video=show_window,
                #conf_threshold=0.1  # 默认0.25，可修改
            )
        else:
            # ONNX版本
            success = process_video(
                input_path=selected_video,
                output_path=output_path,
                show_video=show_window
            )
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 处理完成！")
            print("=" * 60)
            print(f"✅ 输出文件已保存: {output_path}")
            print(f"🤖 使用模型: {'YOLOv8原生版本' if model_version == 'native' else 'ONNX版本'}")
            print("=" * 60)
            
            # 询问是否打开输出文件夹
            choice = input("\n是否打开输出文件夹？(y/n): ").lower().strip()
            if choice == 'y':
                try:
                    import subprocess
                    subprocess.run(['explorer', 'output'], check=True)
                except:
                    print("📁 请手动打开 output/ 文件夹查看结果")
        else:
            print("\n❌ 处理失败，请检查输入文件")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断处理")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 程序退出")
    
    input("\n按任意键关闭窗口...")