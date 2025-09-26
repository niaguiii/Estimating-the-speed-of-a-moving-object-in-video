"""
项目安装和测试工具 - 统一的环境配置和测试脚本
替代原来的 environment_setup.py, quick_test.py, test_opencv.py
"""
import sys
import os
import subprocess
import platform
import glob

def show_header():
    """显示标题"""
    print("=" * 60)
    print("🎯 视频中移动物体速度估计项目 - 环境配置和测试")
    print("=" * 60)

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 7):
        print("❌ Python版本过低，需要3.7+")
        return False
    elif version < (3, 8):
        print("⚠️  建议升级到Python 3.8+")
    else:
        print("✅ Python版本符合要求")
    
    return True

def test_basic_libraries():
    """测试基础库"""
    print("\n📦 检查基础库...")
    
    libraries = {
        'numpy': 'NumPy',
        'cv2': 'OpenCV'
    }
    
    missing = []
    
    for lib, name in libraries.items():
        try:
            module = __import__(lib)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name} {version}")
        except ImportError:
            print(f"❌ {name} 未安装")
            missing.append(lib)
    
    return len(missing) == 0, missing

def install_dependencies():
    """安装依赖"""
    print("\n⬇️  开始安装依赖...")
    
    try:
        # 升级pip
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip升级成功")
        
        # 安装核心依赖
        core_deps = ["opencv-python", "numpy"]
        for dep in core_deps:
            print(f"安装 {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} 安装成功")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def test_opencv_features():
    """测试OpenCV功能"""
    print("\n🔧 测试OpenCV功能...")
    
    try:
        import cv2
        
        # 测试DNN模块
        try:
            cv2.dnn.readNet
            print("✅ OpenCV DNN模块可用")
        except AttributeError:
            print("❌ OpenCV DNN模块不可用")
            return False
        
        # 测试视频处理
        try:
            cv2.VideoCapture
            print("✅ OpenCV视频处理模块可用")
        except AttributeError:
            print("❌ OpenCV视频处理模块不可用")
            return False
        
        # 测试级联分类器
        try:
            cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ OpenCV级联分类器可用")
        except Exception:
            print("⚠️  级联分类器加载失败，但不影响主要功能")
        
        return True
        
    except ImportError:
        print("❌ OpenCV未安装")
        return False

def create_test_video():
    """创建测试视频"""
    print("\n🎬 创建测试视频...")
    
    try:
        import cv2
        import numpy as np
        
        # 确保input文件夹存在
        if not os.path.exists('input'):
            os.makedirs('input')
        
        output_path = 'input/test_video.mp4'
        
        # 视频参数
        width, height = 640, 480
        fps = 30
        duration = 3  # 3秒
        total_frames = fps * duration
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            # 创建背景
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            t = frame_num / fps
            
            # 移动物体1: 绿色矩形
            x1 = int(50 + t * 150) % (width - 80)
            y1 = height // 3
            cv2.rectangle(frame, (x1, y1), (x1 + 60, y1 + 40), (0, 255, 0), -1)
            
            # 移动物体2: 蓝色圆形
            x2 = width // 2
            y2 = int(50 + t * 100) % (height - 80)
            cv2.circle(frame, (x2, y2), 20, (255, 0, 0), -1)
            
            # 添加信息
            cv2.putText(frame, f"Test Frame {frame_num + 1}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✅ 测试视频创建完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 测试视频创建失败: {e}")
        return False

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_dirs = ['input', 'output', 'models', 'docs']
    core_files = ['main.py', 'main_opencv.py', 'config.py', 'README.md']
    
    all_good = True
    
    # 检查目录
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/ 存在")
        else:
            print(f"⚠️  {dir_name}/ 不存在，将自动创建")
            os.makedirs(dir_name, exist_ok=True)
    
    # 检查核心文件
    for file_name in core_files:
        if os.path.exists(file_name):
            print(f"✅ {file_name} 存在")
        else:
            print(f"❌ {file_name} 缺失")
            all_good = False
    
    return all_good

def main():
    """主函数"""
    show_header()
    
    print("请选择要执行的操作：")
    print("1. 🔍 检查环境")
    print("2. ⬇️  安装依赖")
    print("3. 🧪 完整测试")
    print("4. 🎬 创建测试视频")
    print("5. 📁 检查项目结构")
    print("6. 🚀 全部执行")
    
    choice = input("\n请输入选项 (1-6): ").strip()
    
    if choice == "1":
        # 检查环境
        check_python_version()
        test_basic_libraries()
        test_opencv_features()
        
    elif choice == "2":
        # 安装依赖
        if install_dependencies():
            print("\n✅ 依赖安装完成")
        else:
            print("\n❌ 依赖安装失败")
            
    elif choice == "3":
        # 完整测试
        if not check_python_version():
            return
        
        success, missing = test_basic_libraries()
        if not success:
            print(f"\n❌ 缺少依赖: {', '.join(missing)}")
            print("请先运行选项2安装依赖")
            return
            
        if test_opencv_features():
            print("\n🎉 所有测试通过！环境配置正确！")
        else:
            print("\n❌ 测试失败")
            
    elif choice == "4":
        # 创建测试视频
        create_test_video()
        
    elif choice == "5":
        # 检查项目结构
        check_project_structure()
        
    elif choice == "6":
        # 全部执行
        print("\n🚀 执行完整的环境配置和测试...")
        
        if not check_python_version():
            return
            
        # 安装依赖
        success, missing = test_basic_libraries()
        if not success:
            print("\n开始安装缺失的依赖...")
            if not install_dependencies():
                return
        
        # 重新测试
        success, _ = test_basic_libraries()
        if not success:
            print("\n❌ 安装后仍有问题")
            return
            
        # 测试OpenCV功能
        if not test_opencv_features():
            return
            
        # 检查项目结构
        check_project_structure()
        
        # 创建测试视频
        create_test_video()
        
        print("\n" + "=" * 60)
        print("🎉 环境配置完成！现在可以运行：")
        print("   python main.py")
        print("=" * 60)
        
    else:
        print("❌ 无效选项")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    
    input("\n按任意键退出...")
