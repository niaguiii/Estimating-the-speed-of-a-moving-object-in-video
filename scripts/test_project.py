#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目测试工具 - 统一测试脚本
整合GPU检查、依赖测试、RAFT测试等功能
"""
import os
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def test_gpu():
    """测试GPU状态"""
    print("\n" + "=" * 70)
    print("🎮 GPU状态检查")
    print("=" * 70)
    
    try:
        import torch
        print(f"\n  PyTorch版本: {torch.__version__}")
        print(f"  CUDA可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")
        
        if torch.cuda.is_available():
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("  ⚠️ GPU不可用，将使用CPU模式")
            return False
    except ImportError:
        print("  ❌ PyTorch未安装")
        return False


def test_dependencies():
    """测试项目依赖"""
    print("\n" + "=" * 70)
    print("📦 依赖包检查")
    print("=" * 70)
    
    dependencies = [
        ("numpy", "数值计算"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch深度学习"),
        ("torchvision", "PyTorch视觉"),
        ("ultralytics", "YOLOv8"),
        ("supervision", "追踪工具"),
        ("PIL", "图像处理"),
        ("transformers", "Hugging Face"),
        ("timm", "图像模型"),
    ]
    
    all_installed = True
    for package, desc in dependencies:
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "PIL":
                from PIL import Image
                version = Image.__version__ if hasattr(Image, '__version__') else "已安装"
            else:
                module = __import__(package)
                version = getattr(module, '__version__', '未知版本')
            
            print(f"  ✅ {desc:20} ({package:15}) - v{version}")
        except ImportError:
            print(f"  ❌ {desc:20} ({package:15}) - 未安装")
            all_installed = False
    
    return all_installed


def test_raft_simple():
    """简单测试RAFT光流"""
    print("\n" + "=" * 70)
    print("🌊 RAFT光流快速测试")
    print("=" * 70)
    
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    
    try:
        print("\n  1️⃣ 导入RAFT模块...")
        from optical_flow_raft import RAFTOpticalFlow
        print("     ✅ RAFT模块导入成功")
        
        print("\n  2️⃣ 创建测试数据...")
        import numpy as np
        frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("     ✅ 测试帧创建成功")
        
        print("\n  3️⃣ 初始化RAFT...")
        raft = RAFTOpticalFlow(model_type='small')
        print("     ✅ RAFT初始化成功")
        
        print("\n  4️⃣ 计算光流...")
        flow = raft.compute_flow(frame1, frame2)
        print(f"     ✅ 光流计算成功，形状: {flow.shape}")
        
        print("\n  5️⃣ 估计摄像头运动...")
        camera_motion = raft.estimate_camera_motion(flow)
        print(f"     ✅ 摄像头运动估计成功: {camera_motion}")
        
        print("\n  ✅ RAFT光流测试通过！")
        return True
        
    except Exception as e:
        print(f"\n  ❌ RAFT测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_depth_simple():
    """简单测试Depth Anything V2"""
    print("\n" + "=" * 70)
    print("🏔️ Depth Anything V2 快速测试")
    print("=" * 70)
    
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    
    try:
        print("\n  1️⃣ 导入Depth模块...")
        from depth_estimation import DepthAnythingV2
        print("     ✅ Depth模块导入成功")
        
        print("\n  2️⃣ 创建测试图像...")
        import numpy as np
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("     ✅ 测试图像创建成功")
        
        print("\n  3️⃣ 初始化Depth Anything V2...")
        print("     ⚠️ 首次使用会下载模型（~100MB），请耐心等待...")
        depth_estimator = DepthAnythingV2(model_size='small')
        print("     ✅ Depth模型初始化成功")
        
        print("\n  4️⃣ 估计深度...")
        depth_map = depth_estimator.estimate_depth(frame)
        print(f"     ✅ 深度估计成功，形状: {depth_map.shape}")
        
        print("\n  ✅ Depth Anything V2测试通过！")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Depth测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_yolo():
    """测试YOLOv8"""
    print("\n" + "=" * 70)
    print("🎯 YOLOv8检测测试")
    print("=" * 70)
    
    try:
        print("\n  1️⃣ 导入YOLOv8...")
        from ultralytics import YOLO
        print("     ✅ Ultralytics导入成功")
        
        print("\n  2️⃣ 加载YOLOv8n模型...")
        model = YOLO('yolov8n.pt')
        print("     ✅ YOLOv8模型加载成功")
        
        print("\n  3️⃣ 创建测试图像...")
        import numpy as np
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("     ✅ 测试图像创建成功")
        
        print("\n  4️⃣ 运行检测...")
        results = model(frame, verbose=False)
        print(f"     ✅ 检测完成，检测到 {len(results[0].boxes)} 个物体")
        
        print("\n  ✅ YOLOv8测试通过！")
        return True
        
    except Exception as e:
        print(f"\n  ❌ YOLOv8测试失败: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="项目测试工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:
  gpu      - GPU状态测试
  deps     - 依赖包测试
  yolo     - YOLOv8测试
  raft     - RAFT光流测试
  depth    - Depth Anything V2测试
  phase3   - Phase 3完整测试 (RAFT + Depth)
  all      - 所有测试

示例:
  python scripts/test_project.py --mode gpu    # 仅测试GPU
  python scripts/test_project.py --mode phase3 # 测试Phase 3
  python scripts/test_project.py --mode all    # 全面测试
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['gpu', 'deps', 'yolo', 'raft', 'depth', 'phase3', 'all'],
        default='deps',
        help='测试模式 (默认: deps)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🧪 项目测试工具")
    print("=" * 70)
    print(f"测试模式: {args.mode}")
    
    results = []
    
    # 根据模式执行测试
    if args.mode in ['gpu', 'all']:
        results.append(("GPU状态", test_gpu()))
    
    if args.mode in ['deps', 'all']:
        results.append(("依赖包", test_dependencies()))
    
    if args.mode in ['yolo', 'all']:
        results.append(("YOLOv8", test_yolo()))
    
    if args.mode in ['raft', 'phase3', 'all']:
        results.append(("RAFT光流", test_raft_simple()))
    
    if args.mode in ['depth', 'phase3', 'all']:
        results.append(("Depth V2", test_depth_simple()))
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 测试结果总结")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {test_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("⚠️ 部分测试失败，请查看上述详细信息。")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
