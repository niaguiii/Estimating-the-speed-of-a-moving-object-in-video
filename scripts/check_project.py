#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目状态检查工具 - 统一检查脚本
支持多种检查模式，替代原有的多个独立检查脚本
"""
import os
import sys
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


def check_basic_structure():
    """检查基本项目结构"""
    print("\n" + "=" * 70)
    print("📁 基本项目结构检查")
    print("=" * 70)
    
    required_dirs = [
        ("src", "核心代码目录"),
        ("models", "模型文件目录"),
        ("input", "输入视频目录"),
        ("output", "输出结果目录"),
        ("docs", "文档目录"),
        ("scripts", "脚本目录"),
        ("trackers", "追踪器目录"),
    ]
    
    all_exist = True
    for dir_name, desc in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print(f"  ✅ {desc}: {dir_name}/")
        else:
            print(f"  ❌ {desc}: {dir_name}/ [缺失]")
            all_exist = False
    
    return all_exist


def check_phase1_2():
    """检查Phase 1 & 2文件"""
    print("\n" + "=" * 70)
    print("📦 Phase 1 & 2 模块检查")
    print("=" * 70)
    
    files = [
        ("main.py", "主程序入口"),
        ("src/config.py", "配置文件"),
        ("src/main_opencv.py", "Phase 1 - OpenCV实现"),
        ("src/main_yolov8_bytetrack.py", "Phase 2 - ByteTrack追踪"),
        ("src/main_yolov8_speed.py", "Phase 2 - 速度估算"),
        ("trackers/byte_tracker.py", "ByteTrack封装"),
        ("trackers/simple_tracker.py", "简单追踪器"),
    ]
    
    all_exist = True
    for file_path, desc in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✅ {desc}")
            print(f"     {file_path} ({size_kb:.1f}KB)")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    return all_exist


def check_phase3_modules():
    """检查Phase 3核心模块"""
    print("\n" + "=" * 70)
    print("🚀 Phase 3 核心模块检查")
    print("=" * 70)
    
    modules = [
        ("src/optical_flow_raft.py", "RAFT光流模块"),
        ("src/depth_estimation.py", "Depth Anything V2模块"),
        ("src/main_yolov8_raft.py", "RAFT集成版本"),
        ("src/main_phase3_complete.py", "Phase 3完整版本"),
    ]
    
    all_exist = True
    for file_path, desc in modules:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✅ {desc}")
            print(f"     {file_path} ({size_kb:.1f}KB)")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    return all_exist


def check_phase3_models():
    """检查Phase 3预训练模型"""
    print("\n" + "=" * 70)
    print("📦 Phase 3 预训练模型检查")
    print("=" * 70)
    
    # PyTorch缓存
    torch_cache = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    print(f"\n  PyTorch缓存: {torch_cache}")
    
    if torch_cache.exists():
        raft_models = list(torch_cache.glob("raft_*.pth"))
        if raft_models:
            for model in raft_models:
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"    ✅ {model.name} ({size_mb:.1f}MB)")
        else:
            print(f"    ⚪ RAFT模型未下载（首次使用时自动下载）")
    else:
        print(f"    ⚪ PyTorch缓存目录不存在")
    
    # Hugging Face缓存
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"\n  Hugging Face缓存: {hf_cache}")
    
    if hf_cache.exists():
        depth_models = [d for d in hf_cache.iterdir() if "depth-anything" in d.name.lower()]
        if depth_models:
            for model in depth_models:
                print(f"    ✅ {model.name}")
        else:
            print(f"    ⚪ Depth Anything V2未下载（首次使用时自动下载）")
    else:
        print(f"    ⚪ Hugging Face缓存目录不存在")


def check_phase3_integration():
    """检查Phase 3在main.py中的集成"""
    print("\n" + "=" * 70)
    print("🔗 Phase 3 集成检查")
    print("=" * 70)
    
    main_py = PROJECT_ROOT / "main.py"
    if not main_py.exists():
        print("  ❌ main.py不存在")
        return False
    
    content = main_py.read_text(encoding='utf-8')
    
    checks = [
        ("'raft'", "RAFT模式选项"),
        ("'phase3'", "Phase 3模式选项"),
        ("main_yolov8_raft", "RAFT模块导入"),
        ("main_phase3_complete", "Phase 3模块导入"),
        ("depth", "深度估计功能"),
    ]
    
    all_integrated = True
    for keyword, desc in checks:
        if keyword in content:
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc} [缺失]")
            all_integrated = False
    
    return all_integrated


def test_imports():
    """测试核心模块导入"""
    print("\n" + "=" * 70)
    print("🔧 模块导入测试")
    print("=" * 70)
    
    # 添加src到路径
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    
    modules_to_test = [
        ("optical_flow_raft", ["RAFTOpticalFlow", "CameraMotionCompensator"]),
        ("depth_estimation", ["DepthAnythingV2", "DepthBasedCalibrator"]),
        ("main_yolov8_raft", ["SpeedEstimatorWithRAFT"]),
        ("main_phase3_complete", ["Phase3SpeedEstimator"]),
    ]
    
    all_success = True
    for module_name, classes in modules_to_test:
        try:
            module = __import__(module_name)
            print(f"\n  ✅ {module_name}")
            for cls_name in classes:
                if hasattr(module, cls_name):
                    print(f"     - {cls_name}")
                else:
                    print(f"     ❌ {cls_name} [缺失]")
                    all_success = False
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            all_success = False
    
    return all_success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="项目状态检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
检查模式:
  basic     - 基本结构检查
  phase1-2  - Phase 1 & 2 检查
  phase3    - Phase 3 快速检查
  full      - 完整检查（包括导入测试）
  all       - 所有检查

示例:
  python scripts/check_project.py             # 快速检查
  python scripts/check_project.py --mode full # 完整检查
  python scripts/check_project.py --mode all  # 全面检查
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['basic', 'phase1-2', 'phase3', 'full', 'all'],
        default='phase3',
        help='检查模式 (默认: phase3)'
    )
    
    parser.add_argument(
        '--test-imports',
        action='store_true',
        help='测试模块导入'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🔍 项目状态检查工具")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"项目根目录: {PROJECT_ROOT}")
    
    results = []
    
    # 根据模式执行检查
    if args.mode in ['basic', 'all']:
        results.append(("基本结构", check_basic_structure()))
    
    if args.mode in ['phase1-2', 'all']:
        results.append(("Phase 1 & 2", check_phase1_2()))
    
    if args.mode in ['phase3', 'full', 'all']:
        results.append(("Phase 3 模块", check_phase3_modules()))
        results.append(("Phase 3 模型", check_phase3_models() or True))  # 模型检查不影响整体
        results.append(("Phase 3 集成", check_phase3_integration()))
    
    if args.mode in ['full', 'all'] or args.test_imports:
        results.append(("模块导入", test_imports()))
    
    # 总结
    print("\n" + "=" * 70)
    print("📊 检查结果总结")
    print("=" * 70)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status} - {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 所有检查通过！项目状态良好。")
    else:
        print("⚠️ 部分检查未通过，请查看上述详细信息。")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
