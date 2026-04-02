#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目状态检查工具 - 统一检查脚本
支持多种检查模式
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
        ("cfg", "配置文件目录"),
        ("data", "数据目录"),
        ("docs", "文档目录"),
        ("scripts", "脚本目录"),
        ("web", "Web应用目录"),
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


def check_core_modules():
    """检查核心模块文件"""
    print("\n" + "=" * 70)
    print("📦 核心模块检查")
    print("=" * 70)
    
    files = [
        ("main.py", "主程序入口"),
        ("src/__init__.py", "模块初始化"),
        ("src/config.py", "配置文件"),
        ("src/model_config.py", "模型配置"),
        ("src/mode1_detection_tracking.py", "Mode 1: 检测+追踪"),
        ("src/mode2_speed_estimation.py", "Mode 2: 速度估算"),
        ("src/mode3_raft_optical_flow.py", "Mode 3: RAFT光流"),
        ("src/mode4_depth_anything_v2.py", "Mode 4: Depth Anything V2"),
        ("src/mode5_metric3d_v2.py", "Mode 5: Metric3D v2"),
        ("src/mode6_ego_speed.py", "Mode 6: 自车测速"),
        ("src/optical_flow_raft.py", "RAFT光流封装"),
        ("src/depth_estimation.py", "Depth Anything V2封装"),
        ("src/depth_estimation_metric3d.py", "Metric3D v2封装"),
        ("src/enhance_video.py", "视频增强模块"),
        ("src/quality_detector.py", "质量检测模块"),
        ("src/main_opencv.py", "遗留: OpenCV检测"),
        ("src/main_yolov8_native.py", "遗留: YOLOv8原生"),
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


def check_web_modules():
    """检查Web模块"""
    print("\n" + "=" * 70)
    print("🌐 Web模块检查")
    print("=" * 70)
    
    files = [
        ("web/backend/app.py", "后端主程序"),
        ("web/backend/process_worker.py", "处理Worker"),
        ("web/frontend/src/App.vue", "前端根组件"),
        ("web/frontend/src/api/index.js", "API封装"),
        ("web/frontend/src/components/VideoUpload.vue", "上传组件"),
        ("web/frontend/src/components/ModeSelector.vue", "模式选择组件"),
        ("web/frontend/src/components/ProgressBar.vue", "进度条组件"),
        ("web/frontend/src/components/ResultDisplay.vue", "结果展示组件"),
    ]
    
    all_exist = True
    for file_path, desc in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    return all_exist


def check_scripts():
    """检查脚本文件"""
    print("\n" + "=" * 70)
    print("🛠️ 脚本文件检查")
    print("=" * 70)
    
    files = [
        ("scripts/check_project.py", "项目状态检查"),
        ("scripts/test_project.py", "项目测试工具"),
        ("scripts/README_SCRIPTS.md", "脚本目录说明"),
        ("scripts/cpu/setup_and_test.py", "CPU环境测试"),
        ("scripts/cpu/requirements.txt", "CPU依赖列表"),
        ("scripts/gpu/install_gpu.bat", "GPU安装脚本"),
        ("scripts/gpu/switch_to_gpu.bat", "GPU切换脚本"),
    ]
    
    all_exist = True
    for file_path, desc in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    return all_exist


def check_models():
    """检查模型目录"""
    print("\n" + "=" * 70)
    print("🤖 模型文件检查")
    print("=" * 70)
    
    files = [
        ("cfg/bytetrack_stable.yaml", "ByteTrack配置"),
        ("models/coco.names", "COCO类别名称"),
        ("models/README.md", "模型说明文档"),
    ]
    
    all_exist = True
    for file_path, desc in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  ✅ {desc}")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    # 检查YOLOv8模型
    yolo_path = PROJECT_ROOT / "models" / "yolov8n.pt"
    if yolo_path.exists():
        size_mb = yolo_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ YOLOv8模型: yolov8n.pt ({size_mb:.1f}MB)")
    else:
        print(f"  ⚠️ YOLOv8模型未找到（运行时自动下载）")
    
    return all_exist


def check_docs():
    """检查文档文件"""
    print("\n" + "=" * 70)
    print("📚 文档文件检查")
    print("=" * 70)
    
    files = [
        ("README.md", "项目主文档"),
        ("docs/PROJECT_STRUCTURE.md", "项目结构文档"),
        ("docs/TECHNICAL_REPORT.md", "技术原理报告"),
        ("docs/FYP_Progress_Report_2nd.md", "阶段进度报告"),
        ("web/README.md", "Web使用说明"),
    ]
    
    all_exist = True
    for file_path, desc in files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✅ {desc} ({size_kb:.1f}KB)")
        else:
            print(f"  ❌ {desc}: {file_path} [缺失]")
            all_exist = False
    
    return all_exist


def test_imports():
    """测试核心模块导入"""
    print("\n" + "=" * 70)
    print("🔧 模块导入测试")
    print("=" * 70)
    
    # 添加src到路径
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    
    modules_to_test = [
        ("config", "配置文件"),
        ("model_config", "模型配置"),
        ("quality_detector", "质量检测"),
        ("enhance_video", "视频增强"),
    ]
    
    all_success = True
    for module_name, desc in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {desc} ({module_name})")
        except ImportError as e:
            print(f"  ❌ {desc} ({module_name}): {e}")
            all_success = False
    
    return all_success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="项目状态检查工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
检查模式:
  basic    - 基本结构检查
  core     - 核心模块检查
  web      - Web模块检查
  scripts  - 脚本文件检查
  models   - 模型文件检查
  docs     - 文档文件检查
  full     - 完整检查（包括导入测试）
  all      - 所有检查

示例:
  python scripts/check_project.py              # 快速检查基本结构
  python scripts/check_project.py --mode core  # 检查核心模块
  python scripts/check_project.py --mode full  # 完整检查
  python scripts/check_project.py --mode all   # 全面检查
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['basic', 'core', 'web', 'scripts', 'models', 'docs', 'full', 'all'],
        default='basic',
        help='检查模式 (默认: basic)'
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
    
    if args.mode in ['core', 'full', 'all']:
        results.append(("核心模块", check_core_modules()))
    
    if args.mode in ['web', 'full', 'all']:
        results.append(("Web模块", check_web_modules()))
    
    if args.mode in ['scripts', 'full', 'all']:
        results.append(("脚本文件", check_scripts()))
    
    if args.mode in ['models', 'full', 'all']:
        results.append(("模型文件", check_models()))
    
    if args.mode in ['docs', 'full', 'all']:
        results.append(("文档文件", check_docs()))
    
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
