#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI后端服务 - 简化版（本地测试）
支持视频上传、处理模式选择、实时处理
"""
import os
import sys
import uuid
import shutil
import asyncio
import threading
import subprocess
import zipfile
import io
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 创建FastAPI应用
app = FastAPI(
    title="Speed Estimation API",
    description="视频速度估算Web服务",
    version="1.0.0"
)

# 允许跨域（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置路径（统一到项目根目录的data文件夹）
PROJECT_ROOT = Path(__file__).parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data/web/uploads"
OUTPUT_DIR = PROJECT_ROOT / "data/web/outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 任务状态存储（简化版，生产环境应使用数据库）
tasks = {}

# 任务取消标志
cancel_flags = {}

# 存储处理线程对象
task_threads = {}


# 数据模型
class ProcessRequest(BaseModel):
    video_id: str
    mode: int  # 1-6
    show_visualization: bool = True
    focal_mm: Optional[float] = None      # Mode 5/6: 等效焦段(mm)，默认50(Mode5)/24(Mode6)
    depth_frequency: Optional[int] = None  # Mode 5: 深度更新频率，默认5
    apply_enhancement: bool = False       # 是否启用预处理增强
    enhancement_options: Optional[list] = None  # ["blur", "haze", "brightness"]


class DetectQualityRequest(BaseModel):
    video_id: str
    quick: bool = False  # 快速检测（少量采样）


class EnhanceRequest(BaseModel):
    video_id: str
    enhancement_options: list  # ["blur", "haze", "brightness"]


class TaskStatus(BaseModel):
    task_id: str
    status: str  # uploading, processing, completed, failed
    progress: int = 0
    message: str = ""
    video_id: Optional[str] = None
    output_path: Optional[str] = None
    created_at: str
    

# ==================== API端点 ====================

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "Speed Estimation API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    上传视频文件
    """
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="不支持的视频格式")
        
        # 生成唯一ID
        video_id = str(uuid.uuid4())[:8]
        file_ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{video_id}{file_ext}"
        
        # 保存文件
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 获取文件信息
        file_size = save_path.stat().st_size
        
        return {
            "success": True,
            "video_id": video_id,
            "filename": file.filename,
            "size": file_size,
            "message": "视频上传成功"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/process")
async def start_process(request: ProcessRequest):
    """
    开始处理视频（异步）

    如果 apply_enhancement=True，则先将视频预处理增强，
    再将增强后的视频送入主 pipeline。
    """
    try:
        video_id = request.video_id
        mode = request.mode

        # 查找上传的视频
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="视频文件不存在")

        input_path = video_files[0]
        processed_input_path = str(input_path)
        base_ext = video_files[0].suffix
        applied_enhancement = None

        # === 预处理增强阶段 ===
        if request.apply_enhancement and request.enhancement_options:
            enhancement_options = request.enhancement_options
            try:
                from quality_detector import detect_video_quality
                from enhance_video import enhance_video
            except ImportError as e:
                raise HTTPException(status_code=500, detail=f"增强模块加载失败: {str(e)}")
            enhanced_path = UPLOAD_DIR / f"{video_id}_enhanced{base_ext}"

            # 自适应检测（用于参数调优，较快，在主线程执行）
            report = detect_video_quality(str(input_path))

            # 执行增强较慢，放到线程池中避免阻塞 FastAPI 事件循环
            def _do_enhance():
                return enhance_video(
                    input_path=str(input_path),
                    output_path=str(enhanced_path),
                    issues=enhancement_options,
                    quality_report=report,
                    brightness_level=report.brightness_level
                )

            success, applied = await asyncio.to_thread(_do_enhance)

            if not success or not enhanced_path.exists():
                raise HTTPException(status_code=500, detail="视频预处理增强失败")

            processed_input_path = str(enhanced_path)
            applied_enhancement = applied

        # 获取视频总帧数（用于进度显示）
        import cv2
        cap = cv2.VideoCapture(processed_input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 创建任务
        task_id = str(uuid.uuid4())[:8]
        tasks[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "progress": 0,
            "message": "正在处理...",
            "video_id": video_id,
            "mode": mode,
            "input_path": processed_input_path,
            "original_input_path": str(input_path),
            "output_path": None,
            "applied_enhancement": applied_enhancement,
            "created_at": datetime.now().isoformat()
        }

        # 使用subprocess启动独立进程（可以强制终止）
        python_path = sys.executable
        script_path = Path(__file__).parent / "process_worker.py"

        # 收集可选参数
        extra_args = []
        if request.focal_mm is not None:
            extra_args.extend(['--focal-mm', str(request.focal_mm)])
        if request.depth_frequency is not None:
            extra_args.extend(['--depth-freq', str(request.depth_frequency)])

        # 启动子进程（捕获stdout实时输出）
        process = subprocess.Popen(
            [python_path, str(script_path), task_id, processed_input_path, str(mode), str(OUTPUT_DIR)] + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout
            text=True,
            encoding='utf-8',  # ✅ 显式指定UTF-8编码（修复GBK错误）
            errors='replace',  # ✅ 遇到无法解码的字符时用�替代（避免崩溃）
            bufsize=1,  # 行缓冲
            cwd=str(PROJECT_ROOT),  # ✅ 设置工作目录为项目根目录（修复models路径问题）
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
        )
        
        # 保存进程对象
        task_threads[task_id] = process
        
        # 启动stdout监控线程（解析进度）
        monitor_thread = threading.Thread(
            target=monitor_stdout_progress,
            args=(task_id, process, total_frames)
        )
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # 启动进程完成监控线程
        completion_thread = threading.Thread(
            target=monitor_process_completion,
            args=(task_id, process)
        )
        completion_thread.daemon = True
        completion_thread.start()
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "处理任务已启动",
            "applied_enhancement": applied_enhancement
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动处理失败: {str(e)}")


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    查询任务状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]

    # 处理完成时，扫描输出目录下的 CSV 文件和 crops 截图目录
    csv_files = []
    crop_files = []
    if task["status"] == "completed" and OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob(f"{task_id}_output*.csv"):
            csv_files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "url": f"/api/files/{task_id}/{f.name}"
            })
        # 扫描截图目录
        crops_dir = OUTPUT_DIR / f"{task_id}_output_crops"
        if crops_dir.exists():
            for f in sorted(crops_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    crop_files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "url": f"/api/files/{task_id}/{task_id}_output_crops/{f.name}"
                    })

    return {
        **task,
        "csv_files": csv_files,
        "crop_files": crop_files,
        "zip_url": f"/api/download-zip/{task_id}",
    }


@app.get("/api/files/{task_id}/{filepath:.+}")
async def download_file(task_id: str, filepath: str):
    """
    下载指定任务相关的任意文件（CSV 或截图）
    支持:
      /api/files/{id}/xxx.csv
      /api/files/{id}/xxx_crops/xxx.jpg
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    # filepath 可能是 "xxx.csv" 或 "xxx_crops/xxx.jpg"
    file_path = OUTPUT_DIR / filepath
    if not file_path.exists() or not str(file_path).startswith(str(OUTPUT_DIR)):
        raise HTTPException(status_code=404, detail="文件不存在")

    filename = file_path.name
    media_type_map = {
        '.csv':  'text/csv',
        '.jpg':  'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png':  'image/png',
        '.mp4':  'video/mp4',
    }
    media_type = next((v for ext, v in media_type_map.items() if filename.lower().endswith(ext)), 'application/octet-stream')

    return FileResponse(file_path, media_type=media_type, filename=filename)


@app.get("/api/download-enhanced/{video_id}")
async def download_enhanced_video(video_id: str):
    """
    下载预处理增强后的视频
    """
    # 查找增强视频（原始id + _enhanced 后缀）
    video_files = list(UPLOAD_DIR.glob(f"{video_id}_enhanced.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="增强视频不存在，请先执行预处理")

    enhanced_path = video_files[0]
    if not enhanced_path.exists():
        raise HTTPException(status_code=404, detail="增强视频文件不存在")

    stem = enhanced_path.stem  # e.g. "abc123_enhanced"
    ext = enhanced_path.suffix  # e.g. ".mp4"
    filename = f"{stem}{ext}"

    return FileResponse(
        str(enhanced_path),
        media_type="video/mp4",
        filename=filename
    )


@app.get("/api/download-original/{video_id}")
async def download_original_video(video_id: str):
    """
    下载原始上传视频（用于增强前后对比）
    """
    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="原始视频不存在")

    original_path = video_files[0]
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="原始视频文件不存在")

    filename = f"{original_path.stem}{original_path.suffix}"
    return FileResponse(
        str(original_path),
        media_type="video/mp4",
        filename=filename
    )


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """
    下载处理结果
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="视频还未处理完成")

    output_path = task["output_path"]
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="输出文件不存在")

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"result_{task_id}.mp4"
    )


@app.get("/api/download-zip/{task_id}")
async def download_data_zip(task_id: str):
    """
    下载任务的所有数据（CSV + crops 截图），打包为 ZIP
    所有 Mode 统一接口：Mode 5 包含 2个CSV + crops；其他 Mode 各 1个CSV
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="视频还未处理完成")

    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=404, detail="输出目录不存在")

    # 收集所有文件
    files_to_zip = []

    # 1. 处理后的视频文件（所有 Mode）
    task = tasks[task_id]
    output_video = task.get("output_path")
    if output_video and Path(output_video).exists():
        files_to_zip.append((Path(output_video), f"processed_video.mp4"))

    # 2. CSV 文件
    for csv_file in OUTPUT_DIR.glob(f"{task_id}_output*.csv"):
        files_to_zip.append((csv_file, csv_file.name))

    # 3. crops 截图目录（仅 Mode 5 有）
    crops_dir = OUTPUT_DIR / f"{task_id}_output_crops"
    if crops_dir.exists():
        for crop_file in sorted(crops_dir.iterdir()):
            if crop_file.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                rel_path = f"crops/{crop_file.name}"
                files_to_zip.append((crop_file, rel_path))

    if not files_to_zip:
        raise HTTPException(status_code=404, detail="没有找到可下载的数据文件")

    # 写入内存 ZIP 流
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path, arc_name in files_to_zip:
            zf.write(file_path, arc_name)

    zip_buffer.seek(0)
    zip_name = f"data_{task_id}.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{zip_name}"}
    )


@app.get("/api/history")
async def get_history():
    """
    获取处理历史
    """
    history = []
    for task_id, task in tasks.items():
        history.append({
            "task_id": task_id,
            "video_id": task.get("video_id"),
            "mode": task.get("mode"),
            "status": task["status"],
            "created_at": task.get("created_at"),
            "output_available": task["status"] == "completed"
        })
    
    # 按创建时间倒序
    history.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"tasks": history}


# ==================== 视频质量检测与预处理接口 ====================

@app.post("/api/detect-quality")
async def detect_quality(request: DetectQualityRequest):
    """
    检测视频质量（模糊/雾/亮度）
    检测完成后返回结构化报告，无需修改原视频。
    """
    try:
        video_id = request.video_id

        # 查找上传的视频
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="视频文件不存在")

        input_path = video_files[0]

        # 导入检测模块（懒加载）
        try:
            from quality_detector import detect_video_quality, quick_detect
        except ImportError:
            raise HTTPException(status_code=500, detail="质量检测模块加载失败，请检查 src/quality_detector.py")

        # 执行检测
        if request.quick:
            report = quick_detect(str(input_path))
        else:
            report = detect_video_quality(str(input_path))

        return {
            "success": True,
            "report": report.to_dict(),
            "message": "检测完成"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/api/enhance")
async def enhance_video_endpoint(request: EnhanceRequest):
    """
    预处理视频（去雾/去模糊/提亮）
    在原视频上执行预处理增强，生成增强版视频。
    增强后的视频路径将记录到临时存储，供后续 pipeline 使用。
    注意：视频处理较慢，异步在线程池中执行，不阻塞 FastAPI 事件循环。
    """
    try:
        video_id = request.video_id
        enhancement_options = request.enhancement_options

        # 查找上传的视频
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="视频文件不存在")

        input_path = video_files[0]
        base_ext = video_files[0].suffix
        enhanced_path = UPLOAD_DIR / f"{video_id}_enhanced{base_ext}"

        # 导入增强模块（懒加载）
        try:
            from quality_detector import detect_video_quality
            from enhance_video import enhance_video
        except ImportError as e:
            raise HTTPException(status_code=500, detail=f"增强模块加载失败: {str(e)}")

        # 自适应检测（较快，20帧采样，在主线程执行）
        report = detect_video_quality(str(input_path))

        # 视频增强较慢，放到线程池中异步执行，避免阻塞事件循环
        def _do_enhance():
            return enhance_video(
                input_path=str(input_path),
                output_path=str(enhanced_path),
                issues=enhancement_options,
                quality_report=report,
                brightness_level=report.brightness_level
            )

        success, applied = await asyncio.to_thread(_do_enhance)

        if not success:
            raise HTTPException(status_code=500, detail="视频增强处理失败")

        return {
            "success": True,
            "enhanced_video_id": f"{video_id}_enhanced",
            "enhanced_video_path": str(enhanced_path),
            "applied_methods": applied,
            "message": f"增强完成：{', '.join(applied)}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增强失败: {str(e)}")


@app.post("/api/cancel/{task_id}")
async def cancel_task(task_id: str):
    """
    取消正在处理的任务
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if tasks[task_id]["status"] == "processing":
        # 设置取消标志
        cancel_flags[task_id] = True
        tasks[task_id]["status"] = "cancelled"
        tasks[task_id]["message"] = "任务已取消"
        
        # 强制终止进程
        if task_id in task_threads:
            process = task_threads[task_id]
            try:
                # Windows使用taskkill，Linux使用kill
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], 
                                  capture_output=True, timeout=3)
                else:
                    process.terminate()
                    process.wait(timeout=2)
                print(f"✅ 已终止任务进程 PID: {process.pid}")
            except Exception as e:
                print(f"终止进程失败: {e}")
            finally:
                del task_threads[task_id]
        
        # 删除输出文件
        output_path = OUTPUT_DIR / f"{task_id}_output.mp4"
        if output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                print(f"删除输出文件失败: {e}")
        
        return {"success": True, "message": "✅ 任务已取消，处理进程已立即终止"}
    else:
        return {"success": False, "message": f"任务状态为{tasks[task_id]['status']}，无法取消"}


# ==================== 视频处理函数 ====================

def monitor_stdout_progress(task_id: str, process: subprocess.Popen, total_frames: int):
    """
    实时读取subprocess的stdout，解析进度
    """
    import re
    import time
    
    start_time = time.time()
    last_frame = 0
    
    try:
        for line in process.stdout:
            # 输出到FastAPI的终端
            print(line, end='')
            
            # 解析Frame输出
            # 支持两种格式：
            # Mode 1: "Frame 100: Tracking 3 objects"
            # Mode 2: "Frame 100/900: 3 objects" (包含总帧数)
            
            # 尝试匹配 Mode 2 格式（带总帧数）
            match_with_total = re.search(r'Frame (\d+)/(\d+):', line)
            if match_with_total:
                current_frame = int(match_with_total.group(1))
                total_in_output = int(match_with_total.group(2))
                last_frame = current_frame
                
                # ✅ 使用输出中的总帧数（更准确）
                progress = (current_frame / total_in_output) * 100
                tasks[task_id]["progress"] = round(progress, 1)
                
                # 计算实时速度
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = current_frame / elapsed
                    tasks[task_id]["message"] = f"处理中... ({fps:.1f} 帧/秒)"
            else:
                # 尝试匹配 Mode 1 格式（不带总帧数）
                match = re.search(r'Frame (\d+):', line)
                if match:
                    current_frame = int(match.group(1))
                    last_frame = current_frame
                    
                    if total_frames > 0:
                        progress = (current_frame / total_frames) * 100
                        tasks[task_id]["progress"] = round(progress, 1)
                        
                        # 计算实时速度
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            fps = current_frame / elapsed
                            tasks[task_id]["message"] = f"处理中... ({fps:.1f} 帧/秒)"
            
            # 检查是否被取消
            if cancel_flags.get(task_id, False):
                break
                
    except Exception as e:
        print(f"监控stdout出错: {e}")
    finally:
        # 确保进度到达100%
        if tasks.get(task_id, {}).get("status") == "completed":
            tasks[task_id]["progress"] = 100.0


def monitor_process_completion(task_id: str, process: subprocess.Popen):
    """
    监控子进程完成状态
    """
    try:
        # 等待进程完成
        return_code = process.wait()
        
        # 检查输出文件
        output_path = OUTPUT_DIR / f"{task_id}_output.mp4"
        
        if return_code == 0 and output_path.exists():
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["progress"] = 100.0
            tasks[task_id]["message"] = "处理完成"
            tasks[task_id]["output_path"] = str(output_path)
        elif return_code == 2:
            # 被中断
            tasks[task_id]["status"] = "cancelled"
            tasks[task_id]["message"] = "任务已取消"
        else:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["progress"] = 0.0
            tasks[task_id]["message"] = f"处理失败（退出码: {return_code}）"
    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"监控失败: {str(e)}"
    finally:
        # 清理进程对象
        if task_id in task_threads:
            del task_threads[task_id]


# ==================== 启动服务 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Speed Estimation API 启动中...")
    print("=" * 60)
    print(f"📁 上传目录: {UPLOAD_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print(f"🌐 API文档: http://localhost:8000/docs")
    print(f"🌐 前端访问: http://localhost:3000")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
