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
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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

# 质量检测报告缓存（避免同一视频重复采样检测）
# key: video_id, value: report.to_dict()
quality_reports = {}

# 所有全局字典的锁（防止并发读写竞态）
state_lock = threading.RLock()

# 限流：每个 IP 的请求计数和时间窗口
# key: IP address, value: (count, window_start_time)
rate_limit_upload = {}    # /api/upload 限流
rate_limit_process = {}  # /api/process 限流
RATE_WINDOW = 60          # 时间窗口：60 秒
RATE_UPLOAD_MAX = 5       # 60秒内最多5次上传
RATE_PROCESS_MAX = 10     # 60秒内最多10次处理请求

# ZIP 缓存（避免重复打包大文件）
# key: task_id, value: (zip_buffer, created_at)
zip_cache = {}
ZIP_CACHE_TTL = 300       # 缓存有效期：5分钟
zip_cache_lock = threading.Lock()


# 数据模型
class ProcessRequest(BaseModel):
    video_id: str
    mode: int  # 1-6
    show_visualization: bool = True
    focal_mm: Optional[float] = None      # Mode 5/6: 等效焦段(mm)，默认50(Mode5)/24(Mode6)
    depth_frequency: Optional[int] = None  # Mode 5/6: 深度更新频率，默认5
    road_region_ratio: Optional[float] = None  # Mode 6: 路面采样区域比例，默认0.4
    apply_enhancement: bool = False       # 是否启用预处理增强
    enhancement_options: Optional[list] = None  # ["blur", "haze", "brightness"]


class DetectQualityRequest(BaseModel):
    video_id: str
    quick: bool = False  # 快速检测（少量采样）


class EnhanceRequest(BaseModel):
    video_id: str
    enhancement_options: list  # ["blur", "haze", "brightness"]
    mode: Optional[int] = None  # 可选，当前未使用（保留接口扩展性）


class TaskStatus(BaseModel):
    task_id: str
    status: str  # uploading, processing, completed, failed
    progress: int = 0
    message: str = ""
    video_id: Optional[str] = None
    output_path: Optional[str] = None
    created_at: str


# ==================== 限流与清理辅助 ====================

def _get_client_ip(request) -> str:
    """从请求中提取客户端真实 IP（支持代理头）"""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_rate_limit(bucket: dict, key: str, max_requests: int, window: int) -> Optional[str]:
    """
    检查限流桶，返回 None 表示通过，否则返回错误信息字符串。
    """
    now = time.time()
    with state_lock:
        count, window_start = bucket.get(key, (0, now))
        if now - window_start >= window:
            # 窗口过期，重置
            bucket[key] = (1, now)
            return None
        if count >= max_requests:
            reset_in = int(window - (now - window_start))
            return f"请求过于频繁，请 {reset_in} 秒后重试（限流：{max_requests}次/{window}秒）"
        bucket[key] = (count + 1, window_start)
        return None


def _clean_old_tasks(max_age_hours: int = 24):
    """
    清理超过 max_age_hours 的已完成/失败/取消任务，
    及其关联的 output_dir、uploads 和 ZIP 缓存。
    不清理 processing 和 uploading 状态的任务。
    """
    cutoff = time.time() - max_age_hours * 3600
    with state_lock:
        to_delete_task_ids = []
        for tid, task in list(tasks.items()):
            created = task.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created)
                ts = dt.timestamp()
            except Exception:
                continue
            if ts < cutoff and task["status"] in ("completed", "failed", "cancelled"):
                to_delete_task_ids.append(tid)

        for tid in to_delete_task_ids:
            task = tasks[tid]

            # 删除输出目录
            output_dir = OUTPUT_DIR / f"{tid}_output"
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            # 删除输出视频文件
            for ext in ('.mp4', '.avi', '.mov', '.mkv'):
                f = OUTPUT_DIR / f"{tid}_output{ext}"
                if f.exists():
                    f.unlink(missing_ok=True)
            # 删除 CSV 文件
            for f in OUTPUT_DIR.glob(f"{tid}_output*.csv"):
                f.unlink(missing_ok=True)
            # 删除 crops 目录
            crops_dir = OUTPUT_DIR / f"{tid}_output_crops"
            if crops_dir.exists():
                shutil.rmtree(crops_dir, ignore_errors=True)

            # 删除原始上传视频（如果任务还在 uploads 目录）
            video_id = task.get("video_id", "")
            if video_id:
                for f in UPLOAD_DIR.glob(f"{video_id}*"):
                    f.unlink(missing_ok=True)
                # 清除质量缓存
                quality_reports.pop(video_id, None)

            # 清除 ZIP 缓存
            with zip_cache_lock:
                zip_cache.pop(tid, None)

            # 移除任务记录
            tasks.pop(tid, None)
            cancel_flags.pop(tid, None)
            task_threads.pop(tid, None)

        return len(to_delete_task_ids)


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
async def upload_video(request: Request, file: UploadFile = File(...)):
    """
    上传视频文件
    """
    ip = _get_client_ip(request)
    rl_msg = _check_rate_limit(rate_limit_upload, ip, RATE_UPLOAD_MAX, RATE_WINDOW)
    if rl_msg:
        raise HTTPException(status_code=429, detail=rl_msg)

    MAX_SIZE_MB = 500
    MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024

    # 常见视频格式的 Magic Bytes
    VIDEO_SIGNATURES = {
        '.mp4': [b'ftyp', b'free', b'mdat', b'moov'],   # MP4/MOV container
        '.mov': [b'ftyp', b'free', b'mdat', b'moov'],   # MOV (同MP4容器)
        '.avi': [b'RIFF'],                               # AVI
        '.mkv': [b'\x1a\x45\xdf\xa3'],                   # MKV / WebM
    }

    try:
        # 验证文件扩展名
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="不支持的视频格式")

        # 生成唯一ID（完整UUID，防止暴力枚举）
        video_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        save_path = UPLOAD_DIR / f"{video_id}{file_ext}"

        # 流式保存，超限则中断
        bytes_written = 0
        with open(save_path, "wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)   # 每次读1MB
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_SIZE_BYTES:
                    buffer.close()
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件大小超过 {MAX_SIZE_MB}MB 限制"
                    )
                buffer.write(chunk)

        file_size = save_path.stat().st_size

        # Magic Bytes 验证（读文件头）
        valid_sigs = VIDEO_SIGNATURES.get(file_ext.lower(), [])
        if valid_sigs:
            with open(save_path, "rb") as f:
                header = f.read(64)
            sig_found = any(sig in header for sig in valid_sigs)
            if not sig_found:
                save_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="文件内容不是有效视频，请确认视频未损坏")

        return {
            "success": True,
            "video_id": video_id,
            "filename": file.filename,
            "size": file_size,
            "message": "视频上传成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")


@app.post("/api/process")
async def start_process(request: Request, request_body: ProcessRequest):
    """
    开始处理视频（异步）

    如果 apply_enhancement=True，则先将视频预处理增强，
    再将增强后的视频送入主 pipeline。
    """
    ip = _get_client_ip(request)
    rl_msg = _check_rate_limit(rate_limit_process, ip, RATE_PROCESS_MAX, RATE_WINDOW)
    if rl_msg:
        raise HTTPException(status_code=429, detail=rl_msg)

    try:
        video_id = request_body.video_id
        mode = request_body.mode

        # 查找上传的视频
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="视频文件不存在")

        input_path = video_files[0]
        processed_input_path = str(input_path)
        base_ext = video_files[0].suffix
        applied_enhancement = None

        # === 预处理增强阶段 ===
        if request_body.apply_enhancement and request_body.enhancement_options:
            enhancement_options = request_body.enhancement_options
            try:
                from quality_detector import detect_video_quality
                from enhance_video import enhance_video
            except ImportError as e:
                raise HTTPException(status_code=500, detail=f"增强模块加载失败: {str(e)}")
            enhanced_path = UPLOAD_DIR / f"{video_id}_enhanced{base_ext}"

            # 自适应检测（用于参数调优，较快，在主线程执行）
            with state_lock:
                if video_id in quality_reports:
                    report_dict = quality_reports[video_id]
                    from quality_detector import QualityReport
                    report = QualityReport.from_dict(report_dict)
                else:
                    report = detect_video_quality(str(input_path))
                    quality_reports[video_id] = report.to_dict()

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

        # 创建任务（完整UUID，防止枚举）
        task_id = str(uuid.uuid4())
        with state_lock:
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

        # 使用subprocess启动独立进程（可以强制终止），最多重试2次
        MAX_RETRIES = 2
        PROCESS_TIMEOUT_SEC = 3600  # 1小时超时
        python_path = sys.executable
        script_path = Path(__file__).parent / "process_worker.py"

        # 收集可选参数
        extra_args = []
        if request_body.focal_mm is not None:
            extra_args.extend(['--focal-mm', str(request_body.focal_mm)])
        if request_body.depth_frequency is not None:
            extra_args.extend(['--depth-freq', str(request_body.depth_frequency)])
        if request_body.road_region_ratio is not None:
            extra_args.extend(['--road-ratio', str(request_body.road_region_ratio)])

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                # 重试前先终止上一轮残留的进程（避免资源泄漏）
                prev_tid = previous_task_id
                if prev_tid in task_threads:
                    try:
                        old_proc = task_threads[prev_tid]
                        pid = old_proc.pid
                        old_proc.terminate()
                        try:
                            old_proc.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            old_proc.kill()
                        with state_lock:
                            task_threads.pop(prev_tid, None)
                        print(f"[Retry] 已终止残留进程 PID {pid}")
                    except Exception as e:
                        print(f"[Retry] 终止残留进程失败: {e}")
                # 生成新task_id（避免与之前残留文件冲突）
                task_id = str(uuid.uuid4())
                with state_lock:
                    tasks[task_id] = {
                        **tasks.get(original_task_id, {}),
                        "task_id": task_id,
                        "retry": attempt,
                        "message": f"重试第 {attempt} 次...",
                    }
            else:
                original_task_id = task_id

            previous_task_id = task_id  # 记录本次 task_id，供下次循环清理

            # 启动子进程（捕获stdout实时输出）
            process = subprocess.Popen(
                [python_path, str(script_path), task_id, processed_input_path,
                 str(mode), str(OUTPUT_DIR)] + extra_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # 合并stderr到stdout
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )

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

            # 等待本次尝试完成（带超时）
            completion_thread.join(timeout=PROCESS_TIMEOUT_SEC)

            # 超时：强制终止进程
            if completion_thread.is_alive():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                with state_lock:
                    tasks[task_id] = {
                        **tasks.get(task_id, {}),
                        "status": "failed",
                        "message": f"处理超时（超过 {PROCESS_TIMEOUT_SEC} 秒），已强制终止",
                        "progress": 0
                    }

            # 检查结果：成功则退出重试循环
            with state_lock:
                current_status = tasks.get(task_id, {}).get("status")
            if current_status == "completed":
                return {
                    "success": True,
                    "task_id": task_id,
                    "message": "处理任务已完成",
                    "applied_enhancement": applied_enhancement
                }

            # 失败且还有重试机会
            if attempt < MAX_RETRIES:
                with state_lock:
                    tasks[task_id]["message"] = f"处理失败，正在重试 ({attempt}/{MAX_RETRIES})..."
                print(f"[Retry] task_id={task_id} attempt {attempt} failed, retrying...")

        # 全部重试均失败
        raise HTTPException(status_code=500, detail="视频处理多次失败，请检查视频文件或模型配置")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动处理失败: {str(e)}")


@app.get("/api/task/{task_id}")
async def get_task_status(task_id: str):
    """
    查询任务状态
    """
    with state_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 深拷贝，防止返回后外部直接修改内部状态
        task = dict(tasks[task_id])

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
        crops_dir = OUTPUT_DIR / f"{task_id}_output_crops"
        if crops_dir.exists():
            for f in sorted(crops_dir.iterdir()):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    crop_files.append({
                        "name": f.name,
                        "size": f.stat().st_size,
                        "url": f"/api/files/{task_id}/{task_id}_output_crops/{f.name}"
                    })

    # 去除敏感内部路径字段，不泄露给客户端
    for key in ("input_path", "original_input_path"):
        task.pop(key, None)

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

    import os as _os
    # 用 resolve() 先规范化路径（消除 .. 和冗余分隔符），再比较是否越界
    resolved = (OUTPUT_DIR / filepath).resolve()
    safe_base = OUTPUT_DIR.resolve()
    if not str(resolved).startswith(str(safe_base) + _os.sep):
        raise HTTPException(status_code=404, detail="文件不存在")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="文件不存在")

    filename = resolved.name
    media_type_map = {
        '.csv':  'text/csv',
        '.jpg':  'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png':  'image/png',
        '.mp4':  'video/mp4',
    }
    media_type = next((v for ext, v in media_type_map.items() if filename.lower().endswith(ext)), 'application/octet-stream')

    return FileResponse(resolved, media_type=media_type, filename=filename)


@app.get("/api/download-enhanced/{video_id}")
async def download_enhanced_video(video_id: str):
    """
    下载预处理增强后的视频
    video_id 传入原始视频ID（不含_enhanced），函数内部自动追加_enhanced后缀查找文件。
    """
    import os as _os

    # video_id 可能是 "abc123" 或 "abc123_enhanced"（前端有时传增强后的ID）
    # 统一去掉 _enhanced 前缀，再重新拼接
    base_id = video_id.replace('_enhanced', '')
    # video_id 是完整 UUID（含横杠），允许字母/数字/横杠
    if not base_id.replace('-', '').isalnum():
        raise HTTPException(status_code=400, detail="无效的 video_id")

    video_files = list(UPLOAD_DIR.glob(f"{base_id}_enhanced.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="增强视频不存在，请先执行预处理")

    enhanced_path = video_files[0]
    resolved = enhanced_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + _os.sep):
        raise HTTPException(status_code=404, detail="增强视频文件不存在")

    return FileResponse(resolved, media_type="video/mp4", filename=resolved.name)


@app.get("/api/download-original/{video_id}")
async def download_original_video(video_id: str):
    """
    下载原始上传视频（用于增强前后对比）
    """
    import os as _os

    # 过滤危险字符，防止 glob 逃逸到 uploads/ 之外
    # video_id 是完整 UUID（含横杠），允许字母/数字/横杠
    if not video_id.replace('-', '').isalnum():
        raise HTTPException(status_code=400, detail="无效的 video_id")

    video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="原始视频不存在")

    original_path = video_files[0]
    resolved = original_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + _os.sep):
        raise HTTPException(status_code=404, detail="原始视频文件不存在")

    filename = f"{resolved.stem}{resolved.suffix}"
    return FileResponse(resolved, media_type="video/mp4", filename=filename)


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """
    下载处理结果
    """
    import os as _os

    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="视频还未处理完成")

    output_path_str = task.get("output_path", "")
    if not output_path_str:
        raise HTTPException(status_code=404, detail="输出文件路径不存在")

    resolved = Path(output_path_str).resolve()
    if not str(resolved).startswith(str(OUTPUT_DIR.resolve()) + _os.sep):
        raise HTTPException(status_code=404, detail="输出文件路径越界")

    return FileResponse(
        resolved,
        media_type="video/mp4",
        filename=f"result_{task_id}.mp4"
    )


@app.get("/api/download-zip/{task_id}")
async def download_data_zip(task_id: str):
    """
    下载任务的所有数据（CSV + crops 截图），打包为 ZIP
    所有 Mode 统一接口：Mode 5 包含 2个CSV + crops；其他 Mode 各 1个CSV
    支持 5 分钟 ZIP 缓存，避免重复打包大文件。
    """
    import os as _os

    # 1. 先检查缓存
    with zip_cache_lock:
        if task_id in zip_cache:
            cached_buf, cached_at = zip_cache[task_id]
            if time.time() - cached_at < ZIP_CACHE_TTL:
                zip_buffer = io.BytesIO(cached_buf.getvalue())
                zip_name = f"data_{task_id}.zip"
                return StreamingResponse(
                    zip_buffer,
                    media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{zip_name}"}
                )

    # 2. 未命中缓存，构建 ZIP
    with state_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="任务不存在")
        task = dict(tasks[task_id])
        if task["status"] != "completed":
            raise HTTPException(status_code=400, detail="视频还未处理完成")

    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=404, detail="输出目录不存在")

    files_to_zip = []

    # 1. 处理后的视频文件（所有 Mode）
    output_video = task.get("output_path", "")
    if output_video:
        resolved_video = Path(output_video).resolve()
        if resolved_video.exists() and \
                str(resolved_video).startswith(str(OUTPUT_DIR.resolve()) + _os.sep):
            files_to_zip.append((resolved_video, "processed_video.mp4"))

    # 2. CSV 文件
    for csv_file in OUTPUT_DIR.glob(f"{task_id}_output*.csv"):
        files_to_zip.append((csv_file, csv_file.name))

    # 3. crops 截图目录（仅 Mode 5 有）
    # 兼容两种命名方式：
    #   - 新命名：{task_id}_output_crops  （与 process_worker 协调后的新格式）
    #   - 旧命名：{video_name}_crops（历史遗留，{stem}_crops）
    crops_dir = OUTPUT_DIR / f"{task_id}_output_crops"
    if not crops_dir.exists():
        # 回退：搜索历史遗留的 _crops 目录
        for parent in OUTPUT_DIR.glob("*_output.mp4"):
            legacy_dir = parent.with_name(parent.stem + '_crops')
            if legacy_dir.exists():
                crops_dir = legacy_dir
                break
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

    # 3. 写入缓存
    with zip_cache_lock:
        zip_cache[task_id] = (io.BytesIO(zip_buffer.getvalue()), time.time())

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{zip_name}"}
    )


@app.get("/api/history")
async def get_history():
    """
    获取处理历史（不泄露内部路径）
    """
    history = []
    with state_lock:
        for task_id, task in tasks.items():
            history.append({
                "task_id": task_id,
                "video_id": task.get("video_id"),
                "mode": task.get("mode"),
                "status": task["status"],
                "progress": task.get("progress", 0),
                "message": task.get("message", ""),
                "applied_enhancement": task.get("applied_enhancement"),
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

        # 执行检测（先查缓存，避免同一视频重复采样）
        if video_id in quality_reports:
            return {
                "success": True,
                "report": quality_reports[video_id],
                "message": "检测完成（复用缓存）"
            }

        if request.quick:
            report = quick_detect(str(input_path))
        else:
            report = detect_video_quality(str(input_path))

        # 存入缓存
        quality_reports[video_id] = report.to_dict()

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
        # 优先复用已缓存的报告
        if video_id in quality_reports:
            from quality_detector import QualityReport as QR
            report = QR.from_dict(quality_reports[video_id])
        else:
            report = detect_video_quality(str(input_path))
            quality_reports[video_id] = report.to_dict()

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
    with state_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail="任务不存在")

        if tasks[task_id]["status"] != "processing":
            return {"success": False, "message": f"任务状态为 {tasks[task_id]['status']}，无法取消"}

        tasks[task_id]["status"] = "cancelled"
        tasks[task_id]["message"] = "任务已取消"
        cancel_flags[task_id] = True

        # 复制 pid 以便在锁外终止
        pid = task_threads[task_id].pid if task_id in task_threads else None

    # 以下操作在锁外执行，避免在持锁时阻塞
    if pid is not None:
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                               capture_output=True, timeout=3)
            else:
                import signal
                os.kill(pid, signal.SIGTERM)
            print(f"已终止任务进程 PID: {pid}")
        except Exception as e:
            print(f"终止进程失败: {e}")

    with state_lock:
        task_threads.pop(task_id, None)

    # 删除输出文件
    output_path = OUTPUT_DIR / f"{task_id}_output.mp4"
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception as e:
            print(f"删除输出文件失败: {e}")

    return {"success": True, "message": "✅ 任务已取消，处理进程已立即终止"}


# ==================== 视频处理函数 ====================

def monitor_stdout_progress(task_id: str, process: subprocess.Popen, total_frames: int):
    """
    实时读取subprocess的stdout，解析进度。
    所有对 tasks dict 的访问都通过 state_lock 保护。
    """
    import re
    import time

    start_time = time.time()

    try:
        for line in process.stdout:
            # 输出到FastAPI的终端
            print(line, end='')

            # 尝试匹配所有模式的进度输出格式
            # 优先级：Frame N/M: (最明确) > Frame N: > Progress N%: (X/Y) > [N/M] (km/h)

            # Mode 2 格式：Frame 30/900: ...
            m = re.search(r'Frame (\d+)/(\d+):', line)
            if m:
                current_frame = int(m.group(1))
                total_in_output = int(m.group(2))
                progress = round((current_frame / total_in_output) * 100, 1)
            else:
                # Mode 1 格式：Frame 30: ...
                m = re.search(r'Frame (\d+):', line)
                if m and total_frames > 0:
                    current_frame = int(m.group(1))
                    progress = round((current_frame / total_frames) * 100, 1)
                else:
                    # Mode 3/4/5 格式：Progress 50.0% (450/900)  或  Progress 50.0%
                    m = re.search(r'Progress:\s*([\d.]+)%', line)
                    if m:
                        progress = round(float(m.group(1)), 1)
                        if total_frames > 0:
                            # 有 (X/Y) 时可精确验证
                            m2 = re.search(r'\((\d+)/(\d+)\)', line)
                            if m2 and int(m2.group(2)) == total_frames:
                                # 精确值优先
                                current_frame = int(m2.group(1))
                                progress = round((current_frame / total_frames) * 100, 1)
                    else:
                        # Mode 6 格式：[450/900]  ...km/h
                        m = re.search(r'\[\s*(\d+)\s*/\s*(\d+)\s*\]', line)
                        if m and total_frames > 0:
                            current_frame = int(m.group(1))
                            progress = round((current_frame / total_frames) * 100, 1)

            # 只有匹配到进度才更新（未匹配时 progress 未定义，保持原值）
            if 'progress' in dir():
                elapsed = time.time() - start_time
                fps = current_frame / elapsed if elapsed > 0 else 0
                msg = f"处理中... ({fps:.1f} 帧/秒)"
                with state_lock:
                    if task_id in tasks:
                        tasks[task_id]["progress"] = progress
                        tasks[task_id]["message"] = msg

            # 检查是否被取消
            if cancel_flags.get(task_id, False):
                break

    except Exception as e:
        print(f"监控stdout出错: {e}")
    finally:
        with state_lock:
            if task_id in tasks and tasks[task_id].get("status") == "completed":
                tasks[task_id]["progress"] = 100.0


def monitor_process_completion(task_id: str, process: subprocess.Popen):
    """
    监控子进程完成状态。
    所有对 tasks dict 的访问都通过 state_lock 保护。
    """
    try:
        return_code = process.wait()
        output_path = OUTPUT_DIR / f"{task_id}_output.mp4"

        with state_lock:
            if task_id not in tasks:
                return
            if return_code == 0 and output_path.exists():
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["progress"] = 100.0
                tasks[task_id]["message"] = "处理完成"
                tasks[task_id]["output_path"] = str(output_path)
            elif return_code == 2:
                tasks[task_id]["status"] = "cancelled"
                tasks[task_id]["message"] = "任务已取消"
            else:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["progress"] = 0.0
                tasks[task_id]["message"] = f"处理失败（退出码: {return_code}）"
    except Exception as e:
        with state_lock:
            if task_id in tasks:
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["message"] = f"监控失败: {str(e)}"
    finally:
        task_threads.pop(task_id, None)


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
