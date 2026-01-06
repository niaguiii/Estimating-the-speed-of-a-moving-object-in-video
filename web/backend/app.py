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
import threading
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
    mode: int  # 1-4
    show_visualization: bool = True


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
    """
    try:
        video_id = request.video_id
        mode = request.mode
        
        # 查找上传的视频
        video_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="视频文件不存在")
        
        input_path = video_files[0]
        
        # 获取视频总帧数
        import cv2
        cap = cv2.VideoCapture(str(input_path))
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
            "input_path": str(input_path),
            "output_path": None,
            "created_at": datetime.now().isoformat()
        }
        
        # 使用subprocess启动独立进程（可以强制终止）
        python_path = sys.executable
        script_path = Path(__file__).parent / "process_worker.py"
        
        # 启动子进程（捕获stdout实时输出）
        process = subprocess.Popen(
            [python_path, str(script_path), task_id, str(input_path), str(mode), str(OUTPUT_DIR)],
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
            "message": "处理任务已启动"
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
    
    return tasks[task_id]


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
