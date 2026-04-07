# Web应用 - 视频速度估算系统

**[本文档 web/README.md]** — 前端/后端部署、API文档、开发指南

**[项目概述 README.md]** — 项目概述、快速开始、功能演示、模式选择指南

**[项目结构 docs/PROJECT_STRUCTURE.md]** — 详细目录结构、模块关系、配置说明

**[技术报告 docs/TECHNICAL_REPORT.md]** — 六种模式 + 预处理的算法原理、公式推导、论文出处

## 🚀 快速启动

### 1. 首次安装依赖
```bash
cd frontend
npm install
```

### 2. 启动后端
```bash
cd backend
python app.py
```

### 3. 启动前端（新终端）
```bash
cd frontend
npm run dev
```

浏览器自动打开：**http://localhost:3000**

---

## 📁 文件结构

```
web/
├── README.md               # 本文件
│
├── backend/                # Python FastAPI 后端（端口8000）
│   ├── app.py             # 服务器主程序
│   ├── process_worker.py  # 处理worker进程
│   └── requirements.txt   # Python依赖
│
└── frontend/               # Vue 3 + Vite 前端（端口3000）
    ├── package.json       # 前端依赖配置
    ├── vite.config.js     # Vite构建配置
    ├── index.html         # HTML入口
    ├── main.js            # 入口文件
    ├── App.vue            # 根组件
    ├── style.css          # 全局样式
    ├── .env.development   # 开发环境配置
    ├── .env.production    # 生产环境配置
    └── src/
        ├── api/
        │   └── index.js   # API封装
        └── components/
            ├── VideoUpload.vue      # 上传组件
            ├── ModeSelector.vue     # 模式选择组件
            ├── ProgressBar.vue     # 进度显示组件
            └── ResultDisplay.vue    # 结果显示组件
```

**数据存储：** 上传视频存储在项目根目录的 `data/web/uploads/`，处理结果存储在 `data/web/outputs/`（由后端自动创建）

**技术栈：** Vue 3 + Vite + Axios + FastAPI

---

## 🎯 功能特性

- **视频上传**：拖拽或点击上传
- **6种处理模式**：检测、追踪、速度估算、光流分析、深度感知、绝对深度
- **实时进度**：百分比、耗时、状态更新
- **连接监控**：实时后端连接状态指示
- **错误检测**：自动错误反馈和恢复
- **结果下载**：在线预览或下载处理后的视频

---

## 📖 使用步骤

### 步骤1：上传视频
- 拖拽视频文件或点击上传按钮
- 支持格式：MP4、AVI、MOV
- 建议：使用短视频测试（10-60秒）

### 步骤2：选择处理模式
- **模式1**：检测+追踪
- **模式2**：检测+追踪+速度
- **模式3**：RAFT光流（移动摄像头支持）
- **模式4**：RAFT+深度估算
- **模式5**：RAFT+绝对深度（最高精度）
- **模式6**：自车测速

### 步骤3：观察进度
- 实时进度条（0% → 100%）
- 处理时长计时器（分:秒）
- 状态信息显示
- 连接状态指示 🟢/🔴

### 步骤4：获取结果
- 在浏览器中预览视频
- 下载处理后的视频
- 处理下一个视频

---

## 🔧 技术细节

### 后端（端口8000）
- **语言**：Python
- **框架**：FastAPI
- **处理方式**：多线程后台任务
- **API接口**：
  - `POST /api/upload` - 上传视频
  - `POST /api/process` - 开始处理
  - `GET /api/task/{id}` - 查询进度
  - `GET /api/download/{id}` - 下载结果
  - `GET /api/history` - 查看历史
  - `GET /docs` - API文档

### 前端（端口3000）
- **语言**：JavaScript
- **框架**：Vue 3 + Vite
- **构建工具**：Vite 5.0
- **HTTP客户端**：Axios
- **特性**：
  - 组件化开发
  - 热模块替换（HMR）
  - 实时进度轮询（每2秒）
  - 自动错误检测
  - 连接健康监控

### 数据流程
```
用户浏览器 (localhost:3000)
    ↓ HTTP请求
前端Vue应用 (index.html)
    ↓ RESTful API
后端FastAPI (app.py)
    ↓ 调用处理模块
主项目 (../src/*.py)
    ↓ 输出结果
`data/web/outputs/{task_id}_output.mp4`
```

---

## 📦 首次安装

### 0. 环境要求
- **Node.js**: 18.0+ (https://nodejs.org/)
- **Python**: 3.8+
- **npm**: 9.0+ (随Node.js自动安装)

```bash
# 验证环境
node -v    # v18.0.0+
npm -v     # v9.0.0+
python -v  # 3.8+
```

### 1. 安装前端依赖
```bash
cd frontend
npm install
```

**前端依赖：**
- `vue` ^3.4.0 - 前端框架
- `axios` ^1.6.0 - HTTP客户端
- `vite` ^5.0.0 - 构建工具
- `@vitejs/plugin-vue` ^5.0.0 - Vue插件

### 2. 安装后端依赖
```bash
cd ../backend
pip install -r requirements.txt
```

**后端依赖：**
- `fastapi` >= 0.104.0 - Web框架
- `uvicorn` >= 0.24.0 - ASGI服务器
- `python-multipart` >= 0.0.6 - 文件上传支持
- `aiofiles` - 异步文件操作
- `pydantic` - 数据验证

### 3. 确保主项目依赖已安装
```bash
cd ../..
pip install -r requirements.txt
```

主项目需要：`ultralytics`、`opencv-python`、`numpy` 等

---

## 🌐 访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| **前端界面** | http://localhost:3000 | Web用户界面 |
| **后端API** | http://localhost:8000 | RESTful API服务器 |
| **API文档** | http://localhost:8000/docs | 交互式API文档 |

---

## ⚙️ 开发命令说明

### 前端命令
```bash
cd frontend

# 开发模式（热更新）
npm run dev          # 启动开发服务器 (localhost:3000)

# 生产构建
npm run build        # 构建到 dist/ 文件夹

# 预览构建结果
npm run preview      # 预览生产构建
```

### 后端命令
```bash
cd backend

# 启动后端服务
python app.py        # 启动FastAPI服务器 (localhost:8000)
```

### 部署流程
```bash
# 1. 构建前端
cd frontend
npm run build        # 生成 dist/ 文件夹

# 2. 上传 dist/ 到OSS/CDN
# 将 dist/ 文件夹上传到阿里云OSS或其他静态托管服务

# 3. 修改生产环境API地址
# 编辑 .env.production 文件
VITE_API_BASE=https://your-api-domain.com
```

---

## ❓ 常见问题

### 问：端口被占用？
**答：** 
```bash
# Windows查找并终止进程
netstat -ano | findstr :3000
taskkill /PID <进程ID> /F

# Linux/Mac查找并终止进程
lsof -i :3000
kill -9 <进程ID>
```

### 问：npm install失败？
**答：** 
- 确认Node.js版本 >= 18.0
- 尝试删除 node_modules 和 package-lock.json 后重新安装
- 使用国内镜像：`npm config set registry https://registry.npmmirror.com`

### 问：处理过程中后端断开？
**答：** 前端会显示：
- 连续失败3次（6秒后）显示警告
- 连续失败10次（20秒后）显示错误
- 提供"重试"按钮

### 问：如何停止服务？
**答：** 关闭启动时打开的两个命令行窗口

### 问：处理速度太慢？
**答：**
- 使用**模式2**（最快，CPU友好）
- 处理较短的视频（10-30秒）
- 长视频建议使用GPU加速

### 问：视频上传失败？
**答：** 检查：
- 视频格式（MP4、AVI、MOV）
- 视频文件未损坏
- `data/web/uploads/` 有足够的磁盘空间

### 问：处理失败？
**答：** 查看后端控制台错误信息：
- 依赖包缺失
- 视频编码不支持
- 内存不足

---

## 🐛 故障排查

### 后端无法启动
```bash
# 检查Python是否安装
python --version

# 检查依赖
cd backend
pip install -r requirements.txt

# 手动启动测试
python app.py
```

### 前端显示连接错误
```bash
# 确认后端正在运行
# 应该看到："Uvicorn running on http://0.0.0.0:8000"

# 直接测试API
curl http://localhost:8000/docs
```

### 处理卡在0%
- 查看后端控制台错误信息
- 验证主项目模块是否正常
- 尝试更简单的模式（模式1或2）

---

## 📊 性能参考

**模式2（CPU友好）：**
- 短视频（30秒）：约3-5分钟
- 中等视频（1分钟）：约6-10分钟
- 长视频（5分钟）：约30-60分钟

**注意：** 处理时间取决于：
- CPU性能
- 视频分辨率
- 视频中物体数量
- 选择的处理模式

---

## 🔄 工作流程示例

```
1. 用户上传 "test_video.mp4"（30秒）
   → 保存到：data/web/uploads/abc123.mp4

2. 用户选择模式2

3. 后端开始处理
   → 调用：src/mode2_speed_estimation.py
   → 每2秒更新进度

4. 处理完成（5分钟后）
   → 结果保存到：data/web/outputs/{task_id}_output.mp4

5. 前端显示结果
   → 用户可以预览或下载
```

---

## 📝 注意事项

- **首次运行**：可能会自动下载YOLOv8模型
- **存储空间**：上传和输出的视频会占用磁盘空间
- **清理文件**：手动删除 `uploads/` 和 `outputs/` 中的旧文件
- **安全性**：这是本地开发版本，不适合生产环境部署
- **未来扩展**：可以部署到云端（阿里云、AWS等）

---

## 🎓 开发者指南

### 后端架构
```python
# app.py 结构
FastAPI应用
├── CORS中间件
├── 上传接口 → 保存到 uploads/
├── 处理接口 → 启动后台线程
├── 任务状态 → 从tasks字典返回进度
├── 下载接口 → 流式传输输出文件
└── 历史接口 → 列出已完成任务
```

### 前端架构
```javascript
// index.html Vue 3应用
Vue.createApp({
  data: { currentStep, taskProgress, ... },
  methods: {
    uploadVideo()    // 步骤1
    selectMode()     // 步骤2
    startProcessing() // 步骤3
    startPolling()   // 监控进度
    showResult()     // 步骤4
  }
})
```

### 添加新的处理模式
1. 在 `src/` 中创建新的Python脚本
2. 实现 `process_video(input_path, output_path)` 函数
3. 在 `backend/app.py` → `process_video_task()` 中添加模式导入
4. 在 `frontend/index.html` 中更新模式选择界面

---

**版本：** 1.0  
**最后更新：** 2026-04-02  
**测试环境：** Windows 10/11, Python 3.8+
