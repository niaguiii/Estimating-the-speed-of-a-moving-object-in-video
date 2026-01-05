# 项目结构文档

> 视频中移动物体速度估计 - 文件组织说明  
> **更新时间:** 2026-01-05 | **Phase 3 已完成 | Phase 4 Web已完成 ✅**

## 📁 项目目录结构

```
Estimating-the-speed-of-a-moving-object-in-video/
│
├── 📄 main.py                  # CLI模式主程序入口
├── 📄 README.md                # 项目说明文档
├── 📄 .gitignore              # Git忽略配置
├── 📄 .gitattributes          # Git属性配置
│
├── 📁 data/                    # 数据目录 ⭐ 统一管理
│   ├── 📁 cli/                # CLI模式数据
│   │   ├── input/            # 输入视频存放（测试用）
│   │   └── output/           # 处理结果输出
│   └── 📁 web/                # Web模式数据
│       ├── uploads/          # Web上传文件存储
│       └── outputs/          # Web处理结果存储
│
├── 📁 src/                     # 核心处理代码
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── main_opencv.py         # Phase 1 - OpenCV ONNX实现
│   ├── main_yolov8_native.py  # Phase 1 - YOLOv8原生实现
│   ├── main_yolov8_bytetrack.py  # Phase 2 - 检测+追踪
│   ├── main_yolov8_speed.py   # Phase 2 - 速度估算
│   ├── optical_flow_raft.py   # Phase 3 - RAFT光流模块
│   ├── depth_estimation.py    # Phase 3 - 深度估计模块
│   ├── main_yolov8_raft.py    # Phase 3 - RAFT光流版本
│   └── main_phase3_complete.py  # Phase 3 - 深度感知完整版
│
├── 📁 models/                  # AI模型文件
│   ├── yolov8n.pt             # YOLOv8权重文件
│   └── coco.names             # COCO 80类物体标签（如有）
│
├── 📁 trackers/                # 追踪器实现
│   ├── byte_tracker.py        # ByteTrack高精度追踪
│   └── simple_tracker.py      # 简单追踪器（备用）
│
├── 📁 scripts/                 # 安装和测试脚本
│   ├── README_SCRIPTS.md      # 脚本使用说明
│   ├── check_project.py       # 项目检查工具
│   ├── test_project.py        # 项目测试工具
│   │
│   ├── 📁 cpu/                # CPU版本（本地开发）
│   │   ├── README.md          # CPU安装说明
│   │   ├── requirements.txt   # CPU依赖列表
│   │   ├── install.bat        # Windows安装
│   │   ├── install.ps1        # PowerShell安装
│   │   └── setup_and_test.py  # 环境检查
│   │
│   └── 📁 gpu/                # GPU版本（云服务器部署）
│       ├── README.md          # GPU部署说明
│       ├── requirements.txt   # GPU依赖列表
│       ├── install_gpu.bat    # Windows完整安装
│       ├── install_gpu.sh     # Linux完整安装
│       ├── switch_to_gpu.bat  # Windows快速切换
│       └── switch_to_gpu.sh   # Linux快速切换
│
├── 📁 web/                     # Web应用 ⭐ Phase 4
│   ├── README.md              # Web使用指南
│   │
│   ├── 📁 backend/            # FastAPI后端（端口8000）
│   │   ├── app.py             # API服务器主程序
│   │   ├── process_worker.py  # 视频处理worker进程
│   │   └── requirements.txt   # 后端Python依赖
│   │
│   └── 📁 frontend/           # Vue 3前端（端口3000）
│       ├── package.json       # 前端npm依赖
│       ├── vite.config.js     # Vite构建配置
│       ├── index.html         # HTML入口
│       ├── .env.development   # 开发环境配置
│       ├── .env.production    # 生产环境配置
│       └── src/
│           ├── main.js        # 前端入口
│           ├── App.vue        # 根组件
│           ├── style.css      # 全局样式
│           ├── api/
│           │   └── index.js   # API封装
│           └── components/    # Vue组件
│               ├── VideoUpload.vue
│               ├── ModeSelector.vue
│               ├── ProgressBar.vue
│               └── ResultDisplay.vue
│
└── 📁 docs/                    # 项目文档
    ├── ARCHITECTURE.md        # 系统架构文档
    ├── PROJECT_STRUCTURE.md   # 项目结构（本文件）
    ├── FYP progress report 1st.docx  # 学校进度报告
    └── 项目计划书.docx        # 学校项目计划
```

---

## 🎯 文件组织原则

### 1. 根目录（最小化）✅
只保留核心文件：
- ✅ `main.py` - CLI模式程序入口
- ✅ `README.md` - 项目总说明
- ✅ Git配置文件

### 2. data/ - 数据统一管理 ⭐
```
data/
├── cli/          # CLI模式输入输出
└── web/          # Web模式上传和结果
```
**优势**: CLI和Web数据完全分离，清晰易管理

### 3. src/ - 核心处理代码 ✅
```
src/
├── Phase 1: main_opencv.py, main_yolov8_native.py
├── Phase 2: main_yolov8_bytetrack.py, main_yolov8_speed.py
├── Phase 3: optical_flow_raft.py, depth_estimation.py
├── Phase 3: main_yolov8_raft.py, main_phase3_complete.py
└── config.py（统一配置）
```

### 4. web/ - Web应用 ⭐ Phase 4
```
web/
├── backend/      # FastAPI RESTful API
│   ├── app.py
│   └── process_worker.py
└── frontend/     # Vue 3 + Vite
    └── src/      # 组件化开发
```

### 5. scripts/ - 安装和工具
```
scripts/
├── cpu/          # 本地开发（CPU版本）
└── gpu/          # 云服务器（GPU版本）
```
**详细说明**: 参见 `scripts/README_SCRIPTS.md`

### 6. docs/ - 项目文档
- `ARCHITECTURE.md` - 系统架构
- `PROJECT_STRUCTURE.md` - 本文件
- 学校文档（不修改）

### 7. models/ & trackers/
- `models/` - AI模型权重文件
- `trackers/` - 追踪算法实现

---

## 🚀 快速导航

### 📖 新手入门
1. **阅读总文档**: `README.md`
2. **查看项目结构**: `docs/PROJECT_STRUCTURE.md`（本文件）
3. **安装环境**: `scripts/cpu/README.md` 或 `scripts/gpu/README.md`

### 💻 CLI模式使用
```bash
# 1. 放视频到 data/cli/input/
# 2. 运行程序
python main.py

# 3. 查看结果 data/cli/output/
```

### 🌐 Web模式使用
```bash
# 后端
cd web/backend
python app.py

# 前端（新终端）
cd web/frontend
npm run dev
```

### 🔧 开发相关
- **项目检查**: `scripts/check_project.py`
- **项目测试**: `scripts/test_project.py`
- **架构文档**: `docs/ARCHITECTURE.md`

---

## 📊 统计信息

| 模块 | 文件数量 | 说明 |
|------|---------|------|
| 根目录 | 2个 | main.py + README.md |
| src/ | 10个 | 核心处理代码 |
| web/ | 20+ | Vue组件+API |
| scripts/ | 15+ | 安装脚本和工具 |
| docs/ | 4个 | 项目文档 |

**总计**: 约50+关键文件

---

## ✅ 优化效果

### 整理前
```
根目录混乱：
- requirements.txt (旧版本)
- requirements_phase3.txt
- requirements_gpu.txt
- check_gpu_status.py
- test_phase3_dependencies.py
- INSTALLATION_GUIDE.md
```

### 整理后
```
根目录清爽：
- main.py
- README.md
- Git配置文件

其他文件已归类到：
- scripts/cpu/requirements.txt
- scripts/gpu/requirements.txt
- scripts/check_gpu_status.py
- scripts/test_dependencies.py
- docs/INSTALLATION_GUIDE.md
```

**提升：** 根目录文件数量 从 9个 减少到 3个！

---

## 🎓 设计理念

### 1. 清晰的分层
- **根目录**: 最小化，只有核心入口
- **src/**: 程序代码
- **scripts/**: 工具脚本
- **docs/**: 文档

### 2. 明确的分类
- **CPU vs GPU**: 分别管理，避免混淆
- **代码 vs 文档**: 严格分离
- **工具 vs 核心**: 职责清晰

### 3. 易于维护
- 每个文件夹都有 README.md
- 相关文件放在一起
- 路径引用清晰

---

## 📚 相关文档

- **README.md** - 项目主文档
- **docs/ARCHITECTURE.md** - 系统架构
- **docs/INSTALLATION_GUIDE.md** - 安装指南
- **scripts/README_SCRIPTS.md** - 脚本说明

---

## 🎉 总结

**项目结构现在：**
- ✅ 根目录干净整洁
- ✅ 文件分类清晰
- ✅ CPU/GPU环境分离
- ✅ 文档集中管理
- ✅ 易于导航和维护

**适合：**
- 团队协作
- 版本控制
- 长期维护
- FYP答辩展示
