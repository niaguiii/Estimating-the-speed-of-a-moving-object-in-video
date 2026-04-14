# 新版 Mode 6 设计方案（与当前实现对齐）

> 更新日期：2026-04-14
> 当前实现文件：`src/mode6_ego_speed_v2.py`
> 核心方法：全图 RAFT 原生分辨率光流 + Metric3D 绝对深度 + 双帧一致性 YOLO 动态目标掩码 + 有效像素过滤
> 当前定位：面向手持、步行、行车记录仪等前后向主导场景的自运动速度估计；转弯、剧烈俯仰、强颠簸属于 Future Work

---

## 一、核心物理公式

### 1.1 透视投影（含主点偏移）

```
x = fx × X / Z + cx
y = fy × Y / Z + cy
```

其中 `cy` 是主点偏移量——光心在像素坐标系中的 y 坐标。对手机摄像头，`cy ≈ H/2`。

### 1.2 纯 Z 轴运动的光流推导

设相机固定（无旋转），只沿 Z 轴前进，前进速度 `vz`（m/s）。
对像素 `(x, y)` 对应的 3D 点 `(X, Y, Z)`：

```
y_t   = fy × Y / Z + cy
y_t+1 = fy × Y / (Z - vz / fps) + cy
```

两帧相减：

```
dy = y_t+1 - y_t
   = fy × Y × (1/(Z - vz/fps) - 1/Z)
   = fy × Y × (vz / fps) / (Z × (Z - vz/fps))
```

对小量 `vz/fps` 做一阶泰勒近似（`vz/fps << Z`）：

```
dy ≈ fy × Y × (vz / fps) / Z²
```

### 1.3 核心速度公式

将 `Y = (y - cy) × Z / fy` 回代：

```
dy ≈ fy × [(y - cy) × Z / fy] × (vz / fps) / Z²
   = (y - cy) / Z × vz / fps
```

整理得到**当前实现使用的一维有符号速度公式**：

```python
signed_axial_speed = dy * Z * fps / (y - cy)
```

| 参数 | 含义 | 来源 |
|---|---|---|
| `dy` | RAFT 垂直光流（像素/帧） | RAFT 光流场 |
| `Z` | 该像素的绝对深度（米） | Metric3D |
| `y` | 该像素的 y 坐标（像素） | 光流图坐标 |
| `cy` | 主点偏移（像素） | 从 FOV 推导（≈ H/2） |
| `fps` | 帧率 | 视频元数据 |

### 1.4 公式关键性质

```
dy = (y - cy) / Z × vz / fps
```

**dy 只由 (y, Z, vz) 决定，和 x 坐标完全无关。**

这意味着：
- 图像左侧和右侧的像素，dy **数值相同**（同深度/同 y 坐标时）
- 公式在图像**任意位置**都成立，不存在"必须在底部"的设计约束
- 靠近图像中线 `|y - cy| → 0` 时，`dy → 0`，这是**物理正确**的现象
- 为避免分母过小导致数值爆炸，应过滤 `|y - cy| < y_min` 的像素（建议 `y_min = 20~50` 像素）

---

## 二、数据流（完整流水线）

### 2.1 实际实现的总流程

```text
输入视频
  -> RAFT 全图光流（Mode 6 固定使用 native）
  -> Metric3D 深度（默认每 5 帧更新一次）
  -> YOLOv8 每帧检测可运动目标
  -> 双帧一致性动态掩码
  -> 有效像素过滤
  -> 逐像素速度
  -> 中位数聚合
  -> 双阶段 EMA 平滑
  -> 视频 + CSV + 诊断图
```

### 2.2 光流分辨率策略

当前实现与 `Mode 3-5` 不同：

- `Mode 3-5`：继续沿用 RAFT 的固定分辨率逻辑
- `Mode 6`：固定使用 `native` 原生分辨率逻辑

`native` 的含义不是“完全不处理尺寸”，而是：

1. 保持原始分辨率
2. 只做 8 的倍数 padding
3. RAFT 推理后再裁回原图大小

这样可以避免“先缩小、再放大”给逐像素速度公式带来额外量纲误差。

### 2.3 双帧一致性 YOLO 动态目标掩码

当前实现不是“单帧 YOLO 框直接掩掉”，而是更稳的一种时序一致性版本。

实现方式：

1. 每一帧都运行一次 YOLOv8
2. 仅保留可独立运动类别：
   - `person`
   - `bicycle`
   - `motorcycle`
   - `car`
   - `bus`
   - `truck`
   - `train`
   - `boat`
   - `airplane`
   - `dog`
   - `cat`
   - `horse`
   - `bird`
   - `cow`
   - `sheep`
3. 只有当前帧框与前一帧框满足 IoU 重叠时，才会对该目标做掩码
4. 掩码区域不是“交集框”，而是“前后两帧重叠目标的并集框”

因此，更准确的说法是：

`双帧连续检测 + 重叠后取并集框掩码`

### 2.4 四层有效像素过滤

当前实现最终有效像素由下面 4 个条件同时决定：

1. `YOLO mask`
   去掉双帧一致性判断出的动态目标区域
2. `observability mask`
   去掉低可观测像素：`flow_mag > 0.05`
3. `depth mask`
   仅保留 `1.0 < depth < 100.0`
4. `geometry mask`
   仅保留 `|y - cy| > 20`

代码形式为：

```python
valid_mask = yolo_mask & observability_mask & depth_mask & geometry_mask
```

### 2.5 逐像素速度与中位数聚合

对所有有效像素，计算：

```python
speed_i = dy_i * Z_i * fps / (y_i - cy)
```

然后：

- 至多均匀采样 `800` 个有效像素
- 逐像素得到 `speed_i`
- 取中位数作为当前帧原始速度 `raw_speed`

### 2.6 质量分级与转弯判断

当前实现的质量标记只有 4 类：

- `GOOD`
- `FAIR`
- `POOR`
- `TURN`

逻辑如下：

```text
如果 |dx_median| > 2.0      -> TURN
否则 valid_rate >= 0.20     -> GOOD
否则 valid_rate >= 0.08     -> FAIR
否则                         -> POOR
```

### 2.7 双阶段 EMA 平滑

当前实现使用：

- `warmup_frames = 30`
- `warmup_alpha = 0.5`
- `steady_alpha = 0.2`

并保留异常值钳制：

```python
if frame_count >= warmup_frames and abs(current_speed_ms) > 0.5:
    cap_val = max(abs(current_speed_ms) * 5.0, 40.0)
    raw_speed = clip(raw_speed, -cap_val, cap_val)
```

---

## 三、为什么需要两步掩码

单靠 YOLO 掩码不足以保证光流质量：

| 问题 | YOLO 能处理 | 还需要光流/深度/几何过滤 |
|---|---|---|
| 天空/白云（低纹理、低可信光流） | ❌ | ✅ |
| 光滑墙面/纯色区域 | ❌ | ✅ |
| 深度失效区域 | ❌ | ✅ |
| 主点附近分母过小 | ❌ | ✅ |

所以当前实现不是“只靠 YOLO”，而是：

`YOLO 动态掩码 + 低光流过滤 + 深度过滤 + 主点退化过滤`

---

## 四、有效像素率与质量分级

由于当前方法是全图静态背景法，经过多层过滤后，有效像素率通常不会很高，这是正常现象。

当前实现的质量阈值为：

| 质量等级 | 条件 | 显示效果 |
|---|---|---|
| `GOOD` | `valid_rate >= 20%` 且非转弯 | 绿色 |
| `FAIR` | `8% <= valid_rate < 20%` | 青黄色 |
| `POOR` | `valid_rate < 8%` | 橙色 |
| `TURN` | `|dx_median| > 2.0` | 黄色 + 转弯提示 |

---

## 五、与当前代码一致的伪代码

```python
def estimate_speed(flow, depth_map, cy, fps):
    H, W = flow.shape[:2]

    # Step 1: 双帧一致性 YOLO 掩码
    yolo_mask = build_temporal_yolo_mask(prev_detections, curr_detections)

    # Step 2: 低可观测过滤
    flow_mag = np.linalg.norm(flow, axis=2)
    observability_mask = flow_mag > 0.05

    # Step 3: 深度与几何过滤
    depth_mask = (depth_map > 1.0) & (depth_map < 100.0)
    y_rel = np.arange(H)[:, None] - cy
    geometry_mask = np.abs(y_rel) > 20

    valid = yolo_mask & observability_mask & depth_mask & geometry_mask
    flow_valid_rate = valid.sum() / (H * W)

    rows, cols = np.where(valid)
    if len(rows) > 800:
        sel = np.linspace(0, len(rows) - 1, 800, dtype=int)
        rows, cols = rows[sel], cols[sel]

    dy = flow[rows, cols, 1]
    dx = flow[rows, cols, 0]
    Z = depth_map[rows, cols]
    y_minus_cy = rows.astype(np.float32) - cy

    speeds = dy * Z * fps / y_minus_cy
    raw_speed = median(speeds)
    dx_median = median(dx)

    if abs(dx_median) > 2.0:
        quality = "TURN"
    elif flow_valid_rate >= 0.20:
        quality = "GOOD"
    elif flow_valid_rate >= 0.08:
        quality = "FAIR"
    else:
        quality = "POOR"

    return raw_speed, quality, flow_valid_rate
```

---

## 六、输出文件说明（与实现对齐）

### 6.0 命名规则

**CLI**

`main.py` 会先生成 `..._result_mode6.mp4` 基础名，`process_video_ego_speed()` 再在内部追加时间戳，因此最终视频名形如：

```text
data/cli/output/<输入名>_result_mode6_<timestamp>.mp4
```

同一 stem 下继续生成：

- `*_frames.csv`
- `*_stats.csv`
- `*_diagnostics/`

**Web**

Web worker 为了配合轮询、下载和历史记录，固定输出为：

```text
data/web/outputs/{task_id}_output.mp4
data/web/outputs/{task_id}_output_frames.csv
data/web/outputs/{task_id}_output_stats.csv
data/web/outputs/{task_id}_output_diagnostics/
```

用户实际下载时：

- 视频文件名：`mode6_result_{task_id}.mp4`
- ZIP 文件名：`mode6_data_{task_id}.zip`

### 6.1 帧级 CSV

输出路径：

```text
{output_stem}_frames.csv
```

当前真实列名：

- `frame_idx`
- `timestamp_s`
- `ego_speed_ms`
- `quality_flag`
- `flow_valid_rate`
- `valid_pixel_percent`
- `valid_pixels`
- `total_pixels`
- `dx_median`
- `raw_speed_ms`

### 6.2 秒级 CSV

输出路径：

```text
{output_stem}_stats.csv
```

当前真实列名：

- `second`
- `start_frame`
- `end_frame`
- `avg_speed_ms`
- `max_speed_ms`
- `min_speed_ms`
- `displacement_m`
- `cumulative_displacement_m`
- `dominant_quality`
- `avg_valid_pixel_percent`

### 6.3 视频 OSD

当前结果视频会显示：

- 当前有符号速度：`m/s`
- 质量标记：`GOOD / FAIR / POOR / TURN`
- 有效像素占比
- 最近历史速度曲线

### 6.4 诊断图目录

当前 Web/CLI 输出链路统一使用：

```text
{output_stem}_diagnostics/
```

当前实现默认：

- 每 `20` 帧导出一组
- 每组 `3` 张图

文件名示例：

- `frame_000020_valid_mask_overlay.png`
- `frame_000020_valid_mask_binary.png`
- `frame_000020_flow_visualization.png`

三张图分别表示：

1. 原图上叠加最终有效区域
2. 最终有效像素二值图
3. 当前采样帧的光流可视化

---

## 七、与旧版 Mode 6 的差异

| 维度 | 旧版 `mode6_ego_speed.py` | 当前版 `mode6_ego_speed_v2.py` |
|---|---|---|
| 采样策略 | 底部 ROI / 路面经验假设 | 全图静态背景法 |
| 光流分辨率 | 固定缩放逻辑 | 原生分辨率 `native` |
| 动态目标处理 | 主要靠中位数天然抗异常 | 双帧一致性 YOLO 显式掩码 |
| 过滤层 | 路面采样为主 | 动态掩码 + 低光流 + 深度 + 主点退化 |
| 速度公式 | 经验型实现 | `dy * Z * fps / (y - cy)` |
| 输出 | 单一速度读数为主 | 视频 + 双 CSV + 周期性诊断图 |

---

## 八、Future Work

当前版本已经实现并验证了主流程，但仍有边界条件：

1. 大角度转弯时，`dx_median` 会明显增大
   当前做法是标记 `TURN`，把结果视作参考值
2. 剧烈俯仰、上下坡、强颠簸会把额外姿态变化混入 `dy`
3. 低纹理、纯色、强反光区域虽然经过 `flow_mag > 0.05` 过滤，但仍可能有少量伪光流残留

因此，这套方案当前更准确的定位是：

- 已实现
- 可用于多场景前后向主导视频
- 但转弯/姿态剧烈变化仍属于 Future Work

---

## 九、与代码的对应关系

当前文档对应的是以下实现：

- 入口实现：`src/mode6_ego_speed_v2.py`
- 公共光流：`src/optical_flow_raft.py`
  - `Mode 6` 默认 `native`
  - `Mode 3-5` 继续使用固定分辨率逻辑
- CLI 入口：`main.py`
- Web worker 入口：`web/backend/process_worker.py`
- Web 结果展示：`web/frontend/src/components/ResultDisplayV2.vue`

这份文档现在描述的是**当前实现状态**，不再是“待确认后再编码”的设计稿。
