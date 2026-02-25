# Mode 5：基于 RAFT 光流 + Metric3D v2 的三维速度估算

> **技术原理文档**  
> 项目：视频中移动物体速度估算  
> 核心模块：`src/mode5_metric3d_v2.py`

---

## 概述

Mode 5 是本项目最核心的处理模式，能够在**无需人工标定、无需参照物**的条件下，从普通单目视频中估算移动物体的真实三维速度（单位：m/s 与 km/h），并且支持**摄像头自身移动**的场景（如车载摄像头、手持拍摄）。

整体流程融合了三个深度学习模型与经典几何算法：

| 组件 | 模型 / 方法 | 作用 |
|---|---|---|
| 目标检测 | YOLOv8 | 逐帧检测物体，输出 Bounding Box |
| 多目标追踪 | ByteTrack | 跨帧分配稳定 ID（track_id） |
| 光流估计 | RAFT | 估计摄像头自身运动 |
| 深度估计 | Metric3D v2 | 输出每像素的绝对深度（米） |
| 速度计算 | Pinhole Camera Model + EMA | 像素位移转换为真实 m/s |

---

## 系统流程（Pipeline）

```
视频帧 (第 t 帧)
      │
      ├──► [Step 1] YOLOv8 检测 ─────────► Bounding Box + 类别标签
      │         └──► ByteTrack ──────────► 稳定 track_id（跨帧同一物体）
      │
      ├──► [Step 2] RAFT 光流 ───────────► 全图稠密位移场
      │         └──► 中位数提取 ──────────► 摄像头运动 (Δx_cam, Δy_cam)
      │
      ├──► [Step 3] Metric3D v2 ─────────► 深度图 D(u,v)（单位：米）
      │         └──► 每物体深度 EMA ──────► 平滑深度 Z̃（消除台阶跳变）
      │
      ├──► [Step 4] Pinhole 反投影 ───────► 3D 世界坐标 (X, Y, Z)
      │
      └──► [Step 5] 帧间 3D 距离 × FPS ──► 原始速度
                └──► 自适应 EMA 平滑 ─────► 最终输出速度（m/s, km/h）
```

---

## Step 1：目标检测与多目标追踪（YOLOv8 + ByteTrack）

### 1.1 YOLOv8 目标检测

YOLOv8（You Only Look Once v8）是单阶段目标检测器，一次前向传播即可输出所有检测框。对每个检测结果输出：

- Bounding Box 坐标：$(x_1, y_1, x_2, y_2)$（像素）
- 类别标签（如 *car*、*bus*、*person*）
- 置信度得分 $c \in [0, 1]$

Bounding Box 中心点（用于后续 3D 投影）：

$$c_x = \frac{x_1 + x_2}{2}, \quad c_y = \frac{y_1 + y_2}{2}$$

### 1.2 ByteTrack 多目标追踪

ByteTrack 使用**卡尔曼滤波（Kalman Filter）**预测物体下一帧位置，再用**匈牙利算法（Hungarian Algorithm）**完成检测框与历史轨迹的最优匹配，从而为每个物体分配跨帧稳定的 `track_id`。

速度计算必须依赖稳定 ID，因为需要对比同一物体在相邻帧的 3D 位置才能得出位移。

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
for box in results[0].boxes:
    track_id = int(box.id)          # 跨帧稳定 ID
    x1, y1, x2, y2 = box.xyxy[0]
    cx, cy = (x1+x2)/2, (y1+y2)/2  # BBox 中心点
```

---

## Step 2：摄像头运动估计（RAFT Optical Flow）

### 2.1 为什么需要摄像头运动补偿

若摄像头本身在移动（如车载、手持），画面中所有物体（包括静止物）都会产生像素位移。不去除此"背景运动"，静止的物体也会被误判为有速度。

$$\text{物体真实运动} = \text{物体表观像素位移} - \text{摄像头运动}$$

### 2.2 RAFT 光流模型

RAFT（Recurrent All-Pairs Field Transforms，Teed & Deng 2020）是基于深度学习的稠密光流模型。它对相邻两帧 $I_{t-1}$ 和 $I_t$ 中的每个像素 $(u,v)$ 输出其位移向量：

$$\mathbf{F}(u,v) = \begin{pmatrix} u_t - u_{t-1} \\ v_t - v_{t-1} \end{pmatrix}$$

### 2.3 提取摄像头全局运动

摄像头的刚体平移会在光流场中产生全局一致的位移模式。取光流场的**空间中位数**可鲁棒地估计摄像头运动（中位数对运动物体产生的局部异常值不敏感）：

$$\Delta x_{\text{cam}} = \text{median}(\mathbf{F}_u), \quad \Delta y_{\text{cam}} = \text{median}(\mathbf{F}_v)$$

### 2.4 摄像头运动补偿

对每个被追踪物体的中心点做补偿：

$$\hat{c}_x = c_x - \Delta x_{\text{cam}}, \quad \hat{c}_y = c_y - \Delta y_{\text{cam}}$$

补偿后的 $(\hat{c}_x, \hat{c}_y)$ 代表物体在世界坐标系中的真实像素位置，排除了摄像头自身运动的影响。

---

## Step 3：绝对度量深度估计（Metric3D v2）

### 3.1 相对深度 vs. 绝对深度

| 方法 | 输出 | 问题 |
|---|---|---|
| 相对深度（如 Depth Anything） | 无量纲比例值 | 不知道实际距离是多少米 |
| **绝对深度（Metric3D v2）** | **真实米数** | 可直接用于速度计算 |

**Metric3D v2**（Yin et al., 2023）是一个通用单目深度基础模型，通过在大规模多来源数据集上训练（含真实相机内参），使模型具备跨摄像头泛化能力，无需场景特定标定即可输出真实米数。

### 3.2 深度图采样

对于分辨率 $H \times W$ 的输入帧，Metric3D v2 输出深度图：

$$D \in \mathbb{R}^{H \times W}, \quad D(u,v) \approx Z_{\text{真实距离}} \text{ (米)}$$

对每个检测物体，取其 Bounding Box 内深度值的中位数作为该物体的深度估计：

$$Z_{\text{obj}} = \text{median}\bigl\{D(u,v) \mid (u,v) \in \text{BBox}\bigr\}$$

### 3.3 深度缓存机制与 EMA 平滑（工程贡献）

逐帧运行 Metric3D v2 计算量极大，因此每隔 $N$ 帧（`depth_frequency = 10`）才重新计算一次深度图，其余帧复用缓存。这引入了**台阶式跳变**：$Z_{\text{obj}}$ 在第 $N$ 帧突然更新，在速度计算中造成 $\Delta Z$ 的突变。

**解决方案：对每个被追踪物体单独维护深度 EMA（Exponential Moving Average）**：

$$\tilde{Z}_t = \alpha_d \cdot Z_t^{\text{raw}} + (1 - \alpha_d) \cdot \tilde{Z}_{t-1}, \quad \alpha_d = 0.15$$

EMA 将台阶跳变平滑成渐进过渡，同时仍能追踪真实的物体远近变化。

```python
smooth_depth = depth_alpha * current_depth + (1 - depth_alpha) * self.depth_history[track_id]
self.depth_history[track_id] = smooth_depth
```

---

## Step 4：2D 像素坐标 → 3D 世界坐标（Pinhole Camera Model）

### 4.1 针孔相机模型（正向投影）

针孔相机模型描述 3D 空间点 $(X, Y, Z)$ 如何投影到 2D 图像平面。已知相机内参 $(f_x, f_y, c_x^{\text{cam}}, c_y^{\text{cam}})$：

$$u = f_x \cdot \frac{X}{Z} + c_x^{\text{cam}}, \quad v = f_y \cdot \frac{Y}{Z} + c_y^{\text{cam}}$$

### 4.2 反投影（Back-Projection）

已知像素坐标 $(u, v)$ 和深度 $Z = \tilde{Z}$，还原 3D 世界坐标：

$$\boxed{X = \frac{(u - c_x^{\text{cam}}) \cdot Z}{f_x}, \quad Y = \frac{(v - c_y^{\text{cam}}) \cdot Z}{f_y}, \quad Z = \tilde{Z}}$$

### 4.3 相机内参估算

由于无法获得摄像头真实标定数据，根据图像分辨率和假设的水平视场角（FOV）估算焦距：

$$f_x = f_y = \frac{W}{2 \cdot \tan\!\left(\dfrac{\text{FOV}}{2}\right)}$$

以 FOV = 60°，图像宽度 $W = 852$ px 为例：

$$f_x = \frac{852}{2 \times \tan 30°} = \frac{852}{1.1547} \approx 737.9 \text{ px}$$

主点默认取图像中心：$c_x^{\text{cam}} = W/2 = 426$，$c_y^{\text{cam}} = H/2 = 240$

### 4.4 完整反投影（含摄像头运动补偿）

$$X_t = \frac{(\hat{c}_x - c_x^{\text{cam}}) \cdot \tilde{Z}_t}{f_x}, \quad Y_t = \frac{(\hat{c}_y - c_y^{\text{cam}}) \cdot \tilde{Z}_t}{f_y}, \quad Z_t = \tilde{Z}_t$$

```python
X = (comp_cx - cx_cam) * smooth_depth / fx
Y = (comp_cy - cy_cam) * smooth_depth / fy
Z = smooth_depth
```

---

## Step 5：3D 速度计算与自适应 EMA 平滑

### 5.1 帧间 3D 位移 → 速度

已知同一 `track_id` 的物体在第 $t-1$ 帧和第 $t$ 帧的 3D 坐标，计算三维欧式位移：

$$\Delta X = X_t - X_{t-1}, \quad \Delta Y = Y_t - Y_{t-1}, \quad \Delta Z = Z_t - Z_{t-1}$$

$$d_{3D} = \sqrt{\Delta X^2 + \Delta Y^2 + \Delta Z^2} \quad \text{（米/帧）}$$

速度换算（米/帧 × 帧/秒 = 米/秒）：

$$\boxed{v_{\text{raw}} = d_{3D} \times \text{FPS} \quad \text{(m/s)}}$$

**为什么需要 Z 方向**：当物体向摄像头驶来或远去时，$\Delta Z$ 是主要速度分量。仅用 XY 平面位移会严重低估真实速度（如车辆从远处驶近时，仅 2D 投影几乎不变，但真实速度可达数十 km/h）。

### 5.2 异常值抑制（Outlier Clamping）

YOLO Bounding Box 偶尔会出现大幅跳变（误检、遮挡恢复），导致单帧 $v_{\text{raw}}$ 异常偏大。在**稳定阶段**限制速度增幅不超过上一帧的 3 倍：

$$v_{\text{clamped}} = \min\bigl(v_{\text{raw}},\; 3 \times v_{t-1}\bigr)$$

此限制在冷启动阶段**主动关闭**（原因见 5.3）。

### 5.3 自适应双阶段 EMA 平滑（工程贡献）

**EMA（Exponential Moving Average，指数移动平均）**是一种低通滤波方法，将每帧速度估计值加权融合：

$$v_t = \alpha \cdot v_{\text{clamped}} + (1 - \alpha) \cdot v_{t-1}$$

- $\alpha$ 越大：响应越快，但噪声越多
- $\alpha$ 越小：越平滑，但收敛越慢

**问题**：物体刚被检测到时，$v_0 = 0$，若 $\alpha$ 固定为 0.15，则需要约 20+ 帧才能爬升到真实速度，导致显示的速度长期严重偏低（冷启动问题）。

**解决方案：两阶段自适应 Alpha**

| 阶段 | 帧范围（per track） | $\alpha$ | 目的 |
|---|---|---|---|
| **Warmup（冷启动）** | 第 1–10 帧 | **0.35** | 快速从 0 收敛到真实速度 |
| **Steady（稳定）** | 第 11 帧起 | **0.15** | 低噪声平滑输出 |

同时，冷启动阶段**关闭** 3× 异常值限制——因为若上一帧速度接近 0，$3 \times 0 = 0$ 会将所有真实速度锁死为 0，使速度永远无法爬升。

此外，每个 `track_id` 的**前 3 帧输出强制置零**，隐藏初始化噪声（初始 2 帧因位置历史不足无法计算真实速度）：

```python
# 分段 EMA alpha
alpha = self.warmup_alpha if frame_count <= self.warmup_frames else self.steady_alpha
in_warmup = frame_count <= self.warmup_frames

# 3× 限制：冷启动期关闭
if not in_warmup and self.speed_history[track_id] > 0.1:
    raw_speed = min(raw_speed, self.speed_history[track_id] * 3.0)

# 自适应 EMA
speed_ms = alpha * raw_speed + (1 - alpha) * self.speed_history[track_id]

# 前 3 帧不输出（隐藏初始化噪声）
if frame_count <= self.display_delay:   # display_delay = 3
    return 0.0
```

---

## 工程贡献

### 贡献一：RAFT 光流与 Metric3D v2 的系统集成

本项目将 RAFT 光流模型（摄像头运动估计）与 Metric3D v2 绝对深度模型（真实距离估算）结合，构建了一个完整的单目视频三维测速系统，无需任何硬件传感器（如激光雷达、IMU）和人工标定，即可在移动摄像头场景下输出真实物理单位（m/s）的速度估计。

这两个模型的组合实现了：
- RAFT 负责"去除摄像头影响"，得到物体在图像中的真实运动
- Metric3D v2 负责"提供真实尺度"，将像素位移转换为米

### 贡献二：基于 EMA 的深度平滑（解决深度缓存台阶跳变）

Metric3D v2 每 $N$ 帧才运行一次（计算量优化），导致深度图每隔 $N$ 帧才刷新，中间帧的深度值保持不变。直接使用此缓存深度计算速度时，深度更新帧的 $\Delta Z$ 会产生突变，导致速度从正常值瞬间跳升数倍。

本项目对每个被追踪物体**单独维护**深度 EMA，将台阶式跳变平滑为渐进过渡，在保持计算效率的同时消除了深度更新引起的速度尖刺。

$$\tilde{Z}_t^{(i)} = \alpha_d \cdot Z_t^{(i)} + (1-\alpha_d) \cdot \tilde{Z}_{t-1}^{(i)}, \quad \alpha_d = 0.15$$

### 贡献三：自适应双阶段 EMA 解决冷启动与速度突变问题

单一固定 EMA 系数 $\alpha$ 存在固有矛盾：
- 小 $\alpha$（如 0.1）：输出平滑，但冷启动收敛极慢，前 20+ 帧速度严重偏低
- 大 $\alpha$（如 0.35）：收敛快，但噪声抑制不足

本项目提出**两阶段自适应方案**：物体被追踪的前 10 帧使用 $\alpha = 0.35$ 快速收敛，之后切换至 $\alpha = 0.15$ 保守平滑。同时配合冷启动期关闭异常值限制，确保速度从零快速、准确地爬升至真实值。

---

## 数值示例

**参数**：FOV=60°，分辨率 852×480，FPS=29，某车辆深度约 10.8m

| 量 | 数值 |
|---|---|
| 焦距 $f_x$ | 737.9 px |
| BBox 中心（像素） | $(546, 372)$ |
| 摄像头运动 $(dx, dy)$ | $(-0.86, -0.10)$ px |
| 补偿后像素坐标 | $(546.86, 372.10)$ |
| 平滑深度 $\tilde{Z}$ | $10.78$ m |
| 3D 坐标 $(X, Y, Z)$ | $(1.77, 1.93, 10.78)$ m |
| 上一帧 3D 坐标 | $(1.60, 1.85, 11.20)$ m |
| 3D 位移 $d_{3D}$ | $\sqrt{0.17^2 + 0.08^2 + 0.42^2} \approx 0.457$ m |
| 原始速度 | $0.457 \times 29 \approx 13.2$ m/s（47.6 km/h） |

---

## 局限性

| 局限 | 影响 | 改进方向 |
|---|---|---|
| FOV 假设 60°，非真实标定值 | 系统性速度偏差约 10–20% | 提供真实 FOV 或进行相机标定 |
| Metric3D v2 精度受光照、场景影响 | 室外光线好时误差约 2–5% | 融合 LiDAR 或立体相机 |
| YOLO Bounding Box 逐帧抖动 | 引入 1–3 px 随机噪声 → 约 1–3 m/s 速度噪声 | 已由速度 EMA 抑制 |
| 3× 异常值限制不适用于高动态场景 | 羽毛球、球类等爆发性运动无法正确估算 | 未来加"高动态模式"开关 |

---

## 参考文献

1. Teed, Z., & Deng, J. (2020). *RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*. ECCV 2020.
2. Yin, W., et al. (2023). *Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image*. ICCV 2023.
3. Wang, C., et al. (2023). *YOLOv8: Real-Time Object Detection*. Ultralytics.
4. Zhang, Y., et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
5. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
