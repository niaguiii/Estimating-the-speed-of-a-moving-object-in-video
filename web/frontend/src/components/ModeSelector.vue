<template>
  <div class="section">
    <div class="section-title">⚙️ 步骤 2: 选择处理模式</div>

    <div class="mode-selector">
      <div
        class="mode-card"
        :class="{ selected: selectedMode === 1 }"
        @click="selectMode(1)"
      >
        <h3>模式 1: 检测+追踪</h3>
        <p>YOLOv8检测 + ByteTrack追踪</p>
        <span class="badge fast">快速</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 2 }"
        @click="selectMode(2)"
      >
        <h3>模式 2: 检测+速度 ⭐</h3>
        <p>基于物体尺寸的速度估算</p>
        <span class="badge recommended">推荐CPU</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 3 }"
        @click="selectMode(3)"
      >
        <h3>模式 3: RAFT光流</h3>
        <p>支持移动摄像头场景</p>
        <span class="badge slow">较慢</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 4 }"
        @click="selectMode(4)"
      >
        <h3>模式 4: RAFT+深度</h3>
        <p>相对深度感知</p>
        <span class="badge slow">较慢</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 5 }"
        @click="selectMode(5)"
      >
        <h3>模式 5: Metric3D v2 🔥</h3>
        <p>绝对度量深度（真实米数）</p>
        <span class="badge hot">最高精度</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 6 }"
        @click="selectMode(6)"
      >
        <h3>模式 6: 自车测速 🚗</h3>
        <p>无需YOLO，路面光流测本车速度</p>
        <span class="badge recommended">行车记录仪</span>
      </div>
    </div>

    <!-- 焦段输入（Mode 5 / 6 专用） -->
    <div v-if="selectedMode === 5 || selectedMode === 6" class="focal-input-section">
      <label class="focal-label">
        📷 等效全画幅焦段（mm）：
        <span class="focal-hint">
          {{ selectedMode === 5 ? '普通镜头约50mm，行车记录仪约14-24mm' : '行车记录仪约14-24mm，手机广角约24mm' }}
        </span>
      </label>
      <div class="focal-inputs">
        <div class="focal-presets">
          <button
            v-for="preset in focalPresets"
            :key="preset.value"
            class="preset-btn"
            :class="{ active: focalMm === preset.value }"
            @click="focalMm = preset.value"
          >
            {{ preset.label }}
          </button>
        </div>
        <input
          v-model.number="focalMm"
          type="number"
          class="focal-number-input"
          placeholder="例如 50"
          min="1"
          max="300"
        />
        <span class="focal-unit">mm</span>
        <span v-if="computedFov" class="fov-display">≈ {{ computedFov }}° FOV</span>
      </div>

      <div v-if="selectedMode === 5" class="depth-input-section">
        <label class="focal-label">
          📊 深度更新频率：
          <span class="focal-hint">每N帧重算一次深度，越小越精确但越慢（推荐3-5）</span>
        </label>
        <div class="depth-inputs">
          <input
            v-model.number="depthFrequency"
            type="number"
            class="focal-number-input"
            min="1"
            max="30"
          />
          <span class="focal-unit">帧</span>
        </div>
      </div>
    </div>

    <div style="margin-top: 20px;">
      <button @click="$emit('back')" style="background: #6c757d; margin-right: 10px;">
        ← 返回上传
      </button>
      <button @click="startProcessing" :disabled="!canStart">
        {{ processing ? '启动中...' : '开始处理' }}
      </button>
    </div>

    <div v-if="errorMessage" class="alert alert-error">
      {{ errorMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { processVideo } from '../api'

const emit = defineEmits(['select-mode', 'back'])
const props = defineProps(['videoId'])

const selectedMode = ref(null)
const focalMm = ref(50)
const depthFrequency = ref(5)
const processing = ref(false)
const errorMessage = ref('')

const focalPresets = [
  { label: '14mm', value: 14 },
  { label: '24mm', value: 24 },
  { label: '35mm', value: 35 },
  { label: '50mm', value: 50 },
  { label: '85mm', value: 85 },
]

const computedFov = computed(() => {
  if (!focalMm.value || focalMm.value <= 0) return null
  const fov = 2 * Math.atan(18 / focalMm.value) * (180 / Math.PI)
  return fov.toFixed(1)
})

const canStart = computed(() => {
  if (!selectedMode.value || processing.value) return false
  if ((selectedMode.value === 5 || selectedMode.value === 6) && (!focalMm.value || focalMm.value <= 0)) {
    return false
  }
  return true
})

const selectMode = (mode) => {
  selectedMode.value = mode
  // 根据模式设置合理的默认值
  if (mode === 6) {
    focalMm.value = 24 // 行车记录仪默认广角
  } else if (mode === 5) {
    focalMm.value = 50 // 标准镜头默认
  }
}

const startProcessing = async () => {
  if (!selectedMode.value) return

  processing.value = true
  errorMessage.value = ''

  try {
    const focalMmArg = (selectedMode.value === 5 || selectedMode.value === 6) ? focalMm.value : null
    const depthFreqArg = selectedMode.value === 5 ? depthFrequency.value : null

    const response = await processVideo(props.videoId, selectedMode.value, focalMmArg, depthFreqArg)

    if (response.data.success) {
      emit('select-mode', selectedMode.value, response.data.task_id)
    } else {
      errorMessage.value = '❌ 启动处理失败: ' + response.data.message
    }
  } catch (error) {
    console.error('Process error:', error)
    errorMessage.value = '❌ 启动失败: 无法连接到服务器'
  } finally {
    processing.value = false
  }
}
</script>

<style scoped>
.mode-selector {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
    margin-top: 15px;
}

.mode-card {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 14px;
    cursor: pointer;
    transition: all 0.3s;
}

.mode-card:hover {
    border-color: #007bff;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.mode-card.selected {
    border-color: #007bff;
    background: #e7f3ff;
}

.mode-card h3 {
    color: #333;
    margin-bottom: 6px;
    font-size: 15px;
}

.mode-card p {
    color: #666;
    font-size: 12px;
    margin-bottom: 6px;
}

.badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: bold;
}

.badge.recommended { background: #28a745; color: white; }
.badge.fast         { background: #17a2b8; color: white; }
.badge.slow         { background: #ffc107; color: #333; }
.badge.hot          { background: #e63946; color: white; }

/* 焦段输入区域 */
.focal-input-section {
    margin-top: 16px;
    padding: 14px;
    background: #f8f9fa;
    border-radius: 8px;
    border: 1px solid #e9ecef;
}

.focal-label {
    display: block;
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
    font-size: 14px;
}

.focal-hint {
    font-weight: normal;
    color: #888;
    font-size: 12px;
    margin-left: 6px;
}

.focal-inputs, .depth-inputs {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.focal-presets {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}

.preset-btn {
    padding: 4px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.2s;
}

.preset-btn:hover {
    border-color: #007bff;
    background: #e7f3ff;
}

.preset-btn.active {
    border-color: #007bff;
    background: #007bff;
    color: white;
}

.focal-number-input {
    width: 80px;
    padding: 6px 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 14px;
}

.focal-unit {
    color: #666;
    font-size: 13px;
}

.fov-display {
    color: #007bff;
    font-size: 13px;
    font-weight: bold;
}

.depth-input-section {
    margin-top: 12px;
}

.alert {
    padding: 12px 20px;
    border-radius: 6px;
    margin: 15px 0;
}

.alert-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
