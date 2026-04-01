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
        <p>测量物体真实移动速度（排除摄像头运动干扰）</p>
        <span class="badge hot">最高精度</span>
      </div>

      <div
        class="mode-card"
        :class="{ selected: selectedMode === 6 }"
        @click="selectMode(6)"
      >
        <h3>模式 6: 自车测速</h3>
        <p>测量摄像头/设备自身的移动速度</p>
        <span class="badge recommended">手持/行车记录仪</span>
      </div>
    </div>

    <!-- 焦段输入（Mode 5 / 6 专用） -->
    <div v-if="selectedMode === 5 || selectedMode === 6" class="focal-input-section">
      <label class="focal-label">
        📷 等效全画幅焦段（mm）：
        <span class="focal-hint">
          {{ selectedMode === 5 ? '普通镜头约50mm，监控/行车记录仪约14-24mm' : '广角行走/手持约24mm，标准约35-50mm，望远约85mm' }}
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
          🎯 处理精度：
          <span class="focal-hint">决定深度计算的频率，影响速度与精度</span>
        </label>
        <div class="precision-options">
          <label
            v-for="opt in precisionOptions"
            :key="opt.value"
            class="precision-option"
            :class="{ active: precisionLevel === opt.value }"
          >
            <input
              type="radio"
              :value="opt.value"
              v-model="precisionLevel"
              class="precision-radio"
            />
            <span class="precision-name">{{ opt.label }}</span>
            <span class="precision-desc">{{ opt.desc }}</span>
          </label>
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
const precisionLevel = ref('balanced')
const processing = ref(false)
const errorMessage = ref('')

const precisionOptions = [
  {
    value: 'high',
    label: '高精度（慢）',
    desc: '每帧重新计算深度，最准确，适合短视频',
    depthFrequency: 1,
  },
  {
    value: 'balanced',
    label: '平衡（推荐）',
    desc: '每5帧计算一次，速度和精度兼顾',
    depthFrequency: 5,
  },
  {
    value: 'fast',
    label: '快速（快）',
    desc: '每20帧计算一次，适合长视频预览',
    depthFrequency: 20,
  },
]

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
    const selectedPreset = precisionOptions.find(o => o.value === precisionLevel.value)
    const depthFreqArg = selectedPreset ? selectedPreset.depthFrequency : 5

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

.precision-options {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.precision-option {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border: 2px solid #ddd;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    background: white;
}

.precision-option:hover {
    border-color: #007bff;
    background: #f0f7ff;
}

.precision-option.active {
    border-color: #007bff;
    background: #e7f3ff;
}

.precision-radio {
    accent-color: #007bff;
    width: 16px;
    height: 16px;
}

.precision-name {
    font-weight: bold;
    font-size: 14px;
    color: #333;
    min-width: 120px;
}

.precision-desc {
    font-size: 12px;
    color: #666;
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
