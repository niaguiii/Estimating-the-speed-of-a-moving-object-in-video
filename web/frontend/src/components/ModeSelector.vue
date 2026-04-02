<template>
  <div class="section">

    <!-- ============================================================
         阶段一：质量检测结果
         ============================================================ -->
    <div v-if="phase === 'quality'">

      <!-- 步骤标题 -->
      <div class="section-title">🔍 步骤 2: 视频质量检测</div>

      <!-- 检测中 -->
      <div v-if="qualityDetecting" class="quality-detecting">
        <div class="detecting-spinner"></div>
        <span>正在分析视频质量...</span>
        <span class="detecting-hint">采样帧分析中，请稍候</span>
      </div>

      <!-- 检测完成：全部正常 -->
      <div v-else-if="qualityDone && !hasIssues" class="quality-card quality-ok">
        <div class="quality-header">
          <span class="quality-icon">✅</span>
          <span class="quality-title">视频质量良好</span>
        </div>
        <div class="quality-items">
          <div class="quality-item">
            <span class="qi-label">清晰度</span>
            <span class="qi-badge qi-clear">清晰</span>
            <span class="qi-val">Laplacian: {{ qualityReport?.blur_index?.toFixed(0) }}</span>
          </div>
          <div class="quality-item">
            <span class="qi-label">雾气</span>
            <span class="qi-badge qi-clear">无雾</span>
            <span class="qi-val">暗通道: {{ qualityReport?.haze_index?.toFixed(1) }}</span>
          </div>
          <div class="quality-item">
            <span class="qi-label">亮度</span>
            <span class="qi-badge qi-clear">正常</span>
            <span class="qi-val">BI: {{ ((qualityReport?.brightness_index || 0) * 100).toFixed(0) }}%</span>
          </div>
        </div>

        <!-- 质量良好时：给出明确提示 + 两个入口 -->
        <div class="skip-actions">
          <div class="skip-hint">
            ✅ 视频质量良好，<strong>建议跳过预处理</strong>直接进入速度估算
          </div>
          <div class="skip-buttons">
            <button class="btn-primary" @click="skipToModeSelect">
              🚀 直接进入模式选择
            </button>
            <button class="btn-secondary" @click="goToManualEnhance">
              🎛️ 进入预处理（可选）
            </button>
          </div>
        </div>
      </div>

      <!-- 检测完成：有异常 -->
      <div v-else-if="qualityDone && hasIssues" class="quality-card quality-warn">
        <div class="quality-header">
          <span class="quality-icon">⚠️</span>
          <span class="quality-title">检测到以下问题</span>
        </div>

        <div class="quality-items">
          <!-- 模糊 -->
          <div v-if="qualityReport?.blur_level !== 'clear'" class="quality-issue">
            <div class="qi-row" @click="toggleEnhancement('blur')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('blur') }">
                {{ selectedEnhancements.includes('blur') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">📷</span>
              <span class="qi-issue-name">画面模糊</span>
              <span class="qi-badge" :class="qualityReport?.blur_level === 'blur' ? 'qi-blur' : 'qi-moderate'">
                {{ qualityReport?.blur_level === 'blur' ? '严重' : '中等' }}
              </span>
              <span class="qi-val">Laplacian: {{ qualityReport?.blur_index?.toFixed(0) }}</span>
            </div>
            <div class="qi-desc">{{ issueLabels.blur.desc }}</div>
          </div>

          <!-- 雾 -->
          <div v-if="qualityReport?.haze_level !== 'clear'" class="quality-issue">
            <div class="qi-row" @click="toggleEnhancement('haze')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('haze') }">
                {{ selectedEnhancements.includes('haze') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">🌫️</span>
              <span class="qi-issue-name">雾气较重</span>
              <span class="qi-badge" :class="qualityReport?.haze_level === 'foggy' ? 'qi-blur' : 'qi-moderate'">
                {{ qualityReport?.haze_level === 'foggy' ? '浓雾' : '轻度' }}
              </span>
              <span class="qi-val">暗通道: {{ qualityReport?.haze_index?.toFixed(1) }}</span>
            </div>
            <div class="qi-desc">{{ issueLabels.haze.desc }}</div>
          </div>

          <!-- 亮度 -->
          <div v-if="qualityReport?.brightness_level !== 'normal'" class="quality-issue">
            <div class="qi-row" @click="toggleEnhancement('brightness')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('brightness') }">
                {{ selectedEnhancements.includes('brightness') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">{{ qualityReport?.brightness_level === 'overexposed' ? '☀️' : '🌙' }}</span>
              <span class="qi-issue-name">{{ qualityReport?.brightness_level === 'overexposed' ? '画面过曝' : '亮度偏低' }}</span>
              <span class="qi-badge" :class="qualityReport?.brightness_level === 'overexposed' ? 'qi-overexposed' : 'qi-dark'">
                {{ qualityReport?.brightness_level === 'overexposed' ? '过曝' : '偏暗' }}
              </span>
              <span class="qi-val">BI: {{ ((qualityReport?.brightness_index || 0) * 100).toFixed(0) }}%</span>
            </div>
            <div class="qi-desc">
              {{ qualityReport?.brightness_level === 'overexposed'
                  ? issueLabels.brightness_overexposed.desc
                  : issueLabels.brightness.desc }}
            </div>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="enhance-actions">
          <div class="ea-selected-hint" v-if="selectedEnhancements.length > 0">
            已选 {{ selectedEnhancements.length }} 项预处理：
            {{ selectedEnhancements.map(e => getEnhancementLabel(e)).join(' → ') }}
          </div>
          <div class="ea-selected-hint" v-else>
            <span style="color: #92400e;">未选择任何预处理项</span>
          </div>
          <div class="ea-buttons">
            <button
              class="btn-secondary"
              :disabled="selectedEnhancements.length === 0"
              @click="startPreview"
            >
              👁️ 预览增强效果
            </button>
            <button class="btn-skip" @click="skipToModeSelect">
              ⏭️ 跳过预处理，直接进入
            </button>
          </div>
        </div>
      </div>

      <!-- 检测出错（降级，不阻止使用） -->
      <div v-else-if="qualityError" class="quality-card quality-error">
        <span>⚠️ 质量检测不可用，将跳过预处理</span>
        <div style="margin-top: 10px;">
          <button class="btn-primary" @click="skipToModeSelect">直接进入模式选择</button>
        </div>
      </div>

    </div><!-- /phase: quality -->


    <!-- ============================================================
         阶段二：预览增强效果（左右对比）
         ============================================================ -->
    <div v-if="phase === 'preview'">

      <div class="section-title">👁️ 步骤 2.5: 预处理效果预览</div>

      <!-- 预览处理中 -->
      <div v-if="previewing" class="preview-progress">
        <div class="preview-progress-title">
          正在处理增强视频...
        </div>
        <div class="preview-progress-bar">
          <div class="preview-progress-fill" :style="{ width: previewProgress + '%' }"></div>
        </div>
        <div class="preview-progress-msg">{{ previewMessage }}</div>
      </div>

      <!-- 未选择增强项时：引导用户勾选 -->
      <div v-else-if="phase === 'preview' && !previewing && !previewDone" class="quality-card quality-warn">
        <!-- 预处理失败提示 -->
        <div v-if="previewError" class="preview-error-box">
          <span>⚠️ 预处理失败：{{ previewError }}</span>
          <button class="btn-secondary" @click="startPreview" style="margin-left: 8px;">重试</button>
        </div>

        <!-- 增强项选择 -->
        <div class="quality-header">
          <span class="quality-icon">🎛️</span>
          <span class="quality-title">选择要应用的预处理</span>
        </div>
        <div class="quality-items">
          <div class="quality-issue" v-if="!qualityReport || qualityReport?.blur_level === 'clear'">
            <div class="qi-row" @click="toggleEnhancement('blur')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('blur') }">
                {{ selectedEnhancements.includes('blur') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">📷</span>
              <span class="qi-issue-name">去模糊</span>
              <span class="qi-badge qi-moderate">可选</span>
              <span class="qi-desc">{{ issueLabels.blur.desc }}</span>
            </div>
          </div>
          <div class="quality-issue" v-if="!qualityReport || qualityReport?.haze_level === 'clear'">
            <div class="qi-row" @click="toggleEnhancement('haze')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('haze') }">
                {{ selectedEnhancements.includes('haze') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">🌫️</span>
              <span class="qi-issue-name">去雾</span>
              <span class="qi-badge qi-moderate">可选</span>
              <span class="qi-desc">{{ issueLabels.haze.desc }}</span>
            </div>
          </div>
          <div class="quality-issue" v-if="!qualityReport || qualityReport?.brightness_level === 'normal'">
            <div class="qi-row" @click="toggleEnhancement('brightness')">
              <span class="qi-check" :class="{ active: selectedEnhancements.includes('brightness') }">
                {{ selectedEnhancements.includes('brightness') ? '☑️' : '☐' }}
              </span>
              <span class="qi-issue-icon">🌙</span>
              <span class="qi-issue-name">提亮</span>
              <span class="qi-badge qi-moderate">可选</span>
              <span class="qi-desc">{{ issueLabels.brightness.desc }}</span>
            </div>
          </div>
        </div>
        <div class="enhance-actions">
          <div class="ea-selected-hint" v-if="selectedEnhancements.length > 0">
            已选 {{ selectedEnhancements.length }} 项：{{ selectedEnhancements.map(e => getEnhancementLabel(e)).join(' → ') }}
          </div>
          <div class="ea-selected-hint" v-else>
            <span style="color: #92400e;">未选择任何预处理项</span>
          </div>
          <div class="ea-buttons">
            <button
              class="btn-primary"
              :disabled="selectedEnhancements.length === 0"
              @click="startPreview"
            >
              🔧 开始预处理
            </button>
            <button class="btn-skip" @click="skipToModeSelect">
              ⏭️ 跳过预处理
            </button>
          </div>
        </div>
      </div>

      <!-- 预览已完成 -->
      <div v-else-if="previewDone" class="preview-ready">

        <!-- 增强信息 -->
        <div class="preview-info">
          <span class="preview-badge">✨ 预处理完成</span>
          <span class="preview-methods">已应用：{{ appliedMethods.join(' → ') }}</span>
        </div>

        <!-- 左右对比播放器 -->
        <div class="compare-section">
          <div class="compare-hint">
            <span class="compare-arrow">←</span>
            左右拖动滑块对比增强前后效果
            <span class="compare-arrow">→</span>
          </div>
          <div class="compare-container">
            <!-- 原始视频（底层） -->
            <div class="compare-side compare-original">
              <div class="compare-label compare-label-left">📹 原始视频</div>
              <video :src="originalVideoUrl" controls class="compare-video"></video>
            </div>
            <!-- 增强视频（叠加层，带滑块遮罩） -->
            <div class="compare-side compare-enhanced" :style="{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }">
              <div class="compare-label compare-label-right">✨ 增强后</div>
              <video :src="enhancedVideoUrl" controls class="compare-video"></video>
            </div>
            <!-- 滑块 -->
            <div class="compare-slider" :style="{ left: sliderPosition + '%' }" @mousedown="startDrag" @touchstart="startDrag">
              <div class="slider-line"></div>
              <div class="slider-handle">
                <span>⟷</span>
              </div>
            </div>
          </div>
        </div>

        <!-- 操作按钮 -->
        <div class="preview-actions">
          <a :href="enhancedVideoUrl" download class="btn-download">
            📥 下载增强视频
          </a>
          <button class="btn-primary" @click="useEnhancedAndGo">
            ✅ 接受并进入模式选择
          </button>
          <button class="btn-skip" @click="skipToModeSelect">
            ⏭️ 跳过预处理，直接进入
          </button>
        </div>

        <div class="preview-note">
          💡 提示：跳过预处理将使用原始视频进行速度估算
        </div>
      </div>

    </div><!-- /phase: preview -->


    <!-- ============================================================
         阶段三：模式选择
         ============================================================ -->
    <div v-if="phase === 'mode_select'">

      <!-- 顶部提示：若使用增强视频则显示 -->
      <div v-if="usingEnhanced" class="using-enhanced-banner">
        <span>✨ 当前使用增强后的视频进行速度估算</span>
        <button class="btn-text" @click="undoEnhanced">撤销，换回原视频</button>
      </div>

      <div class="section-title">⚙️ 步骤 3: 选择处理模式</div>

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
        <button @click="backToQuality" style="background: #6c757d; margin-right: 10px;">
          ← 返回质量检测
        </button>
        <button @click="startProcessing" :disabled="!canStart">
          {{ processing ? '启动中...' : '开始处理' }}
        </button>
      </div>

      <div v-if="errorMessage" class="alert alert-error">
        {{ errorMessage }}
      </div>
    </div><!-- /phase: mode_select -->

  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { processVideo, detectQuality, enhanceVideo, getEnhancedVideoUrl, getOriginalVideoUrl } from '../api'

const emit = defineEmits(['select-mode', 'back'])
const props = defineProps(['videoId'])

const selectedMode = ref(null)
const focalMm = ref(50)
const precisionLevel = ref('balanced')
const processing = ref(false)
const errorMessage = ref('')

// ================================================================
// 三态阶段机
// ================================================================
// phase: 'quality' → 'preview' → 'mode_select'
const phase = ref('quality')

// 质量检测状态
const qualityDetecting = ref(false)
const qualityDone = ref(false)
const qualityReport = ref(null)
const qualityError = ref('')
const selectedEnhancements = ref([])

// 预览状态
const previewing = ref(false)
const previewDone = ref(false)
const previewError = ref('')
const previewProgress = ref(0)
const previewMessage = ref('')
const enhancedVideoReady = ref('')  // 增强视频的 video_id（如 "abc123_enhanced"）
const appliedMethods = ref([])

// 滑块对比状态
const sliderPosition = ref(50)
const isDragging = ref(false)

// 是否使用增强视频（用于跳过后回退）
const usingEnhanced = ref(false)

// 质量指标标签映射
const issueLabels = {
  blur: { icon: '📷', label: '画面模糊', desc: '检测到运动模糊或失焦，建议去模糊处理' },
  haze: { icon: '🌫️', label: '雾气较重', desc: '检测到雾/霾干扰，建议去雾处理' },
  brightness: { icon: '🌙', label: '亮度偏低', desc: '检测到低光照/暗部过多，建议提亮处理' },
  brightness_overexposed: { icon: '☀️', label: '画面过曝', desc: '检测到亮部过曝/高光溢出，建议降低亮度处理' },
}

const precisionOptions = [
  { value: 'high', label: '高精度（慢）', desc: '每帧重新计算深度，最准确，适合短视频', depthFrequency: 1 },
  { value: 'balanced', label: '平衡（推荐）', desc: '每5帧计算一次，速度和精度兼顾', depthFrequency: 5 },
  { value: 'fast', label: '快速（快）', desc: '每20帧计算一次，适合长视频预览', depthFrequency: 20 },
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

const hasIssues = computed(() => qualityReport.value?.needs_enhancement === true)

const enhancedVideoUrl = computed(() => {
  if (!enhancedVideoReady.value) return ''
  return getEnhancedVideoUrl(enhancedVideoReady.value)
})

const originalVideoUrl = computed(() => {
  if (!props.videoId) return ''
  return getOriginalVideoUrl(props.videoId)
})

// ================================================================
// 质量检测（自动触发）
// ================================================================
onMounted(async () => {
  window.addEventListener('mousemove', onDrag)
  window.addEventListener('mouseup', endDrag)
  window.addEventListener('touchmove', onDrag, { passive: false })
  window.addEventListener('touchend', endDrag)

  if (!props.videoId) return
  qualityDetecting.value = true
  qualityError.value = ''
  try {
    const resp = await detectQuality(props.videoId, false)
    qualityReport.value = resp.data.report
    selectedEnhancements.value = qualityReport.value.needs_enhancement
      ? [...qualityReport.value.issues]
      : []
    qualityDone.value = true
  } catch (e) {
    qualityError.value = '检测失败，将跳过预处理直接处理'
    qualityDone.value = true
  } finally {
    qualityDetecting.value = false
  }
})

onUnmounted(() => {
  window.removeEventListener('mousemove', onDrag)
  window.removeEventListener('mouseup', endDrag)
  window.removeEventListener('touchmove', onDrag)
  window.removeEventListener('touchend', endDrag)
})

// ================================================================
// 交互函数
// ================================================================

const toggleEnhancement = (issue) => {
  const idx = selectedEnhancements.value.indexOf(issue)
  if (idx >= 0) {
    selectedEnhancements.value.splice(idx, 1)
  } else {
    selectedEnhancements.value.push(issue)
  }
}

const getEnhancementLabel = (issue) => {
  const labels = {
    blur: '去模糊 (Wiener)',
    haze: '去雾 (DCP)',
    brightness: '提亮 (CLAHE+Gamma)',
  }
  return labels[issue] || issue
}

// ---------- 阶段切换 ----------

// 质量无问题时手动进入预处理
const goToManualEnhance = () => {
  // 质量无问题时，允许用户手动选择预处理项，进入预览阶段
  // selectedEnhancements 默认为空，用户可自由勾选
  phase.value = 'preview'
}

// 跳过预处理，直接进入模式选择
const skipToModeSelect = () => {
  usingEnhanced.value = false
  enhancedVideoReady.value = ''
  phase.value = 'mode_select'
}

// 返回质量检测页
const backToQuality = () => {
  phase.value = 'quality'
}

// ---------- 预览增强 ----------
const startPreview = async () => {
  if (selectedEnhancements.value.length === 0) {
    previewError.value = '未选择任何预处理项，请先勾选要处理的项'
    return
  }

  previewing.value = true
  previewDone.value = false
  previewError.value = ''
  previewProgress.value = 0
  previewMessage.value = '正在处理...'
  phase.value = 'preview'

  // 轮询增强进度（通过多次检测同一视频的长度间接感知）
  const pollProgress = setInterval(() => {
    if (previewProgress.value < 95) {
      previewProgress.value = Math.min(previewProgress.value + Math.random() * 15, 95)
    }
  }, 3000)

  try {
    const resp = await enhanceVideo(props.videoId, selectedEnhancements.value)
    clearInterval(pollProgress)
    previewProgress.value = 100
    previewMessage.value = '处理完成'
    appliedMethods.value = resp.data.applied_methods || selectedEnhancements.value
    // enhanced_video_id 格式为 "原id_enhanced"
    enhancedVideoReady.value = resp.data.enhanced_video_id || `${props.videoId}_enhanced`
    previewDone.value = true
    usingEnhanced.value = true
  } catch (e) {
    clearInterval(pollProgress)
    previewError.value = e?.response?.data?.detail || e.message || '预处理失败'
    previewDone.value = false
  } finally {
    previewing.value = false
  }
}

// ---------- 滑块对比（鼠标 + 触摸） ----------
const startDrag = (e) => {
  isDragging.value = true
  e.preventDefault()
}

const onDrag = (e) => {
  if (!isDragging.value) return
  const container = document.querySelector('.compare-container')
  if (!container) return
  const rect = container.getBoundingClientRect()
  const clientX = e.touches ? e.touches[0].clientX : e.clientX
  const x = clientX - rect.left
  sliderPosition.value = Math.max(5, Math.min(95, (x / rect.width) * 100))
}

const endDrag = () => {
  isDragging.value = false
}

// ---------- 接受增强，进入模式选择 ----------
const useEnhancedAndGo = () => {
  phase.value = 'mode_select'
}

// ---------- 撤销增强，换回原视频 ----------
const undoEnhanced = () => {
  usingEnhanced.value = false
  enhancedVideoReady.value = ''
}

// ---------- 模式选择 ----------
const selectMode = (mode) => {
  selectedMode.value = mode
  if (mode === 6) {
    focalMm.value = 24
  } else if (mode === 5) {
    focalMm.value = 50
  }
}

// ---------- 开始处理 ----------
const startProcessing = async () => {
  if (!selectedMode.value) return

  processing.value = true
  errorMessage.value = ''

  try {
    const focalMmArg = (selectedMode.value === 5 || selectedMode.value === 6) ? focalMm.value : null
    const selectedPreset = precisionOptions.find(o => o.value === precisionLevel.value)
    const depthFreqArg = selectedPreset ? selectedPreset.depthFrequency : 5

    // 若使用了增强视频，切换到增强版 video_id
    const videoIdToUse = usingEnhanced.value && enhancedVideoReady.value
      ? enhancedVideoReady.value
      : props.videoId

    const response = await processVideo(
      videoIdToUse,
      selectedMode.value,
      focalMmArg,
      depthFreqArg,
      false,   // 不再在 process 内部重复增强（已提前处理）
      null
    )

    if (response.data.success) {
      emit('select-mode', selectedMode.value, response.data.task_id, videoIdToUse)
    } else {
      errorMessage.value = '启动处理失败: ' + (response.data.message || response.data.detail)
    }
  } catch (error) {
    console.error('Process error:', error)
    errorMessage.value = '启动失败: 无法连接到服务器'
  } finally {
    processing.value = false
  }
}
</script>

<style scoped>
/* ================================================================
   通用按钮
   ================================================================ */
.btn-primary {
  padding: 8px 20px;
  background: #007bff;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
  transition: background 0.2s;
}
.btn-primary:hover { background: #0056b3; }
.btn-primary:disabled { background: #a0cfff; cursor: not-allowed; }

.btn-secondary {
  padding: 8px 20px;
  background: #6c757d;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.2s;
}
.btn-secondary:hover { background: #545b62; }

.btn-skip {
  padding: 8px 20px;
  background: white;
  color: #007bff;
  border: 2px solid #007bff;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: bold;
  transition: all 0.2s;
}
.btn-skip:hover { background: #e7f3ff; }

.btn-download {
  display: inline-block;
  padding: 8px 20px;
  background: #28a745;
  color: white;
  border-radius: 6px;
  text-decoration: none;
  font-size: 14px;
  font-weight: bold;
  transition: background 0.2s;
}
.btn-download:hover { background: #1e7e34; }

.btn-text {
  background: none;
  border: none;
  color: #007bff;
  cursor: pointer;
  font-size: 13px;
  text-decoration: underline;
  padding: 0;
  margin-left: 12px;
}
.btn-text:hover { color: #0056b3; }

/* ================================================================
   质量检测 UI
   ================================================================ */
.quality-detecting {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
  color: #555;
  font-size: 14px;
  margin-bottom: 12px;
}

.detecting-spinner {
  width: 20px;
  height: 20px;
  border: 3px solid #e9ecef;
  border-top-color: #007bff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

.detecting-hint { color: #888; font-size: 12px; }

.quality-card {
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 12px;
  border: 1px solid;
}
.quality-ok { background: #f0fdf4; border-color: #bbf7d0; }
.quality-warn { background: #fffbeb; border-color: #fde68a; }
.quality-error { background: #fef2f2; border-color: #fecaca; color: #991b1b; font-size: 13px; }

.quality-header { display: flex; align-items: center; gap: 8px; margin-bottom: 12px; }
.quality-icon { font-size: 20px; }
.quality-title { font-weight: bold; font-size: 15px; color: #333; }

.quality-items { display: flex; flex-direction: column; gap: 8px; }

.quality-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: white;
  border-radius: 6px;
}

.quality-issue {
  padding: 10px 12px;
  background: white;
  border-radius: 6px;
  border: 1px solid #fde68a;
  margin-bottom: 6px;
}

.qi-row {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}

.qi-check { font-size: 16px; cursor: pointer; }
.qi-issue-icon { font-size: 16px; }
.qi-issue-name { font-size: 14px; font-weight: 600; color: #333; }
.qi-desc { font-size: 12px; color: #777; margin-top: 5px; margin-left: 32px; }
.qi-label { font-size: 13px; color: #555; min-width: 48px; }

.qi-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: bold;
}
.qi-clear { background: #dcfce7; color: #15803d; }
.qi-moderate { background: #fef9c3; color: #a16207; }
.qi-blur { background: #fee2e2; color: #dc2626; }
.qi-dark { background: #1e293b; color: #e2e8f0; }
.qi-overexposed { background: #fff1f2; color: #e11d48; }
.qi-val { font-size: 11px; color: #888; margin-left: auto; }

/* 质量良好时的操作区 */
.skip-actions {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px dashed #bbf7d0;
}
.skip-hint {
  font-size: 13px;
  color: #16a34a;
  margin-bottom: 12px;
}
.skip-buttons { display: flex; gap: 10px; flex-wrap: wrap; }

/* 有问题时操作区 */
.enhance-actions {
  margin-top: 14px;
  padding-top: 12px;
  border-top: 1px dashed #fde68a;
}
.ea-selected-hint { font-size: 12px; color: #92400e; margin-bottom: 10px; }
.ea-buttons { display: flex; gap: 10px; flex-wrap: wrap; }

/* ================================================================
   预览阶段 UI
   ================================================================ */
.preview-progress {
  padding: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  border: 1px solid #e9ecef;
  text-align: center;
}
.preview-progress-title { font-weight: bold; color: #333; margin-bottom: 16px; }
.preview-progress-bar {
  width: 100%;
  height: 20px;
  background: #e9ecef;
  border-radius: 10px;
  overflow: hidden;
  margin-bottom: 10px;
}
.preview-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #007bff, #0056b3);
  border-radius: 10px;
  transition: width 0.5s ease;
}
.preview-progress-msg { font-size: 13px; color: #666; }

.preview-ready { display: flex; flex-direction: column; gap: 16px; }

.preview-info {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
  border-radius: 8px;
}
.preview-badge {
  font-size: 13px;
  font-weight: bold;
  color: #16a34a;
  background: #dcfce7;
  padding: 3px 10px;
  border-radius: 4px;
}
.preview-methods { font-size: 13px; color: #555; }

.video-player-card {
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-radius: 10px;
  padding: 14px;
}
.video-label {
  font-weight: bold;
  font-size: 14px;
  color: #333;
  margin-bottom: 10px;
}
.preview-video {
  width: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  display: block;
}

.preview-actions { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.preview-note { font-size: 12px; color: #888; margin-top: 4px; }

/* ================================================================
   增强视频横幅
   ================================================================ */
.using-enhanced-banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: #e0f2fe;
  border: 1px solid #7dd3fc;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 13px;
  color: #0369a1;
  font-weight: 500;
}

/* ================================================================
   模式选择 UI
   ================================================================ */
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
.mode-card:hover { border-color: #007bff; transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.mode-card.selected { border-color: #007bff; background: #e7f3ff; }
.mode-card h3 { color: #333; margin-bottom: 6px; font-size: 15px; }
.mode-card p { color: #666; font-size: 12px; margin-bottom: 6px; }

.badge {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: bold;
}
.badge.recommended { background: #28a745; color: white; }
.badge.fast { background: #17a2b8; color: white; }
.badge.slow { background: #ffc107; color: #333; }
.badge.hot { background: #e63946; color: white; }

/* 焦段输入区域 */
.focal-input-section {
  margin-top: 16px;
  padding: 14px;
  background: #f8f9fa;
  border-radius: 8px;
  border: 1px solid #e9ecef;
}
.focal-label { display: block; font-weight: bold; color: #333; margin-bottom: 8px; font-size: 14px; }
.focal-hint { font-weight: normal; color: #888; font-size: 12px; margin-left: 6px; }
.focal-inputs { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.focal-presets { display: flex; gap: 6px; flex-wrap: wrap; }

.preset-btn {
  padding: 4px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: white;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.2s;
}
.preset-btn:hover { border-color: #007bff; background: #e7f3ff; }
.preset-btn.active { border-color: #007bff; background: #007bff; color: white; }

.focal-number-input {
  width: 80px;
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}
.focal-unit { color: #666; font-size: 13px; }
.fov-display { color: #007bff; font-size: 13px; font-weight: bold; }

.depth-input-section { margin-top: 12px; }
.precision-options { display: flex; flex-direction: column; gap: 8px; }

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
.precision-option:hover { border-color: #007bff; background: #f0f7ff; }
.precision-option.active { border-color: #007bff; background: #e7f3ff; }
.precision-radio { accent-color: #007bff; width: 16px; height: 16px; }
.precision-name { font-weight: bold; font-size: 14px; color: #333; min-width: 120px; }
.precision-desc { font-size: 12px; color: #666; }

/* 通用提示 */
.alert {
  padding: 12px 20px;
  border-radius: 6px;
  margin: 15px 0;
}
.alert-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }

.preview-error-box {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 6px;
  margin-bottom: 14px;
  color: #991b1b;
  font-size: 13px;
}

/* ================================================================
   增强预览左右对比
   ================================================================ */
.compare-section { margin-bottom: 16px; }

.compare-hint {
  text-align: center;
  font-size: 12px;
  color: #888;
  margin-bottom: 10px;
}
.compare-arrow { color: #007bff; font-weight: bold; }

.compare-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  background: #000;
  user-select: none;
}

.compare-side {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.compare-original {
  z-index: 1;
}

.compare-enhanced {
  z-index: 2;
}

.compare-label {
  position: absolute;
  top: 10px;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  z-index: 3;
  pointer-events: none;
}
.compare-label-left {
  left: 10px;
  background: rgba(0,0,0,0.55);
  color: white;
}
.compare-label-right {
  right: 10px;
  background: rgba(0,100,0,0.6);
  color: white;
}

.compare-video {
  width: 100%;
  display: block;
}

.compare-slider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 4px;
  background: white;
  z-index: 10;
  cursor: ew-resize;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.slider-line {
  width: 2px;
  height: 100%;
  background: white;
  box-shadow: 0 0 6px rgba(0,0,0,0.4);
}

.slider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 32px;
  height: 32px;
  background: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  font-size: 16px;
  color: #007bff;
  pointer-events: none;
}
</style>
