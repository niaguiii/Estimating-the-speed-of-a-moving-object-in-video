<template>
  <div class="section">
    <div class="section-title">✅ 步骤 4: 处理完成</div>

    <div class="result-area">
      <div class="alert alert-success">
        🎉 视频处理完成！可以预览或下载结果。
      </div>

      <div class="video-container">
        <video :src="resultVideoUrl" controls style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"></video>
      </div>

      <!-- 增强视频下载（仅当使用了增强预处理时显示） -->
      <div v-if="props.enhancedVideoId" class="enhanced-download-card">
        <div class="edc-info">
          <span class="edc-icon">✨</span>
          <div>
            <div class="edc-title">增强版视频</div>
            <div class="edc-desc">预处理增强后的视频（已用于本次速度估算）</div>
          </div>
        </div>
        <a :href="enhancedVideoDownloadUrl" download class="zip-download-btn">
          📥 下载增强视频
        </a>
      </div>

      <!-- 数据打包下载（统一 ZIP，所有 Mode 一致） -->
      <div class="data-section">
        <div class="data-section-title">📦 数据打包下载</div>
        <div class="zip-card">
          <div class="zip-info">
            <span class="zip-icon">📦</span>
            <div>
              <div class="zip-name">{{ zipName }}</div>
              <div class="zip-desc">{{ dataDesc }}</div>
            </div>
          </div>
          <a :href="zipUrl" download class="zip-download-btn">
            下载 ZIP
          </a>
        </div>
        <p class="data-hint">{{ zipHint }}</p>
      </div>

      <!-- 检测物体截图（仅 Mode 5 有） -->
      <div v-if="cropFiles.length > 0" class="crop-section">
        <div class="section-sub-title">🚗 检测到的物体截图（共 {{ cropFiles.length }} 个）</div>
        <p class="crop-hint">以下截图已包含在上方的 ZIP 压缩包中，可点击查看大图</p>
        <div class="crop-grid">
          <div
            v-for="crop in cropFiles"
            :key="crop.name"
            class="crop-card"
          >
            <img :src="crop.url" :alt="crop.name" class="crop-thumb" />
            <div class="crop-label">{{ getCropLabel(crop.name) }}</div>
          </div>
        </div>
      </div>

      <div style="margin-top: 24px; text-align: center;">
        <a :href="downloadUrl" download style="text-decoration: none;">
          <button style="margin-right: 10px;">
            📥 下载处理后的视频
          </button>
        </a>
        <button @click="$emit('reset')" style="background: #28a745;">
          🔄 处理另一个视频
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { getDownloadUrl, getDataZipUrl, getEnhancedVideoUrl } from '../api'

const props = defineProps(['taskId', 'mode', 'enhancedVideoId'])
const emit = defineEmits(['reset'])

const csvFiles  = ref([])
const cropFiles = ref([])
const zipUrl    = ref('')

const resultVideoUrl = computed(() => getDownloadUrl(props.taskId))
const downloadUrl    = computed(() => getDownloadUrl(props.taskId))
const enhancedVideoDownloadUrl = computed(() =>
  props.enhancedVideoId ? getEnhancedVideoUrl(props.enhancedVideoId) : ''
)

const zipName = computed(() => {
  const prefix = props.mode === 5 ? 'mode5' : props.mode === 6 ? 'mode6' : `mode${props.mode}`
  return `${prefix}_data_${props.taskId}.zip`
})

const dataDesc = computed(() => {
  if (props.mode === 5) {
    return `${csvFiles.value.length} 个 CSV + ${cropFiles.value.length} 张截图`
  }
  return `${csvFiles.value.length} 个 CSV`
})

const zipHint = computed(() => {
  if (props.mode === 5) {
    return 'objects.csv：物体逐帧速度/深度  |  frames.csv：帧级汇总  |  crops/：物体首次出现截图'
  }
  if (props.mode === 6) {
    return 'stats.csv：按秒汇总的自车速度与位移统计'
  }
  return 'CSV：逐帧物体追踪与速度数据'
})

const getCropLabel = (name) => {
  const m = name.match(/track_(\d+)_([^.]+)\.jpe?g/i)
  if (m) return `ID${m[1]} · ${m[2]}`
  return name.replace(/\.jpe?g/i, '')
}

onMounted(async () => {
  try {
    const data = await fetch(`/api/task/${props.taskId}`).then(r => r.json())
    csvFiles.value  = data.csv_files  || []
    cropFiles.value = data.crop_files || []
    zipUrl.value    = data.zip_url    || getDataZipUrl(props.taskId)
  } catch (e) {
    console.warn('获取数据失败:', e)
    zipUrl.value = getDataZipUrl(props.taskId)
  }
})
</script>

<style scoped>
.result-area {
    margin-top: 20px;
}

.video-container {
    width: 100%;
    max-width: 800px;
    margin: 20px auto;
}

.alert {
    padding: 12px 20px;
    border-radius: 6px;
    margin: 15px 0;
}

.alert-success {
    background: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}

/* 数据打包区 */
.data-section {
    margin: 24px 0;
    padding: 16px;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
}

.data-section-title {
    font-weight: bold;
    font-size: 15px;
    color: #333;
    margin-bottom: 12px;
}

.zip-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    background: white;
    border: 2px solid #007bff;
    border-radius: 10px;
    gap: 16px;
}

.zip-info {
    display: flex;
    align-items: center;
    gap: 12px;
}

.zip-icon {
    font-size: 28px;
}

.zip-name {
    font-weight: bold;
    font-size: 14px;
    color: #333;
}

.zip-desc {
    font-size: 12px;
    color: #888;
    margin-top: 2px;
}

.zip-download-btn {
    padding: 8px 24px;
    background: #007bff;
    color: white;
    border-radius: 6px;
    font-weight: bold;
    font-size: 14px;
    text-decoration: none;
    white-space: nowrap;
    transition: background 0.2s;
}

.zip-download-btn:hover {
    background: #0056b3;
}

.data-hint {
    margin-top: 10px;
    font-size: 11px;
    color: #888;
}

/* 截图网格 */
.crop-section {
    margin: 24px 0;
    padding: 16px;
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 10px;
}

.section-sub-title {
    font-weight: bold;
    font-size: 15px;
    color: #333;
    margin-bottom: 6px;
}

.crop-hint {
    font-size: 11px;
    color: #888;
    margin-bottom: 12px;
}

.crop-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
}

.crop-card {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}

.crop-thumb {
    width: 100%;
    max-height: 120px;
    object-fit: contain;
    border-radius: 6px;
    border: 1px solid #eee;
}

.crop-label {
    font-size: 12px;
    font-weight: bold;
    color: #333;
    text-align: center;
}

/* 增强视频下载区 */
.enhanced-download-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    background: #eff6ff;
    border: 2px solid #93c5fd;
    border-radius: 10px;
    margin-bottom: 16px;
    gap: 16px;
}

.edc-info {
    display: flex;
    align-items: center;
    gap: 12px;
}

.edc-icon {
    font-size: 24px;
}

.edc-title {
    font-weight: bold;
    font-size: 14px;
    color: #1e40af;
}

.edc-desc {
    font-size: 12px;
    color: #6b7280;
    margin-top: 2px;
}
</style>
