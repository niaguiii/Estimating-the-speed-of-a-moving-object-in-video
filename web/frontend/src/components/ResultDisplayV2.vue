<template>
  <div class="section">
    <div class="section-title">第 4 步：处理完成</div>

    <div class="result-area">
      <div class="alert alert-success">
        视频处理已完成，可以预览或下载结果。
      </div>

      <div class="video-container">
        <video :src="resultVideoUrl" controls style="width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"></video>
      </div>

      <div v-if="props.enhancedVideoId" class="enhanced-download-card">
        <div class="edc-info">
          <span class="edc-icon">增强</span>
          <div>
            <div class="edc-title">增强版视频</div>
            <div class="edc-desc">这是本次估计实际使用的预处理后视频。</div>
          </div>
        </div>
        <a :href="enhancedVideoDownloadUrl" download class="zip-download-btn">
          下载增强视频
        </a>
      </div>

      <div class="data-section">
        <div class="data-section-title">数据打包下载</div>
        <div class="zip-card">
          <div class="zip-info">
            <span class="zip-icon">ZIP</span>
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

      <div v-if="cropFiles.length > 0" class="crop-section">
        <div class="section-sub-title">{{ cropSectionTitle }}</div>
        <p class="crop-hint">{{ cropSectionHint }}</p>
        <div class="crop-grid">
          <div
            v-for="crop in paginatedCrops"
            :key="crop.name"
            class="crop-card"
          >
            <img :src="crop.url" :alt="crop.name" class="crop-thumb" loading="lazy" />
            <div class="crop-label">{{ getCropLabel(crop.name) }}</div>
          </div>
        </div>

        <div v-if="totalPages > 1" class="pagination">
          <button :disabled="currentPage === 1" @click="currentPage = 1">首页</button>
          <button :disabled="currentPage === 1" @click="currentPage--">上一页</button>
          <span class="page-info">{{ currentPage }} / {{ totalPages }}</span>
          <button :disabled="currentPage === totalPages" @click="currentPage++">下一页</button>
          <button :disabled="currentPage === totalPages" @click="currentPage = totalPages">末页</button>
        </div>
      </div>

      <div style="margin-top: 24px; text-align: center;">
        <a :href="downloadUrl" download style="text-decoration: none;">
          <button style="margin-right: 10px;">
            下载处理后的视频
          </button>
        </a>
        <button @click="$emit('reset')" style="background: #28a745;">
          处理另一个视频
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { getDataZipUrl, getDownloadUrl, getEnhancedVideoUrl } from '../api'

const props = defineProps(['taskId', 'mode', 'enhancedVideoId'])
defineEmits(['reset'])

const csvFiles = ref([])
const cropFiles = ref([])
const zipUrl = ref('')
const currentPage = ref(1)
const PAGE_SIZE = 24

const totalPages = computed(() => Math.max(1, Math.ceil(cropFiles.value.length / PAGE_SIZE)))

const paginatedCrops = computed(() => {
  const start = (currentPage.value - 1) * PAGE_SIZE
  return cropFiles.value.slice(start, start + PAGE_SIZE)
})

const resultVideoUrl = computed(() => getDownloadUrl(props.taskId))
const downloadUrl = computed(() => getDownloadUrl(props.taskId))
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
  if (props.mode === 6) {
    return `${csvFiles.value.length} 个 CSV + ${cropFiles.value.length} 张诊断图`
  }
  return `${csvFiles.value.length} 个 CSV`
})

const zipHint = computed(() => {
  if (props.mode === 5) {
    return 'frames.csv：逐帧目标速度、深度与相机补偿 | objects.csv：按目标汇总统计 | crops/：目标截图'
  }
  if (props.mode === 6) {
    return 'frames.csv：逐帧速度、有效像素占比、质量分级 | stats.csv：按秒汇总速度与位移 | diagnostics/：每20帧导出的有效区域与光流诊断图'
  }
  return 'ZIP 中包含本次处理生成的 CSV 数据文件'
})

const cropSectionTitle = computed(() => {
  if (props.mode === 6) {
    return `诊断截图（共 ${cropFiles.value.length} 张，分页展示）`
  }
  return `检测结果截图（共 ${cropFiles.value.length} 张，分页展示）`
})

const cropSectionHint = computed(() => {
  if (props.mode === 6) {
    return '这些图片按每20帧导出一组，展示真正参与速度估计的有效区域、二值掩码和光流可视化，已包含在上方 ZIP 压缩包中。'
  }
  return '以下截图已包含在上方 ZIP 压缩包中，可点击查看大图。'
})

const getCropLabel = (name) => {
  if (props.mode === 6) {
    const frameMatch = name.match(/^frame_(\d+)_(.+)\.(png|jpe?g)$/i)
    if (frameMatch) {
      const suffixLabels = {
        'valid_mask_overlay': '有效区域叠加图',
        'valid_mask_binary': '有效区域二值图',
        'flow_visualization': '光流可视化'
      }
      const frameNo = parseInt(frameMatch[1], 10)
      const suffix = frameMatch[2]
      return `Frame ${frameNo} - ${suffixLabels[suffix] || suffix}`
    }
    const labels = {
      'diagnostic_valid_mask_overlay.png': '有效区域叠加图',
      'diagnostic_valid_mask_binary.png': '有效区域二值图',
      'diagnostic_flow_visualization.png': '光流可视化'
    }
    return labels[name] || name.replace(/\.(png|jpe?g)$/i, '')
  }

  const match = name.match(/track_(\d+)_([^.]+)\.jpe?g/i)
  if (match) {
    return `ID${match[1]} - ${match[2]}`
  }
  return name.replace(/\.(png|jpe?g)$/i, '')
}

onMounted(async () => {
  try {
    const data = await fetch(`/api/task/${props.taskId}`).then((r) => r.json())
    csvFiles.value = data.csv_files || []
    cropFiles.value = data.crop_files || []
    zipUrl.value = data.zip_url || getDataZipUrl(props.taskId)
  } catch (error) {
    console.warn('获取结果元数据失败:', error)
    zipUrl.value = getDataZipUrl(props.taskId)
  }
})

watch(() => props.taskId, () => {
  currentPage.value = 1
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

.pagination {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.pagination button {
    padding: 4px 10px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 14px;
}

.pagination button:hover:not(:disabled) {
    background: #007bff;
    color: white;
    border-color: #007bff;
}

.pagination button:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.page-info {
    padding: 4px 12px;
    font-weight: bold;
    color: #333;
}
</style>
