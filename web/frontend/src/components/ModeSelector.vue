<template>
  <div class="section">
    <div class="section-title">⚙️ 步骤 2: 选择处理模式</div>
    
    <div class="mode-selector">
      <div 
        class="mode-card" 
        :class="{ selected: selectedMode === 1 }"
        @click="selectedMode = 1"
      >
        <h3>模式 1: 检测+追踪</h3>
        <p>YOLOv8检测 + ByteTrack追踪</p>
        <span class="badge fast">快速</span>
      </div>

      <div 
        class="mode-card" 
        :class="{ selected: selectedMode === 2 }"
        @click="selectedMode = 2"
      >
        <h3>模式 2: 检测+追踪+速度 ⭐</h3>
        <p>基于物体尺寸的速度估算</p>
        <span class="badge recommended">推荐CPU</span>
      </div>

      <div 
        class="mode-card" 
        :class="{ selected: selectedMode === 3 }"
        @click="selectedMode = 3"
      >
        <h3>模式 3: RAFT光流</h3>
        <p>支持移动摄像头场景</p>
        <span class="badge slow">较慢</span>
      </div>

      <div 
        class="mode-card" 
        :class="{ selected: selectedMode === 4 }"
        @click="selectedMode = 4"
      >
        <h3>模式 4: RAFT+深度</h3>
        <p>最高精度，深度感知</p>
        <span class="badge slow">最慢</span>
      </div>
    </div>

    <div style="margin-top: 20px;">
      <button @click="$emit('back')" style="background: #6c757d; margin-right: 10px;">
        ← 返回上传
      </button>
      <button @click="startProcessing" :disabled="!selectedMode || processing">
        {{ processing ? '启动中...' : '开始处理' }}
      </button>
    </div>

    <div v-if="errorMessage" class="alert alert-error">
      {{ errorMessage }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { processVideo } from '../api'

const emit = defineEmits(['select-mode', 'back'])
const props = defineProps(['videoId'])

const selectedMode = ref(null)
const processing = ref(false)
const errorMessage = ref('')

const startProcessing = async () => {
  if (!selectedMode.value) return

  processing.value = true
  errorMessage.value = ''

  try {
    const response = await processVideo(props.videoId, selectedMode.value)
    
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
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.mode-card {
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 15px;
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
    margin-bottom: 8px;
    font-size: 16px;
}

.mode-card p {
    color: #666;
    font-size: 13px;
    margin-bottom: 8px;
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
}

.badge.recommended {
    background: #28a745;
    color: white;
}

.badge.fast {
    background: #17a2b8;
    color: white;
}

.badge.slow {
    background: #ffc107;
    color: #333;
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
