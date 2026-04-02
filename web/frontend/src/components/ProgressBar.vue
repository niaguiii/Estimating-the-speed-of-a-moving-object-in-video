<template>
  <div class="section">
    <div class="section-title">
      ⏳ 步骤 3: 正在处理
      <span class="connection-status" :class="{ error: !connectionAlive }"></span>
      <span style="font-size: 12px; color: #666;">{{ connectionAlive ? '后端连接正常' : '后端连接异常' }}</span>
    </div>
    
    <div class="progress-area">
      <div class="alert alert-info">
        <strong>模式{{ mode }}:</strong> 
        <span v-if="mode === 1">检测+追踪</span>
        <span v-if="mode === 2">检测+追踪+速度（CPU友好）</span>
        <span v-if="mode === 3">RAFT光流（移动摄像头）</span>
        <span v-if="mode === 4">RAFT+深度（最高精度）</span>
        | 已处理时长: {{ processingElapsedTime }}
      </div>

      <!-- 进度条 -->
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: taskProgress + '%' }"></div>
        <div class="progress-text">{{ taskProgress.toFixed(1) }}%</div>
      </div>

      <div class="progress-info">
        <span>{{ taskMessage }}</span>
        <span>耗时: {{ processingElapsedTime }}</span>
      </div>

      <!-- 错误或警告信息 -->
      <div v-if="connectionFailCount >= 3 && connectionFailCount < 10" class="alert alert-warning">
        ⚠️ 警告: 后端连接不稳定（失败{{ connectionFailCount }}次），正在重试...
      </div>

      <div v-if="connectionFailCount >= 10" class="alert alert-error">
        ❌ 错误: 后端连接失败，可能已终止或出现错误。请检查后端状态。
        <button @click="retryConnection" style="margin-left: 10px; font-size: 12px; padding: 5px 10px;">
          重试
        </button>
      </div>

      <button @click="cancelProcessing" style="background: #dc3545; margin-top: 15px;">
        取消处理
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import { getTaskStatus, cancelTask } from '../api'

const props = defineProps(['taskId', 'mode', 'enhancedVideoId'])
const emit = defineEmits(['complete', 'error'])

const taskProgress = ref(0.0)
const taskMessage = ref('初始化...')
const connectionAlive = ref(true)
const connectionFailCount = ref(0)
const processingStartTime = ref(Date.now())
const processingElapsedTime = ref('00:00')
let pollingInterval = null
let timeInterval = null

const startPolling = () => {
  pollingInterval = setInterval(async () => {
    try {
      const response = await getTaskStatus(props.taskId)
      const task = response.data

      // 连接成功，重置失败计数
      connectionFailCount.value = 0
      connectionAlive.value = true

      // 保持浮点数精度
      taskProgress.value = parseFloat(task.progress) || 0.0
      taskMessage.value = task.message

      if (task.status === 'completed') {
        stopPolling()
        emit('complete')
      } else if (task.status === 'failed') {
        stopPolling()
        emit('error')
      }
    } catch (error) {
      console.error('Polling error:', error)
      connectionFailCount.value++
      
      if (connectionFailCount.value >= 3) {
        connectionAlive.value = false
      }
    }
  }, 2000)
}

const stopPolling = () => {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
  if (timeInterval) {
    clearInterval(timeInterval)
    timeInterval = null
  }
}

const updateElapsedTime = () => {
  timeInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - processingStartTime.value) / 1000)
    const minutes = Math.floor(elapsed / 60)
    const seconds = elapsed % 60
    processingElapsedTime.value = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  }, 1000)
}

const retryConnection = () => {
  connectionFailCount.value = 0
  connectionAlive.value = true
}

const cancelProcessing = async () => {
  if (!confirm('确定要取消处理吗？')) {
    return
  }
  
  stopPolling()
  
  try {
    const response = await cancelTask(props.taskId)
    if (response.data.success) {
      taskMessage.value = '任务已取消'
      alert('任务取消成功')
      emit('error')
    } else {
      alert('取消失败: ' + response.data.message)
      startPolling()
    }
  } catch (error) {
    console.error('Cancel error:', error)
    alert('取消失败: 无法连接到服务器')
    startPolling()
  }
}

onMounted(() => {
  startPolling()
  updateElapsedTime()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped>
.progress-area {
    margin-top: 20px;
}

.progress-bar {
    width: 100%;
    height: 50px;
    background: #e9ecef;
    border-radius: 25px;
    overflow: hidden;
    margin: 20px 0;
    position: relative;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #007bff, #0056b3);
    transition: width 0.5s ease;
    position: relative;
}

.progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 18px;
    font-weight: bold;
    color: #333;
    z-index: 10;
    text-shadow: 0 0 3px white;
}

.progress-info {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    color: #666;
    font-size: 14px;
}

.alert {
    padding: 12px 20px;
    border-radius: 6px;
    margin: 15px 0;
}

.alert-info {
    background: #d1ecf1;
    color: #0c5460;
    border: 1px solid #bee5eb;
}

.alert-warning {
    background: #fff3cd;
    color: #856404;
    border: 1px solid #ffeeba;
}

.alert-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
</style>
