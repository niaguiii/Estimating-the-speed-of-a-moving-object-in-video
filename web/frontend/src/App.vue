<template>
  <div class="header">
    <h1>🎥 视频速度估算系统</h1>
    <p>基于YOLOv8检测 + ByteTrack追踪 + RAFT光流的智能速度估算</p>
  </div>

  <div class="container">
    <VideoUpload 
      v-if="currentStep === 'upload'"
      @upload-success="handleUploadSuccess"
    />

    <ModeSelector 
      v-if="currentStep === 'select'"
      :video-id="videoId"
      @select-mode="handleModeSelect"
      @back="currentStep = 'upload'"
    />

    <ProgressBar
      v-if="currentStep === 'processing'"
      :task-id="taskId"
      :mode="selectedMode"
      @complete="handleComplete"
      @error="handleError"
    />

    <ResultDisplay
      v-if="currentStep === 'result'"
      :task-id="taskId"
      :mode="selectedMode"
      @reset="reset"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'
import VideoUpload from './components/VideoUpload.vue'
import ModeSelector from './components/ModeSelector.vue'
import ProgressBar from './components/ProgressBar.vue'
import ResultDisplay from './components/ResultDisplay.vue'

const currentStep = ref('upload')
const videoId = ref('')
const selectedMode = ref(null)
const taskId = ref('')

const handleUploadSuccess = (id) => {
  videoId.value = id
  currentStep.value = 'select'
}

const handleModeSelect = (mode, task) => {
  selectedMode.value = mode
  taskId.value = task
  currentStep.value = 'processing'
}

const handleComplete = () => {
  currentStep.value = 'result'
}

const handleError = () => {
  alert('处理失败，请重试')
  reset()
}

const reset = () => {
  currentStep.value = 'upload'
  videoId.value = ''
  selectedMode.value = null
  taskId.value = ''
}
</script>
