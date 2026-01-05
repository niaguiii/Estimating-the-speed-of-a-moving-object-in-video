<template>
  <div class="section">
    <div class="section-title">📤 步骤 1: 上传视频</div>
    
    <div 
      class="upload-area" 
      :class="{ dragging: isDragging }"
      @click="$refs.fileInput.click()"
      @dragover.prevent="isDragging = true"
      @dragleave.prevent="isDragging = false"
      @drop.prevent="handleDrop"
    >
      <p style="font-size: 48px; margin-bottom: 10px;">📁</p>
      <p style="color: #666; margin-bottom: 10px;">拖拽视频文件到这里或点击上传</p>
      <p style="color: #999; font-size: 12px;">支持 MP4, AVI, MOV 格式</p>
      <input 
        ref="fileInput" 
        type="file" 
        class="file-input" 
        accept="video/*"
        @change="handleFileSelect"
      >
    </div>

    <div v-if="uploadedFile" class="file-info">
      <p><strong>已选择:</strong> {{ uploadedFile.name }}</p>
      <p><strong>大小:</strong> {{ formatFileSize(uploadedFile.size) }}</p>
    </div>

    <!-- 上传进度条 -->
    <div v-if="uploading" class="upload-progress">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: uploadProgress + '%' }"></div>
        <div class="progress-text">{{ uploadProgress }}%</div>
      </div>
      <p class="progress-info">正在上传...</p>
    </div>

    <div v-if="uploadMessage" :class="['alert', uploadSuccess ? 'alert-success' : 'alert-error']">
      {{ uploadMessage }}
    </div>

    <button 
      v-if="uploadedFile && !uploadSuccess" 
      @click="uploadVideo" 
      :disabled="uploading"
    >
      {{ uploading ? '上传中...' : '上传视频' }}
    </button>

    <button v-if="uploadSuccess" @click="$emit('upload-success', videoId)">
      下一步: 选择处理模式 →
    </button>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { uploadVideo as apiUploadVideo } from '../api'

const emit = defineEmits(['upload-success'])

const isDragging = ref(false)
const uploadedFile = ref(null)
const uploading = ref(false)
const uploadSuccess = ref(false)
const uploadMessage = ref('')
const videoId = ref('')
const uploadProgress = ref(0)

const handleFileSelect = (event) => {
  const file = event.target.files[0]
  if (file) {
    uploadedFile.value = file
    uploadSuccess.value = false
    uploadMessage.value = ''
  }
}

const handleDrop = (event) => {
  isDragging.value = false
  const file = event.dataTransfer.files[0]
  if (file && file.type.startsWith('video/')) {
    uploadedFile.value = file
    uploadSuccess.value = false
    uploadMessage.value = ''
  } else {
    uploadMessage.value = '请上传视频文件'
  }
}

const formatFileSize = (bytes) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}

const uploadVideo = async () => {
  if (!uploadedFile.value) return

  uploading.value = true
  uploadMessage.value = ''
  uploadProgress.value = 0

  try {
    const response = await apiUploadVideo(uploadedFile.value, (progressEvent) => {
      // 计算上传进度
      const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
      uploadProgress.value = percentCompleted
    })
    
    if (response.data.success) {
      videoId.value = response.data.video_id
      uploadSuccess.value = true
      uploadProgress.value = 100
      uploadMessage.value = '✅ 上传成功！'
    } else {
      uploadMessage.value = '❌ 上传失败: ' + response.data.message
    }
  } catch (error) {
    console.error('Upload error:', error)
    uploadMessage.value = '❌ 上传失败: 无法连接到服务器'
  } finally {
    uploading.value = false
  }
}
</script>

<style scoped>
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 8px;
    padding: 40px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}

.upload-area:hover {
    border-color: #007bff;
    background: #f8f9fa;
}

.upload-area.dragging {
    border-color: #007bff;
    background: #e7f3ff;
}

.file-input {
    display: none;
}

.file-info {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    margin-top: 15px;
}

.file-info p {
    margin: 5px 0;
    color: #666;
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

.alert-error {
    background: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}

.upload-progress {
    margin: 15px 0;
}

.progress-bar {
    width: 100%;
    height: 30px;
    background: #e9ecef;
    border-radius: 15px;
    overflow: hidden;
    position: relative;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #28a745, #20c997);
    transition: width 0.3s ease;
    position: relative;
}

.progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 14px;
    font-weight: bold;
    color: #333;
    z-index: 10;
}

.progress-info {
    text-align: center;
    color: #666;
    font-size: 13px;
    margin-top: 8px;
}
</style>
