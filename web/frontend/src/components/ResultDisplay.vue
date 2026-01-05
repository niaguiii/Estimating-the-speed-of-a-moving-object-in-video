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

      <div style="margin-top: 20px; text-align: center;">
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
import { computed } from 'vue'
import { getDownloadUrl } from '../api'

const props = defineProps(['taskId'])
const emit = defineEmits(['reset'])

const resultVideoUrl = computed(() => getDownloadUrl(props.taskId))
const downloadUrl = computed(() => getDownloadUrl(props.taskId))
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
</style>
