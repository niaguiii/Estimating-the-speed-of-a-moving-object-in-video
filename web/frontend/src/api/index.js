import axios from 'axios'

const apiBase = import.meta.env.VITE_API_BASE

export const uploadVideo = (file, onUploadProgress) => {
  const formData = new FormData()
  formData.append('file', file)
  return axios.post(`${apiBase}/api/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    onUploadProgress: onUploadProgress
  })
}

export const processVideo = (videoId, mode) => {
  return axios.post(`${apiBase}/api/process`, {
    video_id: videoId,
    mode: mode
  })
}

export const getTaskStatus = (taskId) => {
  return axios.get(`${apiBase}/api/task/${taskId}`, {
    timeout: 5000
  })
}

export const getDownloadUrl = (taskId) => {
  return `${apiBase}/api/download/${taskId}`
}

export const getHistory = () => {
  return axios.get(`${apiBase}/api/history`)
}

export const cancelTask = (taskId) => {
  return axios.post(`${apiBase}/api/cancel/${taskId}`)
}
