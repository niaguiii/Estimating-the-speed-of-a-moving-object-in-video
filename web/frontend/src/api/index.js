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

export const processVideo = (videoId, mode, focalMm = null, depthFrequency = null) => {
  const body = {
    video_id: videoId,
    mode: mode
  }
  if (focalMm !== null) body.focal_mm = focalMm
  if (depthFrequency !== null) body.depth_frequency = depthFrequency
  return axios.post(`${apiBase}/api/process`, body)
}

export const getTaskStatus = (taskId) => {
  return axios.get(`${apiBase}/api/task/${taskId}`, {
    timeout: 5000
  })
}

export const getDownloadUrl = (taskId) => {
  return `${apiBase}/api/download/${taskId}`
}

export const getCsvFiles = (taskId) => {
  return axios.get(`${apiBase}/api/task/${taskId}`).then(res => ({
    csvFiles:  res.data.csv_files  || [],
    cropFiles: res.data.crop_files || [],
    zipUrl:    res.data.zip_url    || null,
  }))
}

export const getDataZipUrl = (taskId) => {
  return `${apiBase}/api/download-zip/${taskId}`
}

export const getHistory = () => {
  return axios.get(`${apiBase}/api/history`)
}

export const cancelTask = (taskId) => {
  return axios.post(`${apiBase}/api/cancel/${taskId}`)
}
