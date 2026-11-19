import axios from 'axios'
import { ElMessage } from 'element-plus'
const api = axios.create({ baseURL: '/api' })
api.interceptors.response.use(response => response.data, error => { ElMessage.error(error.response?.data?.detail || error.message || '请求失败'); return Promise.reject(error) })
export const trainingSocket = id => new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/training/${id}`)
export default api
