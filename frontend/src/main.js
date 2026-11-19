import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/main.css'
import './styles/dashboard.css'
document.documentElement.classList.add('dark')
import App from './App.vue'
import router from './router'
createApp(App).use(createPinia()).use(router).use(ElementPlus).mount('#app')
