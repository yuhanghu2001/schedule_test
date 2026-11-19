# 删除拓扑左右下角外框折线

## 目标与修改

删除用户截图标出的左右下角细外框折线。移除 NetworkMonitorScene.vue 中 monitor-outer-frame 的完整 SVG path，并删除 dashboard.css 对应无用样式；保留平台本身发光边缘、背景和真实节点连接。同步 docs/07 与两种桌面分辨率截图。

## 验证

npm run build 成功，仅有既有包体积提示；git diff --check 通过。Playwright 实开 localhost:8888/topology，确认外框元素不存在，平台发光边缘及 37 个节点保留，1920×1080、1600×900 无页面滚动，页面/控制台无错误。临时浏览器依赖位于 /tmp，验证后删除。

## 边界

仅删除指定外框，不修改后端、核心算法或正式结果；此前全部工作区修改保留。本轮未要求提交推送，未执行 commit/push。
