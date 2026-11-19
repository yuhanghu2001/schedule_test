# 删除拓扑红框说明与标签

## 目标
按用户图片删除红框中的节点数量、层级标签、云名称框、右上图例和底部状态信息。

## 完成事项与关键修改
- TopologyGraph.vue 删除 scene-info、scene-key、layer-label、云节点名称/底框及 graph-legend。
- OverviewDashboard.vue 删除拓扑 network-footer（时隙、任务进度、链路数量）。
- 删除对应闲置 SVG 渐变与 CSS；保留真实云节点图形，将边云连接和高亮箭头端点调整至云图形边界。
- 保留边缘/终端节点与真实连线、推演操作、分组和数据联动；云节点点击与绿色高亮仍可用。
- 更新系统设计文档及实际截图，保留用户和前序工作区全部修改。

## 验证证据
- npm run build 与 git diff --check 通过。
- Chromium 验证指定红框区域 DOM 均为 0；真实节点37、网络链路30、无线接入24条不变。
- 实际会话按时隙推进，任务 T-3 的三段路径高亮正常；云节点详情、重置通过。
- 1920×1080、1600×900 页面尺寸与视口相等，pageerror 为零；人工查看拓扑区域截图，确认云图形与连线衔接正常。
- 更新 docs/screenshots/topology-*、dashboard-* 与 mysql-connected.png。
- 复用用户已运行服务，无服务重启、数据库记录写入、训练或 results 新增。临时验证会话离开时删除，/tmp 浏览器脚本与依赖完成后删除。

## 未解决事项及后续建议
本次无未解决事项。刷新 http://localhost:8888 查看。未提交或推送。
