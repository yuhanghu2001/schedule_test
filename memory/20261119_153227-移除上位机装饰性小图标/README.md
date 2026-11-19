# 移除上位机装饰性小图标

## 目标
按用户截图要求，移除标题和 KPI 等位置的同类装饰图标。

## 完成事项与关键修改
- OverviewDashboard.vue 删除标题、四项 KPI、策略名称及参数、资源趋势和流程卡片中的装饰符号。
- NodeManagement.vue 删除三个面板标题前的装饰图标。
- TopologyGraph.vue 删除拓扑文字图例和数量说明前的装饰字符；保留实际节点图形及路径箭头。
- dashboard.css 删除对应闲置图标样式，更新文档及实际页面截图。

## 验证证据
- npm run build、git diff --check 通过。
- Chromium 在 1920×1080、1600×900 实测：指定装饰图标元素为 0，真实拓扑节点仍为 37，pageerror 为零。
- 人工查看实际截图确认文字对齐与布局正常；截图位于 docs/screenshots/dashboard-* 和 nodes-*。
- 复用用户已运行的服务，没有重启或终止用户进程；未操作数据库记录或运行训练，未写 results。
- 临时浏览器脚本及依赖在 /tmp，完成后删除。

## 未解决事项与后续建议
本次无未解决事项。真实节点、路径与状态标记保留；刷新 http://localhost:8888 查看。未提交或推送。
