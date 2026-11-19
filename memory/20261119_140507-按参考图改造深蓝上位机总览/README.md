# 按参考图改造深蓝上位机总览

## 目标
按用户附图改造现有 Vue 上位机，实现深蓝三栏监控首页、真实仿真数据联动，并保留训练、完整沙盘、评估和实验历史。

## 完成事项与关键修改
- 已阅读 AGENTS、README、系统设计文档、原前端组件/路由/状态/样式及 backend/routers、sim_service 的真实快照结构。
- 使用 impeccable 技能参考，用户已给出完整产品与视觉要求；PRODUCT.md 记录已有仓库和用户要求，不另开设计访谈。
- 新增 OverviewDashboard.vue、ResourceTrend.vue、NodeGlyph.vue 和 dashboard.css；App.vue 改为居中标题、北京时间、系统状态和横向导航。
- 新首页及 /topology、/resources 复用总览，/training 保留训练工作台；历史参数回填跳转更新到 /training。
- TopologyGraph 改为云端、两排边缘和完整端侧分层布局；真实链路、拥塞状态、失效虚线、路径箭头、原生悬停详情、节点抽屉与拖动持久化。
- TaskFlowTable 增加紧凑模式，修正云端 C0 路径标签，保留原候选动作与手动执行。
- 指标/资源/表格/曲线取自同一快照；会话创建、步进、整时隙、播放/暂停、重置、环境切换、权重选择和离开清理已接入。
- 原训练/历史/Pinia 配置操作中 structuredClone 响应式代理导致 DataCloneError 的问题，以 toRaw 修正。
- 原训练图表和评估图表适配深蓝主题；更新页面标题、favicon、README 和系统设计说明。
- 未修改 backend/、src/、数学模型或优化目标，未新增项目运行依赖。

## 验证证据
- npm run build 通过；Vite 仍有既有的大包体积提示。
- npm start 已执行，因用户已有 8888/8000 服务运行，本次启动按 strictPort 报端口占用；随后复用现有同仓库 Vite/FastAPI 服务完成全部浏览器验证，未终止用户服务。
- 临时 Playwright + 本机 Chromium 浏览器实测：37 个真实节点（1 云、12 边缘、24 终端）；创建会话、单步、整时隙、自动播放/暂停、任务行路径选择、节点抽屉、重置与所有导航通过。
- 节点拖动位置写入原布局缓存；页面任务数量与完成率断言匹配实际后端 step 响应。
- 已有 GAT Checkpoint 的 small_smoke 单步推理通过，来源：results/20261118_195306/best_gat_ppo.pt，未修改该权重。
- 原沙盘单步、训练拓扑预览及算法/监控标签切换通过；small_smoke、种子 1042 的两启发式策略真实对比评估通过（persist=false）。未启动正式训练。
- 1920×1080 与 1600×900 检查 document scrollWidth/scrollHeight 等于视口，概况、策略、节点和资源面板无内容裁切；浏览器 console error 与 pageerror 为 0。
- 实际页面截图保存在 docs/screenshots/dashboard-1920x1080.png 和 dashboard-1600x900.png，为真实推演后暂停并选择任务的页面。
- git diff --check 通过。浏览器脚本和依赖使用 /tmp，不写 results；完成后删除临时文件，仅保留用户要求交付的截图。

## 未解决事项与视觉差异
- MySQL 未连接，因此未验证数据库保存、正式训练、MySQL 历史导入；现有入口保留，无伪造成功结论。
- 仓库缺少 frontend/reference/dashboard.png，已对照消息附图作人工视觉检查，没有像素差分结果。布局与配色参考附图；图标为规则 SVG，非原图三维素材，真实链路、节点画像与指标也不会照抄参考图。
- 为完整显示 24 个终端并保留实际操作，终端排列与控制栏占用空间与参考图略有不同。
- 既有构建大包提示未扩展为本次无关打包重构。
- 本次未提交或推送 Git。

## 后续使用
仓库根目录 npm start，访问 http://localhost:8888。已有服务运行时直接刷新页面。MySQL 首次配置见 README。
