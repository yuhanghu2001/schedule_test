# 精简导航并放大标题状态及排查 MySQL

## 目标
移除训练工作台和实验记录导航，继续放大主标题和系统连接状态文字，解释 MySQL 未连接原因。

## 完成事项与关键修改
- App.vue 移除两个导航链接，保留系统总览、节点管理与真实数据库状态。
- OverviewDashboard.vue 同时移除训练工作台跳转入口；原业务路由与实现保留。
- dashboard.css 将桌面标题设为 46px/40px（1920/1600），系统连接状态设为 20px/18px；保留小屏适配。
- README 与系统设计说明同步，更新两种分辨率的总览和节点管理实际截图。

## 验证证据
- 实查实际数据库地址为 127.0.0.1:3306/schedule_test；3306 无监听，mysql/mariadb 服务均 inactive，backend/.env 不存在。
- SQLAlchemy 连接实测报 PyMySQL 2003 Connection refused。后端正常可达与数据库未连接是独立状态，未隐瞒或伪改标签。
- npm run build、git diff --check 通过。
- Chromium 实测两种分辨率无旧导航链接或首页训练跳转，标题与状态无重叠，页面尺寸无溢出，pageerror 为零。截图位于 docs/screenshots/。
- 临时浏览器脚本和依赖在 /tmp 下运行，完成后删除，未写 results。

## 未解决事项及后续建议
MySQL 仍需安装/启动并按 README 配置数据库账号、库表与 backend/.env；本次仅排查，未擅自安装服务或编造连接成功。本机草稿和内存推演仍可用。服务保持运行，刷新 http://localhost:8888 查看。未提交或推送，保留所有先前工作区修改。
