# 前后端 npm 一键启动

## 目标
检查用户已修改的根目录与前端 package.json，实现单命令启动前后端并说明浏览器访问方式。

## 完成事项与关键修改
- 保留根目录 npm start / npm run dev / npm run build 入口，以 npm workspace 管理 frontend，锁文件统一迁移到根目录。
- 前端使用 concurrently 同时启动 FastAPI 和 Vite，其中一个退出时停止另一个，替换原有 shell 后台启动与 kill 0。
- 后端通过 conda run --no-capture-output -n gatppo python 启动，去掉用户目录绝对路径；Vite 固定 5173 并启用 strictPort，后端使用 8000。
- .gitignore 的 node_modules/ 规则覆盖根目录和子目录。
- README 与系统设计文档同步启动、数据库首次配置和访问说明。

## 验证证据
- npm install、npm ls --depth=0 成功。
- 从根目录执行 npm start，前端首页与后端 /docs 均返回 HTTP 200。
- 通过 5173 代理访问 /api/health 和 /api/env/presets 均返回 HTTP 200，包含三套预设。
- small_smoke 内存仿真创建返回 201、单步返回 200、删除返回 204，未运行训练，未新增 results 产物。
- npm run build 成功；构建仍提示现有产物体积大于 500 kB。
- Ctrl+C 后前后端进程停止，ss 检查 5173 与 8000 无监听。
- git diff --check 通过。未执行 Git 提交或推送。

## 未解决事项与后续建议
- 当前 MySQL 3306 无监听，且无 backend/.env；健康检查 db_connected 为 false。数据库相关功能未验证，需按 README 启动 MySQL、配置账号并导入 schema。
- npm install 报告 1 项 moderate 依赖安全问题，本次未进行无关依赖大版本升级。
- 验证后服务已停止，用户执行 npm start 即可重新启动。
