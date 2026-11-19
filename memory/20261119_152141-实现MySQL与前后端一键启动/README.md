# 实现 MySQL 与前后端一键启动

## 目标
用户明确要求完成数据库安装、初始化与服务管理，使根目录 npm start 同时启动 MySQL、FastAPI 和 Vite。

## 完成事项与关键修改
- 实查 Ubuntu/WSL、无免密 sudo、无 MySQL 服务端，采用用户目录内独立 Conda 环境，不改动 gatppo 的训练依赖。
- 安装 conda-forge mysql-server/mysql-client 8.4.2 到 .local/mysql-env，初始化 .local/mysql-data。
- 新增 scripts/dev.py，frontend/package.json 的 dev/start 统一调用 gatppo Python 中的该模块；根目录原有入口继续转发。
- 首次启动自动安装缺少的 MySQL 二进制、生成 backend/.env 随机应用密码、初始化目录、设置随机 root 密码，建立 schedule_test、应用账户及 schema.sql 的五张表。
- MySQL daemonize 成功后才启动当前 gatppo Python 的 Uvicorn 与 Vite；监视应用子进程，任一退出或 Ctrl+C 时先等待应用退出，再执行 mysqladmin shutdown，保留数据库数据。
- .local/mysql-admin.cnf 和 backend/.env 权限为 600；.local/ 与 backend/.env 均在 Git 忽略规则内，不输出密码。
- 移除不再使用的 concurrently，更新根锁文件。README 和系统设计文档同步真实启动步骤、路径和数据库状态。
- 保留前序工作区全部改动；本次出现的 .vscode/settings.json 用户改动未触碰。

## 验证证据
- MySQL 8.4.2 安装命令成功，版本输出确认服务端已安装。
- 根目录 npm start 首次初始化并同时启动 127.0.0.1:3306、FastAPI 8000、Vite 8888。
- 经前端代理 GET /api/health 返回 db_connected:true，前端 HTTP 200。
- POST /api/env/configs 保存中文测试配置返回 201，GET 列表返回 200；完整 Ctrl+C 关闭后再次 npm start，记录仍在。完成验证后 DELETE 返回 204，删除测试数据。
- Playwright + Chromium 打开真实页面，显示“MySQL 已连接”，pageerror 为零；截图 docs/screenshots/mysql-connected.png。
- Ctrl+C 后日志确认 Uvicorn 完成应用关闭，再关闭数据库；ss 确认三个端口均释放。
- 模拟已有监听占用 8888，npm start 非零退出并显示端口占用，同时清理本次启动的 API/MySQL；随后三个端口均可绑定。
- npm install、npm run build、git diff --check 通过；原有大包体积与 moderate 依赖提示未扩大为本次无关升级。
- 已检查凭据文件权限、Git 忽略规则；没有提交运行态数据库、二进制环境或密码。
- 临时浏览器脚本、依赖及测试记录标识使用 /tmp，完成后删除；未训练、未新增 results 产物。

## 未解决事项及后续使用
本次三服务一键启动目标已完成。当前验证服务已停止，8888/8000/3306 均无监听，避免用户再次启动冲突。用户在根目录执行 npm start 后打开 http://localhost:8888，Ctrl+C 停止并保留数据。新机器首次安装需联网，使用现有 gatppo Python 环境和 npm 依赖。未执行 Git 提交或推送。
