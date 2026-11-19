# 前端端口改为 8888

## 目标
将前端开发访问端口从 5173 改为 8888。

## 完成事项与关键修改
- frontend/package.json 的 dev:web 使用 --port 8888，保留 strictPort。
- backend/app.py 的跨域来源端口同步为 8888。
- README 与 docs/07_上位机可视化调度系统设计.md 同步地址与端口。

## 验证证据
- npm start 成功启动前后端。
- http://127.0.0.1:8888/ 与 /api/health 返回 HTTP 200。
- 后端响应正确允许 http://localhost:8888 跨域来源。
- git diff --check 通过；验证后 Ctrl+C 停止服务。

## 未解决事项与后续建议
本次端口修改无未解决事项。运行 npm start 后打开 http://localhost:8888。未提交或推送，保留前序工作区改动。
