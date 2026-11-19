# 排查 npm 端口占用与 MySQL 服务端缺失

## 目标
解释用户 npm start 截图报错及 gatppo 环境含 MySQL 依赖仍未连库的原因，处理本代理遗留的开发服务端口占用。

## 完成事项
- 检查根目录及 frontend/package.json，npm start 实际仅启动 Vite 与 Uvicorn，没有数据库服务进程。
- 检查 Conda gatppo 包清单，相关依赖为 pymysql 1.2.4、sqlalchemy 2.1.9；它们是 Python 数据库客户端/访问库。
- 检查 gatppo/bin、/usr/bin、/usr/sbin，没有找到 mysql/mysqld/mariadb 服务端或客户端可执行文件；3306 无监听。
- 用户截图实际报错为 Port 8888 is already in use；占用者为前次代理验证后保留的 Vite 与 FastAPI 服务（进程 357109、357141）。已通过原会话 Ctrl+C 停止它们。
- 未安装数据库、未修改启动代码、未编造数据库连接成功；向用户解释需要 MySQL 服务端、库表初始化及连接配置。

## 验证证据
conda list -n gatppo --json 读取相关包。停止后 ss 确认 8888、8000、3306 均无监听。前次实际连库已确认 127.0.0.1:3306 Connection refused，backend/.env 尚未配置。

## 未解决事项与后续建议
用户现在可重新执行 npm start 启动前后端。MySQL 尚需单独安装、初始化与启动；当前 npm 脚本没有启动数据库的功能。未提交或推送，保留既有工作区修改。
