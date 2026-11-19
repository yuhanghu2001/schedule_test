"""启动项目专用 MySQL、FastAPI 和 Vite；Ctrl+C 后按依赖顺序停止。"""
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time

import pymysql
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
MYSQL_ENV = LOCAL / "mysql-env"
DATA = LOCAL / "mysql-data"
SOCKET = LOCAL / "mysql.sock"
ADMIN_FILE = LOCAL / "mysql-admin.cnf"
ENV_FILE = ROOT / "backend" / ".env"


def private_file(path, content):
    with open(path, "w", opener=lambda name, flags: os.open(name, flags, 0o600)) as stream:
        stream.write(content)


def main():
    LOCAL.mkdir(exist_ok=True)
    mysqld = MYSQL_ENV / "bin" / "mysqld"
    if not mysqld.exists():
        print("[mysql] 首次安装 MySQL 8.4.2（独立环境，不修改 gatppo）", flush=True)
        subprocess.run([
            "conda", "create", "--prefix", str(MYSQL_ENV), "--override-channels",
            "-c", "conda-forge", "mysql-server=8.4.2", "mysql-client=8.4.2", "-y",
        ], check=True)
    if not ENV_FILE.exists():
        private_file(ENV_FILE, "MYSQL_HOST=127.0.0.1\nMYSQL_PORT=3306\n"
                     f"MYSQL_USER=schedule\nMYSQL_PASSWORD={secrets.token_hex(24)}\n"
                     "MYSQL_DATABASE=schedule_test\n")

    from backend.config import MYSQL_DSN
    url = make_url(MYSQL_DSN)
    first_start = not (DATA / "mysql").exists()
    init_file = LOCAL / "mysql-init.sql"
    options = [str(mysqld), "--no-defaults", f"--basedir={MYSQL_ENV}", f"--datadir={DATA}"]
    if first_start:
        print("[mysql] 首次初始化数据库目录", flush=True)
        subprocess.run([*options, "--initialize-insecure"], check=True)
        root_password = secrets.token_hex(24)
        private_file(ADMIN_FILE, f"[client]\nuser=root\npassword={root_password}\nsocket={SOCKET}\n")
        private_file(init_file, f"ALTER USER 'root'@'localhost' IDENTIFIED BY '{root_password}';\n")

    mysql_started = False
    services = []
    try:
        print(f"[mysql] 启动 127.0.0.1:{url.port}，日志：{LOCAL / 'mysql.log'}", flush=True)
        subprocess.run([
            *options, "--daemonize", "--bind-address=127.0.0.1", f"--port={url.port}",
            f"--socket={SOCKET}", f"--pid-file={LOCAL / 'mysql.pid'}",
            f"--log-error={LOCAL / 'mysql.log'}", "--mysqlx=OFF",
            *([f"--init-file={init_file}"] if first_start else []),
        ], check=True)
        mysql_started = True
        if first_start:
            init_file.unlink()
        with pymysql.connect(read_default_file=str(ADMIN_FILE), autocommit=True) as admin:
            with admin.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{url.database}` CHARACTER SET utf8mb4")
                cursor.execute("CREATE USER IF NOT EXISTS %s@'127.0.0.1' IDENTIFIED BY %s",
                               (url.username, url.password))
                cursor.execute(f"GRANT ALL PRIVILEGES ON `{url.database}`.* TO %s@'127.0.0.1'",
                               (url.username,))
        with pymysql.connect(host=url.host, port=url.port, user=url.username,
                             password=url.password, database=url.database,
                             charset="utf8mb4", autocommit=True) as database:
            with database.cursor() as cursor:
                for statement in (ROOT / "backend" / "schema.sql").read_text().split(";"):
                    if statement.strip():
                        cursor.execute(statement)
        print("[mysql] 已就绪，数据库与表已初始化", flush=True)
        services.append(subprocess.Popen([
            sys.executable, "-m", "uvicorn", "backend.app:app",
            "--host", "0.0.0.0", "--port", "8000",
        ], cwd=ROOT, start_new_session=True))
        services.append(subprocess.Popen([
            "npm", "--prefix", str(ROOT / "frontend"), "run", "dev:web",
        ], cwd=ROOT, start_new_session=True))
        while all(service.poll() is None for service in services):
            time.sleep(0.2)
        return next(service.returncode for service in services if service.returncode is not None)
    finally:
        for service in reversed(services):
            if service.poll() is None:
                os.killpg(service.pid, signal.SIGTERM)
                service.wait()
        if mysql_started:
            print("[mysql] 关闭数据库，保留数据文件", flush=True)
            subprocess.run([str(MYSQL_ENV / "bin" / "mysqladmin"),
                            f"--defaults-file={ADMIN_FILE}", "shutdown"], check=True)


def stop(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("[dev] 已停止", flush=True)
