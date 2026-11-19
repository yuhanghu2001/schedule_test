import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
load_dotenv(Path(__file__).with_name(".env"))

MYSQL_DSN = os.getenv("MYSQL_DSN") or URL.create(
    "mysql+pymysql",
    username=os.getenv("MYSQL_USER", "schedule"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    database=os.getenv("MYSQL_DATABASE", "schedule_test"),
    query={"charset": "utf8mb4"},
).render_as_string(hide_password=False)
