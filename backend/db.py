from sqlalchemy import BigInteger, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.mysql import DATETIME, DOUBLE
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import MYSQL_DSN

engine = create_engine(MYSQL_DSN)
SessionLocal = sessionmaker(engine)
NOW = text("CURRENT_TIMESTAMP(6)")
TABLE_OPTIONS = {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"}


class Base(DeclarativeBase):
    pass


class EnvConfig(Base):
    __tablename__ = "env_configs"
    __table_args__ = TABLE_OPTIONS

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    preset_name: Mapped[str | None] = mapped_column(String(32))
    description: Mapped[str] = mapped_column(Text)
    config_json: Mapped[dict] = mapped_column(JSON)
    created_at = mapped_column(DATETIME(fsp=6), nullable=False, server_default=NOW)
    updated_at = mapped_column(
        DATETIME(fsp=6), nullable=False,
        server_default=text("CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)"),
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = TABLE_OPTIONS

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_name: Mapped[str] = mapped_column(String(128))
    model_type: Mapped[str] = mapped_column(String(16))
    env_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("env_configs.id", ondelete="SET NULL")
    )
    config_json: Mapped[dict] = mapped_column(JSON)
    episodes: Mapped[int] = mapped_column(Integer)
    current_episode: Mapped[int] = mapped_column(Integer)
    seed: Mapped[int] = mapped_column(BigInteger)
    device: Mapped[str] = mapped_column(String(16))
    threads: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16))
    best_score: Mapped[float | None] = mapped_column(DOUBLE)
    results_dir: Mapped[str] = mapped_column(String(512), unique=True)
    best_checkpoint: Mapped[str | None] = mapped_column(Text)
    last_checkpoint: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at = mapped_column(DATETIME(fsp=6), nullable=False, server_default=NOW)
    finished_at = mapped_column(DATETIME(fsp=6), nullable=True)


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (UniqueConstraint("run_id", "episode"), TABLE_OPTIONS)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"))
    episode: Mapped[int] = mapped_column(Integer)
    episode_reward: Mapped[float | None] = mapped_column(DOUBLE)
    objective_score: Mapped[float | None] = mapped_column(DOUBLE)
    task_success_rate: Mapped[float | None] = mapped_column(DOUBLE)
    acceptance_rate: Mapped[float | None] = mapped_column(DOUBLE)
    deadline_miss_rate: Mapped[float | None] = mapped_column(DOUBLE)
    avg_delay_s: Mapped[float | None] = mapped_column(DOUBLE)
    avg_energy_j: Mapped[float | None] = mapped_column(DOUBLE)
    mean_load_variance: Mapped[float | None] = mapped_column(DOUBLE)
    load_balance_degree: Mapped[float | None] = mapped_column(DOUBLE)
    mean_hops_accepted: Mapped[float | None] = mapped_column(DOUBLE)
    mean_path_length: Mapped[float | None] = mapped_column(DOUBLE)
    loss: Mapped[float | None] = mapped_column(DOUBLE)
    approx_kl: Mapped[float | None] = mapped_column(DOUBLE)
    entropy: Mapped[float | None] = mapped_column(DOUBLE)
    explained_variance: Mapped[float | None] = mapped_column(DOUBLE)
    created_at = mapped_column(DATETIME(fsp=6), nullable=False, server_default=NOW)


class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"
    __table_args__ = TABLE_OPTIONS

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    method: Mapped[str] = mapped_column(String(32))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("training_runs.id", ondelete="SET NULL"))
    checkpoint_path: Mapped[str | None] = mapped_column(Text)
    env_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("env_configs.id", ondelete="SET NULL")
    )
    seeds_json: Mapped[list] = mapped_column(JSON)
    summary_json: Mapped[dict] = mapped_column(JSON)
    created_at = mapped_column(DATETIME(fsp=6), nullable=False, server_default=NOW)


class SimulationSnapshot(Base):
    __tablename__ = "simulation_snapshots"
    __table_args__ = TABLE_OPTIONS

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    env_config_id: Mapped[int | None] = mapped_column(
        ForeignKey("env_configs.id", ondelete="SET NULL")
    )
    policy_label: Mapped[str] = mapped_column(String(32))
    seed: Mapped[int] = mapped_column(BigInteger)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    trace_json: Mapped[list] = mapped_column(JSON)
    created_at = mapped_column(DATETIME(fsp=6), nullable=False, server_default=NOW)


def get_db():
    with SessionLocal() as db:
        yield db


def row_dict(row):
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}
