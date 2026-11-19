from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Preset = Literal["default", "high_congestion", "small_smoke"]
ModelType = Literal["gat", "flat", "shuffled_gat"]


class EnvRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    preset_name: Preset = "default"
    config: dict = Field(default_factory=dict)
    seed: int | None = None


class EnvConfigCreate(EnvRequest):
    name: str
    description: str = ""


class SimCreate(EnvRequest):
    policy: Literal["shortest_fastest", "greedy_estimate", "random", "checkpoint"] = (
        "shortest_fastest"
    )
    checkpoint_path: str | None = None


class SimStep(BaseModel):
    action_id: int | None = None


class LinkPerturbation(BaseModel):
    source: int
    target: int
    active: bool | None = None
    congestion: float | None = Field(default=None, ge=0, lt=1)
    background_fwd: float | None = Field(default=None, ge=0, lt=1)
    background_rev: float | None = Field(default=None, ge=0, lt=1)


class TrainingCreate(EnvRequest):
    run_name: str | None = None
    model_type: ModelType = "gat"
    env_config_id: int | None = None
    episodes: int = Field(default=1000, gt=0)
    device: Literal["auto", "cpu", "cuda"] = "cpu"
    threads: int = Field(default=2, gt=0)
    resume_path: str | None = None


class EvaluationStrategy(BaseModel):
    method: Literal["shortest_fastest", "greedy_estimate", "checkpoint"]
    name: str | None = None
    checkpoint_path: str | None = None


class EvaluationCreate(EnvRequest):
    name: str = "对比评估"
    env_config_id: int | None = None
    seeds: list[int] = Field(min_length=1)
    strategies: list[EvaluationStrategy] = Field(min_length=1)
    persist: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"]
    db_connected: bool
    results_dir: str
    active_training_runs: int
    active_sim_sessions: int


class SimCreatedResponse(BaseModel):
    session_id: str
    snapshot: dict
