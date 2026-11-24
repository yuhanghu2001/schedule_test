from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.baselines import evaluate_heuristic
from src.environment import EdgeEnv
from src.evaluation import DAG_EVALUATION_METRICS, EVALUATION_METRICS, evaluate_episode, summarize_metrics
from src.models.policy import MODEL_NAMES

from ..db import EnvConfig, EvaluationRecord, get_db, row_dict
from ..schemas import EvaluationCreate
from ..services.sim_service import load_model_for_environment, request_config

router = APIRouter(prefix="/api/experiments")


def evaluate_strategy(strategy, cfg, seeds):
    checkpoint_path = None
    if strategy.method == "checkpoint" or strategy.method in MODEL_NAMES:
        checkpoint_path = strategy.checkpoint_path
        if checkpoint_path is None:
            raise HTTPException(422, f"{strategy.method} 策略需要 checkpoint_path")
        model, model_type = load_model_for_environment(checkpoint_path, cfg, strategy.method)
        runs = [{"seed": seed, **evaluate_episode(model, cfg, seed, model_type)} for seed in seeds]
    else:
        if strategy.method == "heft_hetero" and not cfg.enable_dag_mode:
            raise HTTPException(422, "heft_hetero 需要启用 enable_dag_mode")
        runs = [
            {"seed": seed,
             **evaluate_heuristic(EdgeEnv(replace(cfg, seed=seed)), strategy.method)}
            for seed in seeds
        ]

    fields = EVALUATION_METRICS + (DAG_EVALUATION_METRICS if cfg.enable_dag_mode else ())
    summary = {f"{metric}_{stat}": values[stat]
               for metric, values in summarize_metrics(runs, fields).items()
               for stat in ("mean", "std")}
    return {
        "method": strategy.method,
        "name": strategy.name or strategy.method,
        "checkpoint_path": checkpoint_path,
        "summary": summary,
        "runs": runs,
    }


@router.post("/evaluate")
def evaluate(body: EvaluationCreate, db: Session = Depends(get_db)):
    saved = None
    if body.env_config_id is not None:
        row = db.get(EnvConfig, body.env_config_id)
        if row is None:
            raise HTTPException(404, "环境配置不存在")
        saved = row.config_json
    cfg = request_config(body, saved)
    strategies = [evaluate_strategy(strategy, cfg, body.seeds) for strategy in body.strategies]
    if body.persist:
        rows = [
            EvaluationRecord(
                name=body.name, method=result["method"], checkpoint_path=result["checkpoint_path"],
                env_config_id=body.env_config_id, seeds_json=body.seeds, summary_json=result,
            )
            for result in strategies
        ]
        db.add_all(rows)
        db.commit()
        for row, result in zip(rows, strategies):
            result["id"] = row.id
    return {"name": body.name, "seeds": body.seeds, "strategies": strategies}


@router.get("/evaluations")
def list_evaluations(db: Session = Depends(get_db), offset: int = Query(0, ge=0),
                     limit: int = Query(100, gt=0)):
    query = select(EvaluationRecord).order_by(EvaluationRecord.id.desc()).offset(offset).limit(limit)
    return [row_dict(row) for row in db.scalars(query)]
