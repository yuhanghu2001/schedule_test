from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, Query
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.baselines import evaluate_heuristic
from src.environment import EdgeEnv
from src.evaluation import EVALUATION_METRICS, evaluate_episode

from ..db import EnvConfig, EvaluationRecord, get_db, row_dict
from ..schemas import EvaluationCreate
from ..services.sim_service import build_model, checkpoint_config, load_checkpoint, request_config

router = APIRouter(prefix="/api/experiments")


def evaluate_strategy(strategy, cfg, seeds):
    checkpoint_path = strategy.checkpoint_path if strategy.method == "checkpoint" else None
    if strategy.method == "checkpoint":
        if checkpoint_path is None:
            raise HTTPException(422, "checkpoint 策略需要 checkpoint_path")
        checkpoint = load_checkpoint(checkpoint_path)
        model_type = checkpoint["model_type"]
        model = build_model(checkpoint_config(checkpoint), model_type, checkpoint, inference=True)
        runs = [{"seed": seed, **evaluate_episode(model, cfg, seed, model_type)} for seed in seeds]
    else:
        runs = [
            {"seed": seed,
             **evaluate_heuristic(EdgeEnv(replace(cfg, seed=seed)), strategy.method)}
            for seed in seeds
        ]

    summary = {}
    for metric in EVALUATION_METRICS:
        values = [run[metric] for run in runs if run[metric] is not None]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
        summary[f"{metric}_std"] = float(np.std(values)) if values else None
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
