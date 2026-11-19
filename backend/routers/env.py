from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import load_config
from src.environment import EdgeEnv

from ..db import EnvConfig, get_db, row_dict
from ..schemas import EnvConfigCreate, EnvRequest, LinkPerturbation
from ..schemas import SimCreate, SimCreatedResponse, SimStep
from ..services.sim_service import (
    RESOURCE_NAMES, TASK_KIND_NAMES, request_config, serialize_env_snapshot,
)

router = APIRouter()


@router.get("/api/env/presets")
def env_presets():
    return {
        "presets": {name: asdict(load_config(name))
                    for name in ("default", "high_congestion", "small_smoke")},
        "task_kind_names": TASK_KIND_NAMES,
        "resource_names": RESOURCE_NAMES,
    }


@router.get("/api/env/configs")
def list_configs(db: Session = Depends(get_db)):
    return [row_dict(row) for row in db.scalars(select(EnvConfig).order_by(EnvConfig.id.desc()))]


@router.post("/api/env/configs", status_code=201)
def create_config(body: EnvConfigCreate, db: Session = Depends(get_db)):
    row = EnvConfig(name=body.name, preset_name=body.preset_name, description=body.description,
                    config_json=asdict(request_config(body)))
    db.add(row)
    db.commit()
    return row_dict(row)


@router.delete("/api/env/configs/{config_id}", status_code=204)
def delete_config(config_id: int, db: Session = Depends(get_db)):
    row = db.get(EnvConfig, config_id)
    if row is None:
        raise HTTPException(404, "环境配置不存在")
    db.delete(row)
    db.commit()
    return Response(status_code=204)


@router.post("/api/env/preview")
def preview(body: EnvRequest):
    return serialize_env_snapshot(EdgeEnv(request_config(body)), [])


@router.post("/api/sim/sessions", status_code=201, response_model=SimCreatedResponse)
def create_session(body: SimCreate, request: Request):
    if body.policy == "checkpoint" and body.checkpoint_path is None:
        raise HTTPException(422, "checkpoint 策略需要 checkpoint_path")
    session_id, snapshot = request.app.state.sim_manager.create_session(
        request_config(body), body.policy, body.checkpoint_path
    )
    return {"session_id": session_id, "snapshot": snapshot}


def _sim_manager(request: Request, session_id: str):
    manager = request.app.state.sim_manager
    if session_id not in manager.sessions:
        raise HTTPException(404, "推演会话不存在")
    return manager


@router.get("/api/sim/sessions/{session_id}")
def session_snapshot(session_id: str, manager=Depends(_sim_manager)):
    return manager.get_snapshot(session_id)


@router.post("/api/sim/sessions/{session_id}/step")
def step_once(session_id: str, body: SimStep | None = None, manager=Depends(_sim_manager)):
    return manager.step_once(session_id, body.action_id if body else None)


@router.post("/api/sim/sessions/{session_id}/step-slot")
def step_slot(session_id: str, manager=Depends(_sim_manager)):
    return manager.step_slot(session_id)


@router.post("/api/sim/sessions/{session_id}/perturb")
def perturb_link(session_id: str, body: LinkPerturbation, manager=Depends(_sim_manager)):
    try:
        return manager.perturb_link(session_id, **body.model_dump())
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/api/sim/sessions/{session_id}/reset")
def reset_session(session_id: str, manager=Depends(_sim_manager)):
    return manager.reset_session(session_id)


@router.delete("/api/sim/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, manager=Depends(_sim_manager)):
    manager.delete_session(session_id)
    return Response(status_code=204)
