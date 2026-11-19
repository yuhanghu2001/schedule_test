import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session

from ..db import EnvConfig, get_db
from ..schemas import TrainingCreate
from ..services.sim_service import request_config
from ..services.train_service import (
    list_checkpoints, list_fs_runs, read_run_metrics, sync_results_to_db,
)

router = APIRouter()
training = APIRouter(prefix="/api/training")


@training.post("/runs", status_code=202)
def start_run(body: TrainingCreate, request: Request, db: Session = Depends(get_db)):
    saved = None
    if body.env_config_id is not None:
        row = db.get(EnvConfig, body.env_config_id)
        if row is None:
            raise HTTPException(404, "环境配置不存在")
        saved = row.config_json
    run = request.app.state.train_manager.start_run(
        request_config(body, saved), episodes=body.episodes, model_type=body.model_type,
        device=body.device, threads=body.threads, resume_path=body.resume_path,
        run_name=body.run_name, env_config_id=body.env_config_id,
    )
    return {"run_id": run["id"], **run}


@training.get("/runs")
def list_runs(request: Request, offset: int = Query(0, ge=0), limit: int = Query(100, gt=0)):
    return request.app.state.train_manager.list_runs(offset, limit)


@training.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    try:
        return request.app.state.train_manager.get_run(run_id)
    except KeyError as error:
        raise HTTPException(404, "训练任务不存在") from error


@training.get("/runs/{run_id}/metrics")
def run_metrics(run_id: str, request: Request):
    manager = request.app.state.train_manager
    try:
        return manager.metrics(run_id)
    except KeyError as error:
        raise HTTPException(404, "训练任务不存在") from error


@training.post("/runs/{run_id}/stop", status_code=202)
def stop_run(run_id: str, request: Request):
    try:
        return request.app.state.train_manager.stop_run(run_id)
    except KeyError as error:
        raise HTTPException(404, "进程内训练任务不存在") from error


@training.post("/sync-results")
def sync_results(request: Request, db: Session = Depends(get_db)):
    manager = request.app.state.train_manager
    with manager.lock:
        return sync_results_to_db(db, manager.results_dir)


@training.get("/checkpoints")
def checkpoints(request: Request):
    return list_checkpoints(request.app.state.train_manager.results_dir)


@training.get("/fs-runs")
def filesystem_runs(request: Request):
    return list_fs_runs(request.app.state.train_manager.results_dir)


@training.get("/fs-runs/{run_dir_name:path}/metrics")
def filesystem_run_metrics(run_dir_name: str, request: Request):
    run_dir = request.app.state.train_manager.results_dir / run_dir_name
    if not (run_dir / "train_config.json").is_file():
        raise HTTPException(404, "历史训练目录不存在")
    return read_run_metrics(run_dir)


@router.websocket("/ws/training/{run_id}")
async def training_stream(websocket: WebSocket, run_id: str):
    run = websocket.app.state.train_manager.runs.get(run_id)
    if run is None:
        await websocket.close(code=1008, reason="进程内训练任务不存在")
        return
    await websocket.accept()
    subscription = run.subscribe()
    queue = subscription[1]
    receive = asyncio.create_task(websocket.receive())
    get_event = asyncio.create_task(queue.get())
    try:
        while True:
            ready, _ = await asyncio.wait((receive, get_event), return_when=asyncio.FIRST_COMPLETED)
            if receive in ready:
                if receive.result()["type"] == "websocket.disconnect":
                    break
                receive = asyncio.create_task(websocket.receive())
            if get_event in ready:
                event = get_event.result()
                await websocket.send_json(jsonable_encoder(event))
                if event["type"] == "terminal":
                    await websocket.close()
                    break
                get_event = asyncio.create_task(queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        run.unsubscribe(subscription)
        receive.cancel()
        get_event.cancel()


router.include_router(training)
