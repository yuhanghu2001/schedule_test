from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from .db import SessionLocal, engine
from .routers import env, experiments, training
from .schemas import HealthResponse
from .services.sim_service import SimManager
from .services.train_service import TrainManager


@asynccontextmanager
async def lifespan(app):
    manager = app.state.train_manager
    yield
    for run in manager.runs.values():
        run.stop_event.set()
    manager.executor.shutdown()


app = FastAPI(title="云边端调度系统", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|\[::1\]):(517[3-9]|4173)",
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.state.sim_manager = SimManager()
app.state.train_manager = TrainManager(SessionLocal)


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        db_connected = True
    except SQLAlchemyError:
        db_connected = False
    return {
        "status": "ok",
        "db_connected": db_connected,
        "results_dir": str(request.app.state.train_manager.results_dir),
        "active_training_runs": request.app.state.train_manager.active_count(),
        "active_sim_sessions": len(request.app.state.sim_manager.sessions),
    }


app.include_router(env.router)
app.include_router(training.router)
app.include_router(experiments.router)
