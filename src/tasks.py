from dataclasses import dataclass
import numpy as np


@dataclass
class Task:
    source: int
    access_edge: int
    kind: int
    data_mb: float
    output_mb: float
    compute_gflop: float
    deadline_s: float
    memory_gb: float
    storage_gb: float
    priority: int
    allow_offload: bool


def sample_tasks(rng: np.random.Generator, n: int, terminals: int, edges: int) -> list[Task]:
    tasks = []
    for _ in range(n):
        source = int(rng.integers(terminals))
        tasks.append(Task(
            source=source,
            access_edge=source % edges,
            kind=int(rng.integers(5)),
            data_mb=float(rng.uniform(0.5, 8.0)),
            output_mb=float(rng.uniform(0.05, 1.0)),
            compute_gflop=float(rng.uniform(0.5, 10.0)),
            deadline_s=float(rng.uniform(0.05, 0.5)),
            memory_gb=float(rng.uniform(0.2, 2.0)),
            storage_gb=float(rng.uniform(0.1, 2.0)),
            priority=int(rng.integers(1, 6)),
            allow_offload=bool(rng.random() < 0.93),
        ))
    return tasks
