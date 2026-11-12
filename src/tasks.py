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


def sample_tasks(
    rng: np.random.Generator,
    n: int,
    terminals: int,
    edges: int,
    max_uplink_rate_mbps: float,
    min_edge_compute_gflops: float,
    deadline_slack_min: float,
    deadline_slack_max: float,
    deadline_ceiling_s: float,
) -> list[Task]:
    tasks = []
    for _ in range(n):
        source = int(rng.integers(terminals))
        data_mb = float(rng.uniform(0.5, 8.0))
        compute_gflop = float(rng.uniform(0.5, 10.0))
        min_uplink = data_mb * 8 / max_uplink_rate_mbps
        min_compute = compute_gflop / min_edge_compute_gflops
        deadline_s = min(
            (min_uplink + min_compute) * rng.uniform(
                deadline_slack_min, deadline_slack_max,
            ),
            deadline_ceiling_s,
        )
        tasks.append(Task(
            source=source,
            access_edge=source % edges,
            kind=int(rng.integers(5)),
            data_mb=data_mb,
            output_mb=float(rng.uniform(0.05, 1.0)),
            compute_gflop=compute_gflop,
            deadline_s=float(deadline_s),
            memory_gb=float(rng.uniform(0.2, 2.0)),
            storage_gb=float(rng.uniform(0.1, 2.0)),
            priority=int(rng.integers(1, 6)),
            allow_offload=bool(rng.random() < 0.93),
        ))
    return tasks
