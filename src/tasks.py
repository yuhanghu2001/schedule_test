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
    must_offload: bool


def sample_tasks(
    rng: np.random.Generator,
    n: int,
    terminals: int,
    edges: int,
    radio_rates: np.ndarray,
    min_edge_compute_gflops: float,
    deadline_slack_min: float,
    deadline_slack_max: float,
    must_offload_rate: float,
) -> list[Task]:
    tasks = []
    for _ in range(n):
        source = int(rng.integers(terminals))
        data_mb = float(rng.uniform(0.5, 8.0))
        output_mb = float(rng.uniform(0.05, 1.0))
        compute_gflop = float(rng.uniform(0.5, 10.0))
        radio_rate = radio_rates[source]
        min_uplink = data_mb * 8 / radio_rate
        min_download = output_mb * 8 / radio_rate
        min_compute = compute_gflop / min_edge_compute_gflops
        deadline_s = (min_uplink + min_download + min_compute) * rng.uniform(
            deadline_slack_min, deadline_slack_max,
        )
        offload_draw = rng.random()
        must_offload = bool(offload_draw < must_offload_rate)
        allow_offload = bool(offload_draw < 0.93) or must_offload
        tasks.append(Task(
            source=source,
            access_edge=source % edges,
            kind=int(rng.integers(5)),
            data_mb=data_mb,
            output_mb=output_mb,
            compute_gflop=compute_gflop,
            deadline_s=float(deadline_s),
            memory_gb=float(rng.uniform(0.2, 2.0)),
            storage_gb=float(rng.uniform(0.1, 2.0)),
            priority=int(rng.integers(1, 6)),
            allow_offload=allow_offload,
            must_offload=must_offload,
        ))
    return tasks
