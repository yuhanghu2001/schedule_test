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
    allow_offload: bool = True
    must_offload: bool = True
    slot: int = 0


def sample_tasks(
    rng: np.random.Generator,
    episode_slots: int,
    arrival_rate: float,
    burst_prob: float,
    burst_multiplier: float,
    hotspot_fraction: float,
    hotspot_share: float,
    topology_period_slots: int,
    terminals: int,
    edges: int,
    radio_rates: np.ndarray,
    min_edge_compute_gflops: float,
    deadline_slack_min: float,
    deadline_slack_max: float,
    access_edges: np.ndarray | None = None,
) -> list[Task]:
    tasks = []
    hotspot_count = int(np.ceil(hotspot_fraction * terminals))
    for slot in range(episode_slots):
        if slot % topology_period_slots == 0:
            hotspots = rng.choice(terminals, size=hotspot_count, replace=False)
        rate = arrival_rate * (burst_multiplier if rng.random() < burst_prob else 1.0)
        arrivals = max(1, int(rng.poisson(rate)))
        for _ in range(arrivals):
            source = int(rng.choice(hotspots) if rng.random() < hotspot_share
                         else rng.integers(terminals))
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
            tasks.append(Task(
                source=source,
                access_edge=source % edges if access_edges is None else int(access_edges[source]),
                kind=int(rng.integers(5)),
                data_mb=data_mb,
                output_mb=output_mb,
                compute_gflop=compute_gflop,
                deadline_s=float(deadline_s),
                memory_gb=float(rng.uniform(0.2, 2.0)),
                storage_gb=float(rng.uniform(0.1, 2.0)),
                priority=int(rng.integers(1, 6)),
                slot=slot,
            ))
    return tasks
