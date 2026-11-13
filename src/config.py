from dataclasses import dataclass
import json


@dataclass
class Config:
    seed: int = 42
    evaluation_seed: int = 1042
    n_terminals: int = 20
    n_edges: int = 6
    n_tasks: int = 200
    tasks_per_slot: int = 12
    topology_period_slots: int = 25
    undirected_edge_links: int = 9
    max_hops: int = 4
    k_routes_per_destination: int = 2
    slot_seconds: float = 1.0
    congestion_min: float = 0.1
    congestion_max: float = 0.6
    link_rate_min_mbps: float = 500.0
    link_rate_max_mbps: float = 2000.0
    cloud_rate_min_mbps: float = 50.0
    cloud_rate_max_mbps: float = 200.0
    radio_rate_min_mbps: float = 60.0
    radio_rate_max_mbps: float = 180.0
    service_deployment: tuple[tuple[int, ...], ...] = (
        (1, 1, 1, 0, 0),
        (0, 0, 1, 1, 1),
        (1, 0, 0, 1, 0),
        (1, 1, 1, 0, 0),
        (0, 0, 1, 1, 1),
        (0, 1, 0, 0, 1),
    )
    edge_speed_matrix: tuple[tuple[float, float, float], ...] = (
        (150.0, 500.0, 150.0),
        (150.0, 150.0, 500.0),
        (400.0, 200.0, 200.0),
        (150.0, 500.0, 150.0),
        (150.0, 150.0, 500.0),
        (400.0, 200.0, 200.0),
    )
    cloud_speed: tuple[float, float, float] = (8000.0, 12000.0, 8000.0)
    local_speed: tuple[float, float, float] = (12.0, 16.0, 6.0)
    deadline_slack_min: float = 1.8
    deadline_slack_max: float = 3.0
    must_offload_rate: float = 0.6
    radio_bandwidth_mhz: float = 20.0
    uplink_max_power_w: float = 0.5
    uplink_circuit_power_w: float = 0.1
    dinkelbach_tolerance: float = 1e-3
    dinkelbach_max_iterations: int = 4
    compute_priority_exponent: float = 1.0
    reward_beta_completion: float = 2.0
    reward_beta_delay: float = 1.0
    reward_beta_energy: float = 1.0
    reward_beta_load: float = 1.0
    reward_penalty_incomplete: float = 1.0
    reward_penalty_invalid: float = 1.0
    reward_penalty_conditional_reject: float = 1.0
    reward_delay_ref: float = 1.0
    reward_energy_ref: float = 3.0
    reward_variance_ref: float = 0.05
    reward_version: str = "v4_paper_formula_2_1_1_1"
    observation_version: str = "v2_global9"
    gat_hidden_dim: int = 256
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 5
    ppo_batch_size: int = 64
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    eval_interval: int = 25
    evaluation_seed_count: int = 5


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(**json.load(f))
