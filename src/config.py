from dataclasses import dataclass, replace


@dataclass
class Config:
    # 随机种子与任务规模
    seed: int = 42
    evaluation_seed: int = 1042
    n_terminals: int = 24
    n_edges: int = 12
    episode_slots: int = 20
    arrival_rate: float = 18.0
    burst_prob: float = 0.15
    burst_multiplier: float = 2.5
    hotspot_fraction: float = 0.3
    hotspot_share: float = 0.6

    # 动态拓扑与链路
    topology_period_slots: int = 5
    undirected_edge_links: int = 18
    max_hops: int = 4
    k_routes_per_destination: int = 3
    link_flap_prob: float = 0.12
    congestion_ar_coef: float = 0.65
    congestion_step_noise: float = 0.15
    link_spike_prob: float = 0.15
    link_spike_min: float = 0.15
    link_spike_max: float = 0.35
    link_rate_fluctuation: float = 0.25
    slot_seconds: float = 1.0
    congestion_min: float = 0.1
    congestion_max: float = 0.6
    congestion_noise: float = 0.15
    congestion_realized_max: float = 0.95
    link_rate_min_mbps: float = 500.0
    link_rate_max_mbps: float = 2000.0
    cloud_rate_min_mbps: float = 50.0
    cloud_rate_max_mbps: float = 200.0
    radio_rate_min_mbps: float = 60.0
    radio_rate_max_mbps: float = 180.0

    # 服务部署与异构计算资源
    service_deployment: tuple[tuple[int, ...], ...] = (
        (1, 1, 0, 0, 0),
        (0, 0, 1, 1, 0),
        (0, 0, 0, 1, 1),
        (1, 0, 0, 0, 0),
        (1, 1, 0, 0, 0),
        (0, 0, 1, 0, 1),
        (0, 1, 0, 1, 0),
        (0, 0, 1, 0, 0),
        (1, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 1, 0, 0, 1),
        (0, 0, 0, 0, 1),
    )
    edge_speed_matrix: tuple[tuple[float, float, float], ...] = (
        (150.0, 520.0, 150.0),
        (150.0, 150.0, 520.0),
        (420.0, 180.0, 180.0),
        (120.0, 120.0, 120.0),
    ) * 3
    cloud_speed: tuple[float, float, float] = (8000.0, 12000.0, 8000.0)

    # 任务时限与卸载
    deadline_slack_min: float = 1.8
    deadline_slack_max: float = 3.0

    # 连续优化子问题
    radio_bandwidth_mhz: float = 20.0
    uplink_max_power_w: float = 0.5
    uplink_circuit_power_w: float = 0.1
    dinkelbach_tolerance: float = 1e-3
    dinkelbach_max_iterations: int = 4
    compute_priority_exponent: float = 1.0
    compute_degradation_max: float = 0.2

    # 论文奖励公式：beta_1*delta-y*(beta_2*t+beta_3*E)-beta_4*D_load-Omega
    reward_beta_completion: float = 2.0
    reward_beta_delay: float = 1.0
    reward_beta_energy: float = 1.0
    reward_beta_load: float = 1.0
    reward_penalty_incomplete: float = 1.0
    reward_penalty_invalid: float = 1.0
    reward_penalty_conditional_reject: float = 1.0
    reward_delay_ref: float = 1.0
    reward_energy_ref: float = 3.0
    # 随机非拒绝策略(1042—1046)：D̄_rand=0.08608214，t̄_rand=0.82071578s。
    reward_variance_ref: float = 0.0058
    reward_version: str = "v6_dynamic_edge_only"
    observation_version: str = "v4_edge_only_4d_edges"

    # GAT-PPO 网络与训练
    gat_hidden_dim: int = 256
    actor_hidden_dim: int = 128
    critic_hidden_dim: int = 128
    actor_learning_rate: float = 3e-4
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


def load_config(name: str = "default") -> Config:
    """返回Python场景配置；所有完整默认值都在Config中。"""
    if name == "default":
        return Config()
    if name == "high_congestion":
        return replace(
            Config(), topology_period_slots=4, undirected_edge_links=16,
            congestion_min=0.45, congestion_max=0.8, arrival_rate=24.0,
            link_flap_prob=0.18, link_spike_prob=0.25, link_rate_fluctuation=0.35,
        )
    if name == "small_smoke":
        return replace(
            Config(), episode_slots=4, arrival_rate=5.0,
            topology_period_slots=2, ppo_batch_size=16,
        )
    raise ValueError(f"未知配置名称: {name}")
