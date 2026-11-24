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
    cloud_gateway_count: int = 0  # 0 保留旧配置全直连；三套预设使用前 4 个边缘作网关
    terminal_access_mode: str = "cyclic"  # 旧配置按编号取模；当前预设使用示意位置最近邻
    edge_topology: str = "ring"  # 旧配置保留环和随机弦；当前预设使用八邻接网格
    max_hops: int = 4
    k_routes_per_destination: int = 3
    link_flap_prob: float = 0.20
    congestion_ar_coef: float = 0.65
    congestion_step_noise: float = 0.15
    link_spike_prob: float = 0.25
    link_spike_min: float = 0.15
    link_spike_max: float = 0.35
    link_rate_fluctuation: float = 0.25
    slot_seconds: float = 1.0
    congestion_min: float = 0.1
    congestion_max: float = 0.6
    congestion_noise: float = 0.15
    congestion_realized_max: float = 0.95
    link_rate_min_mbps: float = 200.0
    link_rate_max_mbps: float = 500.0
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

    # 节点管理：空元组使用原场景默认值，计算节点按边缘后接云端排列
    terminal_radio_rates_mbps: tuple[float, ...] = ()
    node_memory_gb: tuple[float, ...] = ()
    node_storage_gb: tuple[float, ...] = ()
    node_queue_capacity: tuple[int, ...] = ()

    # 任务时限与卸载
    deadline_slack_min: float = 2.2
    deadline_slack_max: float = 3.5

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
    reward_penalty_conditional_reject: float = 2.5
    reward_delay_ref: float = 1.0
    reward_energy_ref: float = 3.0
    # 随机非拒绝策略（种子 1042–1046）实测 D_ref ≈ 0.00445229。
    reward_variance_ref: float = 0.0045
    reward_version: str = "v8_dynamic_bottleneck"
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
    ppo_lr_end_factor: float = 0.1  # 学习率退火终值占初值的比例
    ppo_ent_coef_end: float = 0.001  # 熵系数退火终值
    max_grad_norm: float = 0.5
    eval_interval: int = 25
    evaluation_seed_count: int = 5

    # DQN 回放、目标网络和探索
    dqn_learning_rate: float = 1e-4
    dqn_buffer_size: int = 50_000
    dqn_batch_size: int = 64
    dqn_train_every: int = 4  # 环境步
    dqn_target_sync: int = 1_000  # 硬同步间隔（环境步）
    dqn_warmup_steps: int = 2_000
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_end: float = 0.05
    dqn_epsilon_decay_fraction: float = 0.6  # 训练回合占比
    dqn_double: bool = False

    # DDPG（Wolpertinger 离散映射）回放、软更新和探索
    ddpg_actor_learning_rate: float = 1e-4
    ddpg_critic_learning_rate: float = 1e-3
    ddpg_tau: float = 0.005
    ddpg_buffer_size: int = 50_000
    ddpg_batch_size: int = 64
    ddpg_train_every: int = 4  # 环境步
    ddpg_warmup_steps: int = 2_000
    ddpg_noise_start: float = 0.3
    ddpg_noise_end: float = 0.05
    ddpg_noise_decay_fraction: float = 0.6  # 训练回合占比
    # 候选特征统一界限，第 7、8 列为环境缩放后的时延、能耗
    ddpg_candidate_low: tuple[float, ...] = (0.0,) * 10
    ddpg_candidate_high: tuple[float, ...] = (1., 1., 1., 1., 1., 1., 1., 5., 10., 1.)

    # 时隙内元启发式搜索，允许的偏好秩为 0 至 max_rank，另设拒绝
    ga_population: int = 16
    ga_generations: int = 10
    ga_crossover: float = 0.8
    ga_mutation: float = 0.1
    ga_elite: int = 2
    ga_tournament_size: int = 3
    ga_max_rank: int = 3
    aco_ants: int = 16
    aco_iterations: int = 10
    aco_alpha: float = 1.0
    aco_beta: float = 2.0
    aco_evaporation: float = 0.3
    aco_max_rank: int = 3


def load_config(name: str = "default") -> Config:
    """返回 Python 场景配置；预设启用八邻接、网关上云和最近邻接入。"""
    cfg = replace(Config(), cloud_gateway_count=4, terminal_access_mode="nearest",
                  edge_topology="eight_neighbor", undirected_edge_links=29)
    if name == "default":
        return cfg
    if name == "high_congestion":
        return replace(
            cfg, topology_period_slots=4,
            congestion_min=0.45, congestion_max=0.8, arrival_rate=24.0,
            link_flap_prob=0.25, link_spike_prob=0.35, link_rate_fluctuation=0.35,
        )
    if name == "small_smoke":
        return replace(
            cfg, episode_slots=4, arrival_rate=5.0,
            topology_period_slots=2, ppo_batch_size=16,
        )
    raise ValueError(f"未知配置名称: {name}")
