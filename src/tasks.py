from dataclasses import dataclass
from graphlib import TopologicalSorter

import numpy as np

from .config import Config, NODE_TIERS
from .topology import node_positions, node_tier_roles


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


def _arrival_sources(rng, episode_slots, arrival_rate, burst_prob, burst_multiplier,
                     hotspot_fraction, hotspot_share, topology_period_slots, terminals):
    hotspot_count = int(np.ceil(hotspot_fraction * terminals))
    for slot in range(episode_slots):
        if slot % topology_period_slots == 0:
            hotspots = rng.choice(terminals, size=hotspot_count, replace=False)
        rate = arrival_rate * (burst_multiplier if rng.random() < burst_prob else 1.0)
        arrivals = max(1, int(rng.poisson(rate)))
        for _ in range(arrivals):
            source = int(rng.choice(hotspots) if rng.random() < hotspot_share
                         else rng.integers(terminals))
            yield slot, source


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
    for slot, source in _arrival_sources(
        rng, episode_slots, arrival_rate, burst_prob, burst_multiplier,
        hotspot_fraction, hotspot_share, topology_period_slots, terminals,
    ):
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


OP_CONTROL_PARSE = 0
OP_STREAM_FILTER_ROI = 1
OP_FFT_FIR = 2
OP_TENSOR_INFER = 3
OP_AGGREGATE_TRACK = 4
OP_NAMES = ("CONTROL_PARSE", "STREAM_FILTER_ROI", "FFT_FIR", "TENSOR_INFER", "AGGREGATE_TRACK")

CPU, NPU, FPGA = 0, 1, 2


@dataclass
class SubTask:
    dag_id: int
    subtask_id: int
    name: str
    op_type: int
    compatible_chips: tuple[int, ...]
    compute_gflop: float
    cpu_cores: int
    ram_gb: float
    npu_vram_gb: float
    fpga_lut: int
    fpga_ff: int
    fpga_bram: int
    fpga_dsp: int
    preds: tuple[int, ...]
    succs: tuple[int, ...]
    rank_u: float = 0.0
    est_s: float = 0.0
    lst_s: float = 0.0
    slack_s: float = 0.0


@dataclass
class TaskDAG:
    dag_id: int
    source: int
    access_edge: int
    kind: int
    input_data_mb: float
    output_mb: float
    compute_gflop: float
    deadline_s: float
    storage_gb: float
    priority: int
    slot: int
    subtasks: list[SubTask]
    edge_data_mb: dict[tuple[int, int], float]
    entry_ids: tuple[int, ...]
    exit_ids: tuple[int, ...]
    allow_offload: bool = True
    must_offload: bool = True


# 每行依次为名称、算子、兼容芯片、工作量百分比和入边（前驱编号、整图输入量比例）。
_DAG_TEMPLATES = (
    (  # 图像识别
        ("解码", OP_CONTROL_PARSE, (CPU,), 10, ()),
        ("左块增强", OP_STREAM_FILTER_ROI, (FPGA, CPU), 20, ((0, .50),)),
        ("右块增强", OP_STREAM_FILTER_ROI, (FPGA, CPU), 20, ((0, .50),)),
        ("分类推理", OP_TENSOR_INFER, (NPU, CPU), 50, ((1, .15), (2, .15))),
    ),
    (  # 目标检测
        ("解析", OP_CONTROL_PARSE, (CPU,), 8, ()),
        ("粗筛", OP_STREAM_FILTER_ROI, (FPGA, CPU), 12, ((0, .50),)),
        ("ROI提取", OP_STREAM_FILTER_ROI, (FPGA, CPU), 20, ((0, .50),)),
        ("检测推理", OP_TENSOR_INFER, (NPU, FPGA, CPU), 55, ((1, .08), (2, .16))),
        ("框合并", OP_AGGREGATE_TRACK, (CPU,), 5, ((3, .02),)),
    ),
    (  # 视频分析
        ("关键帧抽取", OP_CONTROL_PARSE, (CPU,), 10, ()),
        ("去噪", OP_STREAM_FILTER_ROI, (FPGA, CPU), 12, ((0, .30),)),
        ("运动区域", OP_STREAM_FILTER_ROI, (FPGA, CPU), 16, ((0, .30),)),
        ("ROI提取", OP_STREAM_FILTER_ROI, (FPGA, CPU), 20, ((0, .30),)),
        ("多路推理", OP_TENSOR_INFER, (NPU, FPGA, CPU), 36, ((1, .06), (2, .05), (3, .10))),
        ("跟踪汇总", OP_AGGREGATE_TRACK, (CPU,), 6, ((4, .02),)),
    ),
    (  # 语音处理
        ("分帧", OP_CONTROL_PARSE, (CPU,), 10, ()),
        ("FFT频谱", OP_FFT_FIR, (FPGA, CPU), 24, ((0, .50),)),
        ("FIR滤波", OP_FFT_FIR, (FPGA, CPU), 16, ((0, .50),)),
        ("声学推理", OP_TENSOR_INFER, (NPU, CPU), 42, ((1, .12), (2, .10))),
        ("解码汇总", OP_AGGREGATE_TRACK, (CPU,), 8, ((3, .02),)),
    ),
    (  # 信号计算
        ("协议解析", OP_CONTROL_PARSE, (CPU,), 15, ()),
        ("FFT频域分析", OP_FFT_FIR, (FPGA, CPU), 40, ((0, .50),)),
        ("FIR时域滤波", OP_FFT_FIR, (FPGA, CPU), 30, ((0, .50),)),
        ("状态聚合", OP_AGGREGATE_TRACK, (CPU,), 15, ((1, .10), (2, .08))),
    ),
)

# 每类算子的仿真需求：CPU 核数、共享 RAM(GB)、NPU 显存(GB)、FPGA LUT/FF/BRAM/DSP。
_OP_DEMANDS = (
    (1, .25, 0., 0, 0, 0, 0),
    (2, .50, 0., 12000, 24000, 32, 64),
    (2, .50, 0., 18000, 36000, 48, 128),
    (4, 1.0, 2., 36000, 72000, 96, 256),
    (1, .25, 0., 0, 0, 0, 0),
)


def _instantiate_template(dag_id, kind, compute_gflop, input_data_mb):
    template = _DAG_TEMPLATES[kind]
    subtasks = []
    edge_data_mb = {}
    successors = [[] for _ in template]
    for i, (name, op, chips, percent, incoming) in enumerate(template):
        cores, ram, vram, lut, ff, bram, dsp = _OP_DEMANDS[op]
        if FPGA not in chips:
            lut, ff, bram, dsp = 0, 0, 0, 0
        subtasks.append(SubTask(
            dag_id=dag_id, subtask_id=i, name=name, op_type=op, compatible_chips=chips,
            compute_gflop=compute_gflop * percent / 100, cpu_cores=cores, ram_gb=ram,
            npu_vram_gb=vram, fpga_lut=lut, fpga_ff=ff, fpga_bram=bram, fpga_dsp=dsp,
            preds=tuple(p for p, _ in incoming), succs=(),
        ))
        for p, fraction in incoming:
            successors[p].append(i)
            edge_data_mb[p, i] = input_data_mb * fraction
    for subtask, succs in zip(subtasks, successors):
        subtask.succs = tuple(succs)
    return subtasks, edge_data_mb


def _critical_path_work(subtasks):
    """模板按拓扑序存储，累计到各节点的最大路径工作量（GFLOP）。"""
    work = {}
    for task in subtasks:
        work[task.subtask_id] = task.compute_gflop + max((work[p] for p in task.preds), default=0.)
    return max(work.values())


def compute_dag_priorities(dag: TaskDAG, avg_chip_speeds, avg_link_rate_mbps, affinity_matrix):
    """用兼容芯片执行时间的算术平均值和依赖传输时间，就地更新结构优先级。"""
    nodes = {task.subtask_id: task for task in dag.subtasks}
    order = tuple(TopologicalSorter({i: task.preds for i, task in nodes.items()}).static_order())
    comm = {edge: 8 * mb / avg_link_rate_mbps for edge, mb in dag.edge_data_mb.items()}
    duration = {}
    for i in order:
        task = nodes[i]
        duration[i] = sum(task.compute_gflop / (avg_chip_speeds[u] * affinity_matrix[task.op_type][u])
                          for u in task.compatible_chips) / len(task.compatible_chips)
        task.est_s = max((nodes[p].est_s + duration[p] + comm[p, i] for p in task.preds), default=0.)

    horizon = max(nodes[i].est_s + duration[i] for i in dag.exit_ids)
    for i in reversed(order):
        task = nodes[i]
        task.rank_u = duration[i] + max((comm[i, j] + nodes[j].rank_u for j in task.succs), default=0.)
        latest_finish = min((nodes[j].lst_s - comm[i, j] for j in task.succs), default=horizon)
        task.lst_s = latest_finish - duration[i]
        task.slack_s = task.lst_s - task.est_s


def sample_dag_tasks(rng: np.random.Generator, cfg: Config, radio_rates: np.ndarray,
                     min_edge_compute_gflops: float | None = None,
                     access_edges: np.ndarray | None = None) -> list[TaskDAG]:
    """按场景生成整图到达流，时限采用源端实际速率与最长依赖路径工作量。"""
    tiers = [NODE_TIERS.index(role) for role in node_tier_roles(cfg.n_edges)]
    speeds = np.array([cfg.tier_cpu_gflops, cfg.tier_npu_gflops, cfg.tier_fpga_gflops])[:, tiers]
    avg_chip_speeds = speeds.mean(axis=1)
    if min_edge_compute_gflops is None:
        min_edge_compute_gflops = float(speeds.min())
    avg_rate = (cfg.link_rate_min_mbps + cfg.link_rate_max_mbps) / 2
    if access_edges is None:
        if cfg.terminal_access_mode == "nearest":
            positions, terminals = node_positions(cfg.n_edges, cfg.n_terminals)
            access_edges = ((terminals[:, None] - positions[:-1]) ** 2).sum(axis=2).argmin(axis=1)
        else:
            access_edges = np.arange(cfg.n_terminals) % cfg.n_edges

    arrivals = _arrival_sources(rng, cfg.episode_slots, cfg.arrival_rate, cfg.burst_prob,
                                cfg.burst_multiplier, cfg.hotspot_fraction, cfg.hotspot_share,
                                cfg.topology_period_slots, cfg.n_terminals)
    dags = []
    for dag_id, (slot, source) in enumerate(arrivals):
        kind = int(rng.integers(len(_DAG_TEMPLATES)))
        input_mb = float(rng.uniform(0.5, 8.0))
        output_mb = float(rng.uniform(0.05, 1.0))
        work = float(rng.uniform(0.5, 10.0))
        subtasks, edge_data = _instantiate_template(dag_id, kind, work, input_mb)
        radio_time = 8 * (input_mb + output_mb) / radio_rates[source]
        compute_time = _critical_path_work(subtasks) / min_edge_compute_gflops
        slack = rng.uniform(cfg.deadline_slack_min, cfg.deadline_slack_max)
        dag = TaskDAG(
            dag_id=dag_id, source=source, access_edge=int(access_edges[source]), kind=kind,
            input_data_mb=input_mb, output_mb=output_mb, compute_gflop=work,
            deadline_s=float(slack * (radio_time + compute_time)), storage_gb=float(rng.uniform(.1, 2.)),
            priority=int(rng.integers(1, 6)), slot=slot, subtasks=subtasks, edge_data_mb=edge_data,
            entry_ids=(0,), exit_ids=(len(subtasks) - 1,),
        )
        compute_dag_priorities(dag, avg_chip_speeds, avg_rate, cfg.operator_affinity_matrix)
        dags.append(dag)
    return dags
