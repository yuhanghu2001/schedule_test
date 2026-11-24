from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch

from src.baselines import baseline_action
from src.config import Config, load_config
from src.environment import EdgeEnv, ShuffledGraphEnv, shuffle_graph_observation
from src.models.gat import GATEncoder
from src.models.policy import DAGShuffledGraphEnv, GATPPO, MODEL_NAMES
from src.resources import DEMAND_FIELDS
from src.tasks import NPU, FPGA
from src.topology import _activate_link, edge_pairs, node_tier_roles

from ..config import REPO_ROOT

TASK_KIND_NAMES = ("图像识别", "目标检测", "视频分析", "语音处理", "信号计算")
RESOURCE_NAMES = ("CPU", "NPU", "FPGA")


def config_from_dict(data):
    values = dict(data)
    for key in ("service_deployment", "edge_speed_matrix"):
        if key in values:
            values[key] = tuple(map(tuple, values[key]))
    for key in ("cloud_speed", "terminal_radio_rates_mbps", "node_memory_gb",
                "node_storage_gb", "node_queue_capacity"):
        if key in values:
            values[key] = tuple(values[key])
    if "n_edges" in values and "undirected_edge_links" not in values:
        values["undirected_edge_links"] = max(Config.undirected_edge_links, values["n_edges"])
    return Config(**values)


def request_config(request, saved=None):
    values = asdict(load_config(request.preset_name)) if saved is None else dict(saved)
    overrides = {**request.model_extra, **request.config}
    values.update(overrides)
    if "n_edges" in overrides and "undirected_edge_links" not in overrides:
        values["undirected_edge_links"] = max(values["undirected_edge_links"], values["n_edges"])
    if request.seed is not None:
        values["seed"] = request.seed
    return config_from_dict(values)


def load_checkpoint(path):
    return torch.load(REPO_ROOT / Path(path), map_location="cpu", weights_only=True)


def checkpoint_metadata(checkpoint):
    return {
        "model_type": checkpoint["model_type"],
        **{key: checkpoint.get(key, checkpoint["config"].get(key, getattr(Config, key)))
           for key in ("observation_version", "reward_version", "enable_dag_mode")},
    }


def checkpoint_config(checkpoint):
    values = {key: value for key, value in checkpoint["config"].items()
              if key in Config.__dataclass_fields__}
    metadata = checkpoint_metadata(checkpoint)
    del metadata["model_type"]
    values.update(metadata)
    state = checkpoint["model"]
    encoder = "flat_encoder.0.weight" if values["enable_dag_mode"] and (
        MODEL_NAMES[checkpoint["model_type"]] == "ppo"
    ) else "gat.layer1.transform.weight"
    values.update(
        gat_hidden_dim=state[encoder].shape[0],
        actor_hidden_dim=state["actor.0.weight"].shape[0],
        critic_hidden_dim=state["critic.0.weight"].shape[0],
    )
    return config_from_dict(values)


def build_model(cfg, model_type, checkpoint=None, *, inference=False):
    model_type = MODEL_NAMES[model_type]
    device = "meta" if inference else "cpu"
    with torch.device(device):
        model = GATPPO.from_config(cfg, model_type)
        if model.dag_mode != cfg.enable_dag_mode:
            raise ValueError(f"{model_type} 与环境的 observation_version / enable_dag_mode 不匹配")
        if checkpoint is not None and not model.dag_mode:
            state = checkpoint["model"]
            key = "gat.layer1.attention.weight"
            edge_dim = state[key].shape[1] - 2 * cfg.gat_hidden_dim
            if model.state_dict()[key].shape != state[key].shape:
                model.gat = GATEncoder(hidden=cfg.gat_hidden_dim, edge_dim=edge_dim)
                model.gat.register_forward_pre_hook(
                    lambda _, args: (args[0], args[1], args[2][..., :edge_dim])
                )
            if "gat.skip.weight" not in state:
                model.gat.layer1.norm = model.gat.layer2.norm = model.gat.skip = None
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"], assign=inference)
    return model.eval() if inference else model


def load_model_for_environment(path, cfg, method):
    checkpoint = load_checkpoint(path)
    saved = checkpoint_config(checkpoint)
    model_type = MODEL_NAMES[checkpoint["model_type"]]
    legacy_adapter = (saved.observation_version in ("v2_global9", "v3_slot_arrivals_carryover")
                      and cfg.observation_version == Config.observation_version)
    if (saved.enable_dag_mode != cfg.enable_dag_mode
            or (saved.observation_version != cfg.observation_version and not legacy_adapter)):
        raise ValueError(
            f"checkpoint observation_version={saved.observation_version!r}, "
            f"enable_dag_mode={saved.enable_dag_mode} 与目标环境 "
            f"observation_version={cfg.observation_version!r}, "
            f"enable_dag_mode={cfg.enable_dag_mode} 不匹配"
        )
    if method != "checkpoint" and MODEL_NAMES[method] != model_type:
        raise ValueError(f"{method} 策略与 checkpoint 的模型类型 {model_type} 不匹配")
    return build_model(saved, model_type, checkpoint, inference=True), model_type


def node_label(env, node):
    if node == env.cloud:
        return "云端"
    if node == -2:
        return "拒绝"
    return f"边缘 {node}"


def action_fields(env, action):
    return {
        "destination": action.destination,
        "destination_label": node_label(env, action.destination),
        "resource": action.resource,
        "resource_name": RESOURCE_NAMES[action.resource] if action.resource >= 0 else None,
        "path": list(action.path),
        "compute_share": float(action.compute_share),
        "uplink_power": float(action.uplink_power),
        "estimated_delay": float(action.estimated_delay),
        "estimated_energy": float(action.estimated_energy),
        "delay": float(action.delay),
        "energy": float(action.energy),
    }


def node_snapshot(env, node, active_task_ids):
    cloud = node == env.cloud
    speed = env.speed[node]
    profile = "balanced" if np.all(speed == speed[0]) else RESOURCE_NAMES[int(speed.argmax())]
    return {
        "id": node,
        "label": node_label(env, node),
        "kind": "cloud" if cloud else "edge",
        "position": env.node_positions[node].tolist(),
        "profile": "cloud" if cloud else profile,
        "speed": speed.tolist(),
        "resource_used": env.slot_resource_used[node].tolist(),
        "carryover_used": env.carryover_resource_used[node].tolist(),
        "mem_used": float(env.slot_mem_used[node]),
        "mem_cap": float(env.mem_cap[node]),
        "storage_used": float(env.slot_storage_used[node]),
        "storage_cap": float(env.storage_cap[node]),
        "queue_used": int(env.slot_queues[node]),
        "queue_cap": int(env.queue_cap[node]),
        "services": list(range(len(TASK_KIND_NAMES))) if cloud else
        np.flatnonzero(env.service_deployment[node]).tolist(),
        "active_task_ids": active_task_ids,
    }


def dag_node_snapshot(env, node, tier, active_task_ids):
    pool = env.resources
    return {
        "id": node,
        "label": node_label(env, node),
        "kind": "cloud" if node == env.cloud else "edge",
        "position": env.node_positions[node].tolist(),
        "profile": "cloud" if node == env.cloud else RESOURCE_NAMES[int(pool.speed[node].argmax())],
        "tier": tier,
        "speed": pool.speed[node].tolist(),
        "resource_used": pool.compute_used[node].tolist(),
        "compute_free": pool.free_compute[node].tolist(),
        "mem_used": float(pool.capacity["ram_gb"][node] - pool.free["ram_gb"][node]),
        "mem_cap": float(pool.capacity["ram_gb"][node]),
        "storage_used": float(pool.capacity["storage_gb"][node] - pool.free["storage_gb"][node]),
        "storage_cap": float(pool.capacity["storage_gb"][node]),
        "queue_used": int(pool.queues[node].sum()),
        "queue_cap": int(pool.queue_cap[node]),
        "services": np.flatnonzero(env.z_service[node]).tolist(),
        "active_task_ids": active_task_ids,
        "npu_resident_op": int(pool.resident_op[node, NPU]),
        "fpga_resident_op": int(pool.resident_op[node, FPGA]),
        **{f"{key}_{suffix}": float(values[key][node])
           for key in DEMAND_FIELDS
           for suffix, values in (("cap", pool.capacity), ("free", pool.free))},
    }


def link_snapshot(env, source, target, status):
    return {
        "source": source,
        "target": target,
        "kind": "cloud" if target == env.cloud else "edge",
        "rate_mbps": float(env.rates[source, target]),
        "base_rate_mbps": float(env.base_rates[source, target]),
        "background_fwd": float(env.background[source, target]),
        "background_rev": float(env.background[target, source]),
        "congestion_delta_fwd": float(env.congestion_delta[source, target]),
        "congestion_delta_rev": float(env.congestion_delta[target, source]),
        "used_fwd_mb": float(env.slot_link_used[source, target]),
        "used_rev_mb": float(env.slot_link_used[target, source]),
        "capacity_fwd_mb": float(env._slot_capacity_mb(source, target)),
        "capacity_rev_mb": float(env._slot_capacity_mb(target, source)),
        "prop_ms": float(env.prop[source, target]) * 1000,
        "status": status,
    }


def task_fields(task, index):
    fields = asdict(task)
    del fields["allow_offload"]
    del fields["must_offload"]
    return {**fields, "task_index": index, "kind_name": TASK_KIND_NAMES[task.kind]}


def dag_task_fields(env):
    dag, subtask = env.current_dag, env.current_subtask
    return {
        **asdict(subtask),
        "task_index": env.idx,
        "slot": env.slot,
        "source": dag.source,
        "access_edge": dag.access_edge,
        "kind": dag.kind,
        "kind_name": TASK_KIND_NAMES[dag.kind],
        "data_mb": dag.input_data_mb,
        "output_mb": dag.output_mb,
        "deadline_s": dag.deadline_s,
        "priority": dag.priority,
    }


def dag_status(state, slot):
    if state.terminal_time is not None:
        return "completed" if state.success else "failed"
    return "active" if state.dag.slot <= slot else "waiting"


def dag_trace_record(env, record):
    state = env.dag_states[record["dag_id"]]
    subtask_id = record["subtask_id"]
    status = state.status[subtask_id]
    assignment = state.assignments.get(subtask_id)
    if assignment is not None:
        action = assignment.action
        record = {
            **record,
            **action_fields(env, action),
            "start_time_s": float(action.start_time_s),
            "finish_time_s": float(action.finish_time_s),
            "finish_time": float(action.finish_time_s),
            "setup_delay_s": float(action.setup_delay_s),
            "setup_energy_j": float(action.setup_energy_j),
            "comm_tier": action.comm_tier,
            "flows": [{"pred_id": flow.pred_id, "path": list(flow.path),
                       "comm_tier": flow.comm_tier, "data_mb": float(flow.data_mb)}
                      for flow in action.flows],
        }
    return {**record, "status": status if assignment is not None else "rejected",
            "completed": status == "completed", "dag_status": dag_status(state, env.slot)}


def dag_snapshot_fields(env, history, done):
    current_task = None
    if not done:
        state = env.dag_states[env.current_dag.dag_id]
        current_task = {
            **dag_task_fields(env),
            "subtasks": [{**asdict(task), "status": state.status[task.subtask_id]}
                         for task in state.dag.subtasks],
            "edges": [{"src": u, "dst": v, "data_mb": float(mb)}
                      for (u, v), mb in state.dag.edge_data_mb.items()],
            "feasible_action_count": int(env.action_mask[:-1].sum()),
            "candidates": [
                {"action_id": i, "valid": bool(valid), **action_fields(env, action)}
                for i, (action, valid) in enumerate(zip(env.actions, env.action_mask))
            ],
        }

    active_task_ids = [[] for _ in range(env.n_graph_nodes)]
    for record in history:
        key = (record["dag_id"], record["subtask_id"])
        if key in env.active_reservations:
            active_task_ids[env.active_reservations[key].node].append(record["task_index"])
    return {
        "enable_dag_mode": True,
        "observation_version": env.cfg.observation_version,
        "reward_version": env.cfg.reward_version,
        "total_tasks": sum(len(dag.subtasks) for dag in env.dags),
        "nodes": [dag_node_snapshot(env, node, tier, active_task_ids[node])
                  for node, tier in enumerate((*node_tier_roles(env.cloud), "cloud"))],
        "current_task": current_task,
        "recent_tasks": [dag_trace_record(env, record) for record in history],
        "metrics": env.metrics(),
        "dags": [
            {"dag_id": state.dag.dag_id, "status": dag_status(state, env.slot),
             "finish_time_s": state.terminal_time}
            for state in env.dag_states.values()
        ],
    }


def legacy_snapshot_fields(env, history, done):
    current_task = None
    if not done:
        current_task = {
            **task_fields(env.tasks[env.idx], env.idx),
            "feasible_action_count": len(env.actions) - 1,
            "candidates": [
                {"action_id": number, **action_fields(env, action)}
                for number, action in enumerate(env.actions)
            ],
        }

    active_task_ids = [[] for _ in range(env.n_graph_nodes)]
    for assignment in env.active_assignments:
        active_task_ids[assignment.action.destination].append(assignment.task_index)

    # 同一时隙的后续分配可以更新已接收任务的算力、时延和能耗。
    assignments = {assignment.task_index: assignment for assignment in env.assignments}
    recent_tasks = []
    for record in history:
        assignment = assignments.get(record["task_index"])
        if assignment is not None:
            record = {
                **record,
                **action_fields(env, assignment.action),
                "completed": bool(assignment.action.delay <= record["deadline_s"]),
                "finish_time": float(assignment.finish_time),
            }
        recent_tasks.append(record)

    # 核心用任务总数和已结算时隙数做分母。推演只统计已处理任务。
    settled = copy(env)
    settled.tasks = env.tasks[:max(1, env.idx)]
    settled.load_samples = max(1, env.load_samples)
    metrics = {key: float(value) if isinstance(value, np.generic) else value
               for key, value in settled.metrics().items()}
    metrics["tasks"] = env.idx

    return {
        "total_tasks": len(env.tasks),
        "nodes": [node_snapshot(env, node, active_task_ids[node])
                  for node in range(env.n_graph_nodes)],
        "current_task": current_task,
        "recent_tasks": recent_tasks,
        "metrics": metrics,
    }


def serialize_env_snapshot(env, history, last_decision=None, *, done=False):
    snapshot_fields = dag_snapshot_fields if env.cfg.enable_dag_mode else legacy_snapshot_fields
    links = [
        link_snapshot(env, int(source), int(target), "active")
        for source, target in zip(*np.nonzero(np.triu(env.adj, k=1)))
    ]
    links.extend(link_snapshot(env, source, target, "flapped")
                 for source, target in env.flapped_edges)
    return {
        "slot": env.slot,
        "episode_slots": env.cfg.episode_slots,
        "task_index": env.idx,
        "done": done,
        "topology_period_slots": env.cfg.topology_period_slots,
        **snapshot_fields(env, history, done),
        "terminals": [
            {"id": terminal, "label": f"端侧 {terminal}",
             "access_edge": int(env.access_edges[terminal]),
             "position": env.terminal_positions[terminal].tolist(),
             "radio_rate_mbps": float(rate)}
            for terminal, rate in enumerate(env.radio_rates)
        ],
        "links": links,
        "last_decision": last_decision,
    }


@dataclass
class SimSession:
    env: EdgeEnv
    policy: str
    model: GATPPO | None = None
    done: bool = False
    history: list = field(default_factory=list)
    last_decision: dict | None = None
    suspended_links: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self.reset_policy()

    def reset_policy(self):
        self.rng = np.random.default_rng(self.env.cfg.seed)
        self.policy_env = (self.model.rollout_environment(self.env, self.env.cfg.seed)
                           if self.model is not None else self.env)

    def snapshot(self):
        return serialize_env_snapshot(self.env, self.history, self.last_decision, done=self.done)

    def choose_action(self):
        if self.policy == "random" and not self.env.cfg.enable_dag_mode:
            return int(self.rng.integers(len(self.env.actions)))
        if self.model is None:
            return baseline_action(self.policy, self.env)
        obs = self.env.observe()
        if isinstance(self.policy_env, ShuffledGraphEnv):
            obs = shuffle_graph_observation(obs, self.policy_env.permutation)
            if isinstance(self.policy_env, DAGShuffledGraphEnv):
                obs = self.policy_env._align_nodes(obs)
        return self.model.act(obs, greedy=True)[0]

    def step(self, action_id=None):
        env = self.env
        if self.done:
            return
        if env.cfg.enable_dag_mode:
            self.step_dag(action_id)
            return
        index = env.idx
        task = env.tasks[index]
        if action_id is None:
            action_id = self.choose_action()
        rejection = env.actions[-1]
        _, reward, self.done, info = env.step(action_id)
        if task.slot // env.cfg.topology_period_slots != env.slot // env.cfg.topology_period_slots:
            self.suspended_links.clear()
        action = env.assignments[-1].action if info["accepted"] else rejection
        finish_time = float(env.assignments[-1].finish_time) if info["accepted"] else None
        self.history.append({
            **task_fields(task, index),
            **action_fields(env, action),
            "accepted": info["accepted"],
            "completed": bool(info["completed"]),
            "reward": reward,
            "finish_time": finish_time,
        })
        self.last_decision = {
            "action_id": action_id,
            "path": list(info["path"]),
            "source": task.source,
            "access_edge": task.access_edge,
            "destination": info["destination"],
        }

    def step_dag(self, action_id):
        env = self.env
        task = dag_task_fields(env)
        if action_id is None:
            action_id = self.choose_action()
        reject = env.actions[-1]
        _, reward, self.done, info = env.step(action_id)
        if task["slot"] // env.cfg.topology_period_slots != env.slot // env.cfg.topology_period_slots:
            self.suspended_links.clear()
        action = env.assignments[-1].action if info["accepted"] else reject
        self.history.append({
            **task,
            **action_fields(env, action),
            "accepted": info["accepted"],
            "reward": reward,
            "start_time_s": None,
            "finish_time_s": None,
            "finish_time": None,
            "setup_delay_s": 0.0,
            "setup_energy_j": 0.0,
            "comm_tier": None,
            "flows": [],
        })
        self.last_decision = {
            "action_id": action_id,
            "dag_id": task["dag_id"],
            "subtask_id": task["subtask_id"],
            "path": list(info["path"]),
            "source": task["source"],
            "access_edge": task["access_edge"],
            "destination": info["destination"],
        }


class SimManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, cfg, policy="shortest_fastest", checkpoint_path=None):
        model = None
        if policy == "checkpoint" or policy in MODEL_NAMES:
            model, _ = load_model_for_environment(checkpoint_path, cfg, policy)
        if policy == "heft_hetero" and not cfg.enable_dag_mode:
            raise ValueError("heft_hetero 需要启用 enable_dag_mode")
        session = SimSession(EdgeEnv(cfg), policy, model)
        session_id = uuid4().hex
        self.sessions[session_id] = session
        return session_id, session.snapshot()

    def get_snapshot(self, session_id):
        session = self.sessions[session_id]
        with session.lock:
            return session.snapshot()

    def step_once(self, session_id, action_id=None):
        session = self.sessions[session_id]
        with session.lock:
            session.step(action_id)
            return session.snapshot()

    def step_slot(self, session_id):
        session = self.sessions[session_id]
        with session.lock:
            env = session.env
            current_slot = env.slot
            while env.slot == current_slot and not session.done:
                session.step()
            return session.snapshot()

    def perturb_link(self, session_id, source, target, active=None, congestion=None,
                     background_fwd=None, background_rev=None):
        session = self.sessions[session_id]
        with session.lock:
            env = session.env
            a, b = sorted((source, target))
            if not 0 <= a < b < env.cloud:
                raise ValueError("扰动仅适用于不同边缘节点之间的链路")
            pair = (a, b)
            if active is False and env.adj[a, b]:
                matrices = (env.rates, env.base_rates, env.background,
                            env.base_background, env.prop, env.forward_power)
                session.suspended_links[pair] = [
                    (matrix[a, b], matrix[b, a]) for matrix in matrices
                ]
                for matrix in (*matrices, env.adj, env.congestion_delta):
                    matrix[a, b] = matrix[b, a] = 0
                env.flapped_edges.append(pair)
            if active is True and not env.adj[a, b]:
                edge_adj = env.adj[:env.cloud, :env.cloud]
                allowed = edge_pairs(env.cfg)
                if pair not in allowed:
                    raise ValueError("八邻接模式只能连接上下左右及四个对角的相邻节点")
                limit = len(allowed) if env.cfg.edge_topology == "eight_neighbor" else env.cfg.undirected_edge_links
                if np.count_nonzero(np.triu(edge_adj, k=1)) >= limit:
                    raise ValueError("活跃边缘链路已达配置上限")
                if pair in session.suspended_links:
                    values = session.suspended_links.pop(pair)
                    matrices = (env.rates, env.base_rates, env.background,
                                env.base_background, env.prop, env.forward_power)
                    for matrix, (fwd, rev) in zip(matrices, values):
                        matrix[a, b], matrix[b, a] = fwd, rev
                    env.adj[a, b] = env.adj[b, a] = 1
                else:
                    _activate_link(env.rng, env.cfg, a, b, env.adj, env.base_rates,
                                   env.base_background, env.prop, env.forward_power)
                    for start, end in ((a, b), (b, a)):
                        env.rates[start, end] = env.base_rates[start, end]
                        env.background[start, end] = env.base_background[start, end]
                if pair in env.flapped_edges:
                    env.flapped_edges.remove(pair)
            if env.adj[a, b]:
                for start, end, value in (
                    (source, target, background_fwd),
                    (target, source, background_rev),
                ):
                    value = congestion if value is None else value
                    if value is not None:
                        env.congestion_delta[start, end] = value - env.background[start, end]
                        env.background[start, end] = value
            if not session.done:
                env.observe()
            return session.snapshot()

    def reset_session(self, session_id):
        session = self.sessions[session_id]
        with session.lock:
            session.env = EdgeEnv(session.env.cfg)
            session.done = False
            session.history.clear()
            session.suspended_links.clear()
            session.last_decision = None
            session.reset_policy()
            return session.snapshot()

    def delete_session(self, session_id):
        del self.sessions[session_id]
