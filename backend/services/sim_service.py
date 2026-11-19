from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np
import torch

from src.ablation import model_environment, shuffle_graph_observation, ShuffledGraphEnv
from src.baselines import greedy_estimate_action, shortest_fastest_action
from src.config import Config, load_config
from src.environment import EdgeEnv
from src.models.gat import GATEncoder
from src.models.policy import GATPPO
from src.topology import _activate_link

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


def checkpoint_config(checkpoint):
    values = {key: value for key, value in checkpoint["config"].items()
              if key in Config.__dataclass_fields__}
    state = checkpoint["model"]
    values.update(
        gat_hidden_dim=state["gat.layer1.transform.weight"].shape[0],
        actor_hidden_dim=state["actor.0.weight"].shape[0],
        critic_hidden_dim=state["critic.0.weight"].shape[0],
    )
    return config_from_dict(values)


def build_model(cfg, model_type, checkpoint=None, *, inference=False):
    device = "meta" if inference else "cpu"
    with torch.device(device):
        model = GATPPO(
            gat_hidden=cfg.gat_hidden_dim,
            actor_hidden=cfg.actor_hidden_dim,
            critic_hidden=cfg.critic_hidden_dim,
            use_gat=model_type != "flat",
        )
        if checkpoint is not None:
            state = checkpoint["model"]
            key = "gat.layer1.attention.weight"
            edge_dim = state[key].shape[1] - 2 * cfg.gat_hidden_dim
            if model.state_dict()[key].shape != state[key].shape:
                model.gat = GATEncoder(hidden=cfg.gat_hidden_dim, edge_dim=edge_dim)
                model.gat.register_forward_pre_hook(
                    lambda _, args: (args[0], args[1], args[2][..., :edge_dim])
                )
            model.load_state_dict(state, assign=inference)
    return model.eval() if inference else model


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


def serialize_env_snapshot(env, history, last_decision=None):
    done = env.idx == len(env.tasks)
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

    links = [
        link_snapshot(env, int(source), int(target), "active")
        for source, target in zip(*np.nonzero(np.triu(env.adj, k=1)))
    ]
    links.extend(link_snapshot(env, source, target, "flapped")
                 for source, target in env.flapped_edges)

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
        "slot": env.slot,
        "episode_slots": env.cfg.episode_slots,
        "task_index": env.idx,
        "total_tasks": len(env.tasks),
        "done": done,
        "topology_period_slots": env.cfg.topology_period_slots,
        "nodes": [node_snapshot(env, node, active_task_ids[node])
                  for node in range(env.n_graph_nodes)],
        "terminals": [
            {"id": terminal, "label": f"端侧 {terminal}",
             "access_edge": terminal % env.cloud,
             "radio_rate_mbps": float(rate)}
            for terminal, rate in enumerate(env.radio_rates)
        ],
        "links": links,
        "current_task": current_task,
        "recent_tasks": recent_tasks,
        "metrics": metrics,
        "last_decision": last_decision,
    }


@dataclass
class SimSession:
    env: EdgeEnv
    policy: str
    model: GATPPO | None = None
    model_type: str | None = None
    history: list = field(default_factory=list)
    last_decision: dict | None = None
    suspended_links: dict = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self):
        self.reset_policy()

    def reset_policy(self):
        self.rng = np.random.default_rng(self.env.cfg.seed)
        wrapped = model_environment(self.env, self.model_type, self.env.cfg.seed)
        self.permutation = wrapped.permutation if isinstance(wrapped, ShuffledGraphEnv) else None

    def snapshot(self):
        return serialize_env_snapshot(self.env, self.history, self.last_decision)

    def choose_action(self):
        if self.policy == "shortest_fastest":
            return shortest_fastest_action(self.env.actions)
        if self.policy == "greedy_estimate":
            return greedy_estimate_action(self.env.actions, self.env.cfg)
        if self.policy == "random":
            return int(self.rng.integers(len(self.env.actions)))
        obs = self.env.observe()
        if self.permutation is not None:
            obs = shuffle_graph_observation(obs, self.permutation)
        return self.model.act(obs, greedy=True)[0]

    def step(self, action_id=None):
        env = self.env
        if env.idx == len(env.tasks):
            return
        index = env.idx
        task = env.tasks[index]
        if action_id is None:
            action_id = self.choose_action()
        rejection = env.actions[-1]
        _, reward, _, info = env.step(action_id)
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


class SimManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self, cfg, policy="shortest_fastest", checkpoint_path=None):
        env = EdgeEnv(cfg)
        model = None
        model_type = None
        if policy == "checkpoint":
            checkpoint = load_checkpoint(checkpoint_path)
            model_type = checkpoint["model_type"]
            model = build_model(checkpoint_config(checkpoint), model_type, checkpoint, inference=True)
        session = SimSession(env, policy, model, model_type)
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
            while env.idx < len(env.tasks) and env.slot == current_slot:
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
                if np.count_nonzero(np.triu(edge_adj, k=1)) >= env.cfg.undirected_edge_links:
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
            if env.idx < len(env.tasks):
                env.observe()
            return session.snapshot()

    def reset_session(self, session_id):
        session = self.sessions[session_id]
        with session.lock:
            session.env = EdgeEnv(session.env.cfg)
            session.history.clear()
            session.suspended_links.clear()
            session.last_decision = None
            session.reset_policy()
            return session.snapshot()

    def delete_session(self, session_id):
        del self.sessions[session_id]
