"""云–边–端多跳异构调度环境：在线任务、联合节点/路径/资源动作。"""
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import ulp
from typing import NamedTuple
import numpy as np

from .config import Config
from .resources import ActiveReservation, DEMAND_FIELDS, ResourcePool
from .tasks import CPU, FPGA, NPU, OP_NAMES, SubTask, TaskDAG, sample_dag_tasks, sample_tasks
from .topology import fluctuate_rates, fluctuate_topology, make_topology, simple_routes, node_positions
from .continuous import allocate_compute, choose_uplink_power


@dataclass
class Action:
    destination: int  # -2: 拒绝, 0..M-1: 边缘, M: 云
    resource: int     # 0: CPU, 1: NPU, 2: FPGA
    path: tuple[int, ...]
    delay: float
    energy: float
    compute_share: float
    transport_delay: float = 0.0
    transport_energy: float = 0.0
    uplink_power: float = 0.0
    uplink_rate: float = 0.0
    dinkelbach_residual: float = 0.0
    estimated_delay: float = 0.0
    estimated_energy: float = 0.0


@dataclass
class Assignment:
    task_index: int
    action: Action
    compute_degradation: float
    realized_background: dict
    actual_transport_delay: float
    actual_transport_energy: float
    finish_time: float


class Flow(NamedTuple):
    pred_id: int  # -1 为入口无线输入，-2 为出口回传
    path: tuple[int, ...]
    comm_tier: int
    data_mb: float


class Transport(NamedTuple):
    incoming: float  # 从派发到最后一条入流到齐，不含入口无线段
    outgoing: float  # 从计算完成到出口交付，含无线下行
    energy: float


@dataclass
class DAGAction(Action):
    dag_id: int = -1
    subtask_id: int = -1
    flows: tuple[Flow, ...] = ()
    comm_tier: int = 0
    cold_switch: bool = False
    setup_delay_s: float = 0.
    setup_energy_j: float = 0.
    dispatch_time_s: float = 0.
    context_ready_at: float = 0.
    start_time_s: float = 0.
    finish_time_s: float = 0.
    delivery_time_s: float = 0.
    effective_work: float = 0.
    minimum_f: float = 0.
    transport: Transport = Transport(0., 0., 0.)


@dataclass
class DAGAssignment:
    action: DAGAction
    reservation: ActiveReservation
    compute_degradation: float
    actual_transport: Transport


@dataclass
class DAGState:
    dag: TaskDAG
    arrival: float
    nodes: dict[int, SubTask]
    status: dict[int, str]
    remaining_s: dict[int, float]
    assignments: dict[int, DAGAssignment] = field(default_factory=dict)
    terminal_time: float | None = None
    success: bool = False
    balance: tuple[float, float, float] = (0., 0., 0.)

    @property
    def deadline(self):
        return self.arrival + self.dag.deadline_s


class EdgeEnv:
    """一个 step 调度一项任务，任务按自身到达时隙推进环境。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.cloud = cfg.n_edges
        self.n_graph_nodes = cfg.n_edges + 1
        self.node_positions, self.terminal_positions = node_positions(cfg.n_edges, cfg.n_terminals)
        self.access_edges = np.arange(cfg.n_terminals) % cfg.n_edges
        if cfg.terminal_access_mode == "nearest":
            distances = ((self.terminal_positions[:, None, :]
                          - self.node_positions[None, :cfg.n_edges, :]) ** 2).sum(axis=2)
            self.access_edges = distances.argmin(axis=1)
        edge_ids = np.arange(cfg.n_edges)
        edge_speeds = np.asarray(cfg.edge_speed_matrix)[edge_ids % len(cfg.edge_speed_matrix)]
        self.speed = np.vstack((edge_speeds, cfg.cloud_speed)).astype(np.float32)
        self.service_deployment = np.asarray(cfg.service_deployment, dtype=bool)[
            edge_ids % len(cfg.service_deployment)
        ]
        self.mem_cap = np.array(cfg.node_memory_gb or [32.] * cfg.n_edges + [128.], dtype=np.float32)
        self.storage_cap = np.array(cfg.node_storage_gb or [1000.] * cfg.n_edges + [4000.], dtype=np.float32)
        self.queue_cap = np.array(cfg.node_queue_capacity or [60.] * cfg.n_edges + [80.], dtype=np.float32)
        self.resource_power = np.array([10., 18., 12.], dtype=np.float32)
        self.support = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool)
        self.reset()

    def reset(self):
        c = self.cfg
        # 接收动作决定扰动采样次数，独立的执行随机流让同种子策略共享链路轨迹。
        self.execution_rng = np.random.default_rng(self.rng.integers(2**63))
        self.radio_rates = self.rng.uniform(
            c.radio_rate_min_mbps, c.radio_rate_max_mbps, c.n_terminals,
        )
        if c.terminal_radio_rates_mbps:
            self.radio_rates = np.asarray(c.terminal_radio_rates_mbps, dtype=float)
        self.radio_channel_gain = (
            2.0 ** (self.radio_rates / c.radio_bandwidth_mhz) - 1.0
        ) / c.uplink_max_power_w
        if c.enable_dag_mode:
            return self._reset_dag()
        self.tasks = sample_tasks(
            self.rng,
            c.episode_slots,
            c.arrival_rate,
            c.burst_prob,
            c.burst_multiplier,
            c.hotspot_fraction,
            c.hotspot_share,
            c.topology_period_slots,
            c.n_terminals,
            c.n_edges,
            radio_rates=self.radio_rates,
            min_edge_compute_gflops=float(self.speed[:self.cloud].min()),
            deadline_slack_min=c.deadline_slack_min,
            deadline_slack_max=c.deadline_slack_max,
            access_edges=self.access_edges,
        )
        self._new_topology()
        self.idx = 0
        self.slot = 0
        self.slot_resource_used = np.zeros_like(self.speed)
        self.slot_mem_used = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_storage_used = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_queues = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_link_used = np.zeros_like(self.rates)
        self.slot_assignments = []
        self.active_assignments = []
        self.assignments = []
        self.carryover_resource_used = np.zeros_like(self.speed)
        self.slot_step_count = 0
        self.accepted = 0
        self.sum_delay = 0.
        self.sum_priority_delay = 0.
        self.sum_priority = 0.
        self.total_energy = 0.
        self.total_reward = 0.
        self.total_hops = 0
        self.total_path_length = 0
        self.total_load_variance = 0.
        self.load_samples = 0
        self.feasible_tasks = 0
        self.conditional_rejects = 0
        self.forced_rejects = 0
        self.action_count_sum = 0
        self.edge_executions = 0
        self.cloud_executions = 0
        return self.observe()

    def _new_topology(self):
        (self.adj, self.base_rates, self.base_background,
         self.prop, self.forward_power) = make_topology(self.rng, self.cfg)
        self.rates = fluctuate_rates(self.rng, self.cfg, self.base_rates)
        self.background = self.base_background.copy()
        self.congestion_delta = np.zeros_like(self.background)
        self.flapped_edges = []
        self._route_cache = {}

    def _routes(self, source, destination, max_hops):
        sig = self.adj.tobytes()
        if getattr(self, "_route_cache_sig", None) != sig:
            self._route_cache = {}
            self._route_cache_sig = sig
        key = (source, destination, max_hops)
        cached = self._route_cache.get(key)
        if cached is None:
            cached = tuple(simple_routes(self.adj, source, destination, max_hops))
            self._route_cache[key] = cached
        return list(cached)

    def _slot_capacity_mb(self, a, b):
        return self.rates[a, b] * (1 - self.background[a, b]) * self.cfg.slot_seconds / 8

    def _load_variance(self):
        edge_load = (
            self.slot_resource_used[:self.cloud].sum(1)
            / self.speed[:self.cloud].sum(1)
        )
        return float(np.var(edge_load))

    def _rebuild_occupancy(self):
        self.slot_resource_used.fill(0)
        self.slot_mem_used.fill(0)
        self.slot_storage_used.fill(0)
        self.slot_queues.fill(0)
        for assignment in self.active_assignments:
            task = self.tasks[assignment.task_index]
            action = assignment.action
            destination = action.destination
            self.slot_resource_used[destination, action.resource] += action.compute_share
            self.slot_mem_used[destination] += task.memory_gb
            self.slot_storage_used[destination] += task.storage_gb
            self.slot_queues[destination] += 1

    def _advance_to_slot(self, target_slot):
        while self.slot < target_slot:
            self.total_load_variance += self._load_variance()
            self.load_samples += 1
            self.slot += 1
            slot_start = self.slot * self.cfg.slot_seconds
            self.active_assignments = [
                assignment for assignment in self.active_assignments
                if assignment.finish_time > slot_start
            ]
            self._rebuild_occupancy()
            self.carryover_resource_used = self.slot_resource_used.copy()
            self.slot_link_used.fill(0)
            self.slot_assignments.clear()
            self.slot_step_count = 0
            self._update_topology()

    def _update_topology(self):
        if self.slot % self.cfg.topology_period_slots == 0:
            self._new_topology()
        else:
            (self.rates, self.background, self.congestion_delta,
             self.flapped_edges) = fluctuate_topology(
                self.rng, self.cfg, self.adj, self.base_rates, self.base_background,
                self.prop, self.forward_power, self.background, self.flapped_edges,
            )
            self._route_cache = {}

    def _transport_terms(self, task, destination, path, realized_background=None):
        delay = task.output_mb * 8 / self.radio_rates[task.source]
        energy = self.cfg.uplink_circuit_power_w * delay
        for a, b in zip(path, path[1:]):
            if realized_background is None:
                if task.data_mb + self.slot_link_used[a, b] > self._slot_capacity_mb(a, b):
                    return None
                if task.output_mb + self.slot_link_used[b, a] > self._slot_capacity_mb(b, a):
                    return None
                upload_background = self.background[a, b]
                download_background = self.background[b, a]
            else:
                upload_background = realized_background[a, b]
                download_background = realized_background[b, a]
            upload_capacity = (
                self.rates[a, b] * (1 - upload_background) * self.cfg.slot_seconds / 8
            )
            download_capacity = (
                self.rates[b, a] * (1 - download_background) * self.cfg.slot_seconds / 8
            )
            upload = task.data_mb * 8 / (self.rates[a, b] * (1 - upload_background))
            upload_queue = .02 * self.slot_link_used[a, b] / upload_capacity
            download = task.output_mb * 8 / (self.rates[b, a] * (1 - download_background))
            download_queue = .02 * self.slot_link_used[b, a] / download_capacity
            delay += upload + upload_queue + self.prop[a, b]
            delay += download + download_queue + self.prop[b, a]
            energy += self.forward_power[a, b] * (upload + upload_queue)
            energy += self.forward_power[b, a] * (download + download_queue)
        delay += .005 * self.slot_queues[destination]
        return delay, energy

    def _estimated_completion(
        self, task, resource, share, transport_delay, transport_energy,
    ):
        compute_delay = task.compute_gflop / share
        compute_energy = self.resource_power[resource] * compute_delay
        remaining_uplink_time = task.deadline_s - transport_delay - compute_delay
        if remaining_uplink_time <= 0.0:
            return None
        min_rate = task.data_mb * 8 / remaining_uplink_time
        if min_rate > self.radio_rates[task.source]:
            return None
        solution = choose_uplink_power(
            self.cfg.radio_bandwidth_mhz,
            self.radio_channel_gain[task.source],
            self.cfg.uplink_circuit_power_w,
            self.cfg.uplink_max_power_w,
            min_rate,
            self.cfg.dinkelbach_tolerance,
            self.cfg.dinkelbach_max_iterations,
        )
        uplink_time = task.data_mb * 8 / solution.rate_mbps
        delay = transport_delay + compute_delay + uplink_time
        radio_energy = (solution.power_w + self.cfg.uplink_circuit_power_w) * uplink_time
        energy = transport_energy + compute_energy + radio_energy
        return delay, energy, solution.power_w, solution.rate_mbps, solution.residual

    def _actual_completion(
        self, task, action, share, degradation, transport_delay, transport_energy,
    ):
        compute_delay = task.compute_gflop / (share * (1 - degradation))
        compute_energy = self.resource_power[action.resource] * compute_delay
        uplink_time = task.data_mb * 8 / action.uplink_rate
        radio_energy = (
            action.uplink_power + self.cfg.uplink_circuit_power_w
        ) * uplink_time
        return (
            transport_delay + compute_delay + uplink_time,
            transport_energy + compute_energy + radio_energy,
        )

    def _group_allocation(self, task, destination, resource, transport_delay, transport_energy):
        group = [
            assignment for assignment in self.slot_assignments
            if assignment.action.destination == destination and assignment.action.resource == resource
        ]
        group_tasks = [self.tasks[assignment.task_index] for assignment in group] + [task]
        transport_delays = [assignment.action.transport_delay for assignment in group] + [transport_delay]
        capacity = self.speed[destination, resource] - self.carryover_resource_used[destination, resource]
        minimum_shares = []
        for item, fixed_delay in zip(group_tasks, transport_delays):
            fixed_delay += item.data_mb * 8 / self.radio_rates[item.source]
            if fixed_delay >= item.deadline_s:
                return None
            minimum_shares.append(item.compute_gflop / (item.deadline_s - fixed_delay))
        shares = allocate_compute(
            [item.compute_gflop for item in group_tasks],
            [item.priority for item in group_tasks],
            capacity,
            minimum_shares,
            self.cfg.compute_priority_exponent,
        )
        if shares is None:
            return None
        outcomes = []
        for assignment, share in zip(group, shares[:-1]):
            action = assignment.action
            other_task = self.tasks[assignment.task_index]
            outcome = self._estimated_completion(
                other_task, resource, share, action.transport_delay,
                action.transport_energy,
            )
            if outcome is None:
                return None
            outcomes.append((assignment, float(share), outcome))
        candidate = self._estimated_completion(
            task, resource, shares[-1], transport_delay, transport_energy,
        )
        if candidate is None:
            return None
        return outcomes, float(shares[-1]), candidate

    def _candidate(self, task, destination, resource, path):
        if self.slot_mem_used[destination] + task.memory_gb > self.mem_cap[destination]:
            return None
        if self.slot_storage_used[destination] + task.storage_gb > self.storage_cap[destination]:
            return None
        if self.slot_queues[destination] + 1 > self.queue_cap[destination]:
            return None
        transport = self._transport_terms(task, destination, path)
        if transport is None:
            return None
        transport_delay, transport_energy = transport
        allocation = self._group_allocation(
            task, destination, resource, transport_delay, transport_energy,
        )
        if allocation is None:
            return None
        _, share, completion = allocation
        delay, energy, power, rate, residual = completion
        return Action(
            destination, resource, path, delay, energy, share,
            transport_delay, transport_energy, power, rate, residual,
            delay, energy,
        )

    def _sample_uncertainty(self, task, action):
        realized = {}
        for a, b in zip(action.path, action.path[1:]):
            for source, target in ((a, b), (b, a)):
                background = self.background[source, target]
                sigma_eff = self.cfg.congestion_noise * (0.5 + background)
                realized[source, target] = float(np.clip(
                    background + self.execution_rng.uniform(-sigma_eff, sigma_eff),
                    0.0, self.cfg.congestion_realized_max,
                ))
        degradation = float(self.execution_rng.uniform(0.0, self.cfg.compute_degradation_max))
        transport_delay, transport_energy = self._transport_terms(
            task, action.destination, action.path, realized,
        )
        return realized, degradation, transport_delay, transport_energy

    def _task_reward_parts(self, task, action):
        completed = action.delay <= task.deadline_s
        completion = self.cfg.reward_beta_completion * float(completed)
        delay = -self.cfg.reward_beta_delay * action.delay / self.cfg.reward_delay_ref
        energy = -self.cfg.reward_beta_energy * action.energy / self.cfg.reward_energy_ref
        incomplete = -self.cfg.reward_penalty_incomplete * float(not completed)
        return completion, delay, energy, incomplete

    def _make_candidates(self):
        task = self.tasks[self.idx]
        actions = []
        if task.allow_offload:
            source = task.access_edge
            for destination in range(self.n_graph_nodes):
                cloud = destination == self.cloud
                if not cloud and not self.service_deployment[destination, task.kind]:
                    continue
                if cloud and self.cfg.cloud_gateway_count == 0:
                    # 历史配置保持源边缘直接上云，保证旧 checkpoint 可复现。
                    routes = [(source, self.cloud)] if self.adj[source, self.cloud] else []
                else:
                    routes = [(source,)] if destination == source else self._routes(
                        source, destination, self.cfg.max_hops + int(cloud))
                routes.sort(key=lambda p: sum(
                    8 * task.data_mb / (self.rates[a, b] * (1 - self.background[a, b]))
                    for a, b in zip(p, p[1:])))
                for path in routes[:self.cfg.k_routes_per_destination]:
                    for q in range(3):
                        if self.support[q, task.kind]:
                            action = self._candidate(task, destination, q, path)
                            if action is not None:
                                actions.append(action)
        actions.append(Action(-2, -1, (), 0., 0., 0.))
        return actions

    def _graph_features(self):
        n = self.n_graph_nodes
        node = np.zeros((n, 10), dtype=np.float32)
        for j in range(n):
            neighbors = [k for k in range(n) if k != j and self.adj[j, k]]
            node[j, :3] = self.slot_resource_used[j] / self.speed[j]
            node[j, 3] = self.slot_mem_used[j] / self.mem_cap[j]
            node[j, 4] = self.slot_storage_used[j] / self.storage_cap[j]
            node[j, 5] = self.slot_queues[j] / self.queue_cap[j]
            node[j, 6] = np.mean([self.background[j, k] + self.slot_link_used[j, k] /
                                  self._slot_capacity_mb(j, k) for k in neighbors])
            node[j, 7] = len(neighbors) / n
            node[j, 8] = float(j == self.cloud)
            node[j, 9] = float(self.speed[j].sum() / 10000)
        return node, self._edge_features()

    def _edge_features(self):
        n = self.n_graph_nodes
        edge = np.zeros((n, n, 4), dtype=np.float32)
        edge[:, :, 0] = self.rates / self.cfg.link_rate_max_mbps
        slot_capacity = self.rates * (1 - self.background) * self.cfg.slot_seconds / 8
        edge[:, :, 1] = self.background + np.divide(
            self.slot_link_used,
            slot_capacity,
            out=np.zeros_like(self.slot_link_used),
            where=slot_capacity > 0,
        )
        edge[:, :, 2] = self.prop / .003
        edge[:, :, 3] = self.congestion_delta
        return edge

    def observe(self):
        if self.cfg.enable_dag_mode:
            return self._observe_dag()
        task = self.tasks[self.idx]
        self.actions = self._make_candidates()
        node, edge = self._graph_features()
        cand = np.zeros((len(self.actions), 10), dtype=np.float32)
        targets = np.empty(len(self.actions), dtype=np.int64)
        paths = np.zeros((len(self.actions), self.n_graph_nodes), dtype=np.float32)
        for a, action in enumerate(self.actions):
            reject = action.destination == -2
            targets[a] = task.access_edge if action.destination < 0 else action.destination
            route = action.path or (task.access_edge,)
            paths[a, list(route)] = 1 / len(route)
            cand[a, :3] = [float(reject), 0.0, float(action.destination == self.cloud)]
            cand[a, 3:6] = [float(action.resource == q) for q in range(3)]
            edge_hops = max(0, len(route) - 1 - int(action.destination == self.cloud))
            cand[a, 6] = edge_hops / self.cfg.max_hops
            cand[a, 7] = action.delay / self.cfg.reward_delay_ref
            cand[a, 8] = action.energy / 10
            cand[a, 9] = self.slot_queues[targets[a]] / self.queue_cap[targets[a]]
        task_vector = np.array([
            task.data_mb / 8, task.output_mb, task.compute_gflop / 10,
            task.deadline_s / self.cfg.reward_delay_ref, task.priority / 5,
            task.memory_gb / 2, task.kind / 4, float(task.allow_offload),
        ], dtype=np.float32)
        task_count = len(self.tasks)
        slot_task_count = sum(item.slot == self.slot for item in self.tasks)
        processed_in_slot = sum(item.slot == self.slot for item in self.tasks[:self.idx])
        global_features = np.array([
            self.idx / task_count,
            self.accepted / task_count,
            self.accepted / max(1, self.idx),
            self.sum_delay / (task_count * self.cfg.reward_delay_ref),
            self.total_energy / (task_count * self.cfg.reward_energy_ref),
            self.total_load_variance / (self.cfg.episode_slots * self.cfg.reward_variance_ref),
            self._load_variance() / self.cfg.reward_variance_ref,
            self.slot / self.cfg.episode_slots,
            processed_in_slot / slot_task_count,
        ], dtype=np.float32)
        gat_adj = self.adj.copy()
        # 云端用于执行和广域链路，不作为边缘间图消息传递的超级中继。
        gat_adj[:self.cloud, self.cloud] = 0
        gat_adj[self.cloud, :self.cloud] = 0
        return {
            "nodes": node,
            "adjacency": gat_adj,
            "edges": edge,
            "task": task_vector,
            "source": task.access_edge,
            "candidate_features": cand,
            "candidate_targets": targets,
            "candidate_paths": paths,
            "global_features": global_features,
        }

    def step(self, action_id: int):
        if self.cfg.enable_dag_mode:
            return self._step_dag(action_id)
        task = self.tasks[self.idx]
        feasible_count = len(self.actions) - 1
        invalid_action = action_id < 0 or action_id >= len(self.actions)
        action = self.actions[-1] if invalid_action else self.actions[action_id]
        accepted = not invalid_action and action.destination != -2
        load_before = 0.0 if self.slot_step_count == 0 else self._load_variance()
        self.action_count_sum += len(self.actions)
        if feasible_count:
            self.feasible_tasks += 1
            if not accepted and not invalid_action:
                self.conditional_rejects += 1
        elif not accepted and not invalid_action:
            self.forced_rejects += 1
        reward_group_correction_part = 0.0
        if accepted:
            allocation = self._group_allocation(
                task, action.destination, action.resource,
                action.transport_delay, action.transport_energy,
            )
            group_outcomes, candidate_share, candidate_outcome = allocation
            for assignment, share, outcome in group_outcomes:
                other_task = self.tasks[assignment.task_index]
                previous_parts = self._task_reward_parts(other_task, assignment.action)
                previous_delay = assignment.action.delay
                previous_energy = assignment.action.energy
                estimated_delay, estimated_energy, power, rate, residual = outcome
                assignment.action.compute_share = share
                assignment.action.estimated_delay = estimated_delay
                assignment.action.estimated_energy = estimated_energy
                assignment.action.uplink_power = power
                assignment.action.uplink_rate = rate
                assignment.action.dinkelbach_residual = residual
                delay, energy = self._actual_completion(
                    other_task,
                    assignment.action,
                    share,
                    assignment.compute_degradation,
                    assignment.actual_transport_delay,
                    assignment.actual_transport_energy,
                )
                self.sum_delay += delay - previous_delay
                self.sum_priority_delay += other_task.priority * (delay - previous_delay)
                self.total_energy += energy - previous_energy
                assignment.action.delay = delay
                assignment.action.energy = energy
                assignment.finish_time = (
                    other_task.slot * self.cfg.slot_seconds + delay
                )
                current_parts = self._task_reward_parts(other_task, assignment.action)
                reward_group_correction_part += sum(current_parts) - sum(previous_parts)
            action.compute_share = candidate_share
            (action.estimated_delay, action.estimated_energy, action.uplink_power,
             action.uplink_rate, action.dinkelbach_residual) = candidate_outcome
            realized, degradation, actual_transport_delay, actual_transport_energy = (
                self._sample_uncertainty(task, action)
            )
            action.delay, action.energy = self._actual_completion(
                task,
                action,
                candidate_share,
                degradation,
                actual_transport_delay,
                actual_transport_energy,
            )
            assignment = Assignment(
                self.idx,
                action,
                degradation,
                realized,
                actual_transport_delay,
                actual_transport_energy,
                task.slot * self.cfg.slot_seconds + action.delay,
            )
            self.slot_assignments.append(assignment)
            self.active_assignments.append(assignment)
            self.assignments.append(assignment)
            self.accepted += 1
            self.sum_delay += action.delay
            self.sum_priority_delay += task.priority * action.delay
            self.sum_priority += task.priority
            self.total_energy += action.energy
            self.total_hops += max(0, len(action.path) - 1 - int(action.destination == self.cloud))
            self.total_path_length += len(action.path)
            self._rebuild_occupancy()
            for a, b in zip(action.path, action.path[1:]):
                self.slot_link_used[a, b] += task.data_mb
                self.slot_link_used[b, a] += task.output_mb
            if action.destination == self.cloud:
                self.cloud_executions += 1
            else:
                self.edge_executions += 1
        completed = accepted and action.delay <= task.deadline_s
        normalized_delay = action.delay / self.cfg.reward_delay_ref if accepted else 0.0
        normalized_energy = action.energy / self.cfg.reward_energy_ref if accepted else 0.0
        load_after = self._load_variance()
        if invalid_action:
            constraint_penalty = self.cfg.reward_penalty_invalid
        elif accepted and not completed:
            constraint_penalty = self.cfg.reward_penalty_incomplete
        elif not accepted and feasible_count:
            constraint_penalty = self.cfg.reward_penalty_conditional_reject
        else:
            constraint_penalty = 0.0
        reward_completion_part = self.cfg.reward_beta_completion * float(completed)
        reward_delay_part = (
            -self.cfg.reward_beta_delay * float(accepted) * normalized_delay
        )
        reward_energy_part = (
            -self.cfg.reward_beta_energy * float(accepted) * normalized_energy
        )
        reward_load_part = (
            -self.cfg.reward_beta_load
            * (load_after - load_before)
            / self.cfg.reward_variance_ref
        )
        reward_constraint_part = -constraint_penalty
        # r_i = beta_1*delta_i - y_i*(beta_2*t_i + beta_3*E_i)
        #       - beta_4*D_load - Omega_i
        reward = (
            reward_completion_part
            + reward_delay_part
            + reward_energy_part
            + reward_group_correction_part
            + reward_load_part
            + reward_constraint_part
        )
        self.total_reward += reward
        self.slot_step_count += 1
        self.idx += 1
        done = self.idx == len(self.tasks)
        if done:
            self.total_load_variance += self._load_variance()
            self.load_samples += 1
        elif self.tasks[self.idx].slot != self.slot:
            self._advance_to_slot(self.tasks[self.idx].slot)
        obs = None if done else self.observe()
        return obs, float(reward), done, {"accepted": accepted, "destination": action.destination,
                                          "resource": action.resource, "path": action.path,
                                          "delay": action.delay, "energy": action.energy,
                                          "estimated_delay": action.estimated_delay,
                                          "estimated_energy": action.estimated_energy,
                                          "load_variance": load_after,
                                          "load_variance_before": load_before,
                                          "f": action.compute_share,
                                          "p_up": action.uplink_power,
                                          "dinkelbach_residual": action.dinkelbach_residual,
                                          "feasible_count": feasible_count,
                                          "invalid_action": invalid_action,
                                          "completed": completed,
                                          "reward_completion_part": reward_completion_part,
                                          "reward_delay_part": reward_delay_part,
                                          "reward_energy_part": reward_energy_part,
                                          "reward_group_correction_part": reward_group_correction_part,
                                          "reward_load_part": reward_load_part,
                                          "reward_constraint_part": reward_constraint_part}

    def metrics(self):
        if self.cfg.enable_dag_mode:
            return self._dag_metrics()
        count = len(self.tasks)
        successful = [
            assignment for assignment in self.assignments
            if assignment.action.delay <= self.tasks[assignment.task_index].deadline_s
        ]
        avg_delay = (
            sum(item.action.delay for item in successful) / len(successful)
            if successful else None
        )
        successful_priority = sum(
            self.tasks[item.task_index].priority for item in successful
        )
        priority_delay = (
            sum(
                self.tasks[item.task_index].priority * item.action.delay
                for item in successful
            ) / successful_priority
            if successful else None
        )
        avg_energy = (
            sum(item.action.energy for item in successful) / len(successful)
            if successful else None
        )
        mean_hops = self.total_hops / self.accepted if self.accepted else None
        mean_path_length = (
            self.total_path_length / self.accepted if self.accepted else None
        )
        load_variance = self.total_load_variance / self.load_samples
        objective_score = self.total_reward / count
        return {
            "tasks": count,
            "task_success_rate": len(successful) / count,
            "acceptance_rate": self.accepted / count,
            "deadline_miss_rate": (
                (self.accepted - len(successful)) / self.accepted
                if self.accepted else 0.0
            ),
            "avg_delay_s": avg_delay,
            "priority_weighted_delay_s": priority_delay,
            "total_energy_j": self.total_energy,
            "avg_energy_j": avg_energy,
            "mean_load_variance": load_variance,
            "load_balance_degree": 1.0 / (1.0 + load_variance),
            "mean_hops_accepted": mean_hops,
            "episode_reward": self.total_reward,
            "objective_score": objective_score,
            "reward_sum_check": self.total_reward / count,
            "feasible_task_rate": self.feasible_tasks / count,
            "conditional_reject_rate": (
                self.conditional_rejects / self.feasible_tasks
                if self.feasible_tasks else 0.0
            ),
            "forced_reject_rate": self.forced_rejects / count,
            "mean_action_count": self.action_count_sum / count,
            "local_execution_rate": 0.0,
            "edge_execution_rate": self.edge_executions / count,
            "cloud_execution_rate": self.cloud_executions / count,
            "mean_path_length": mean_path_length,
        }


    def _reset_dag(self):
        c = self.cfg
        self.resources = ResourcePool(c)
        self.speed = self.resources.speed
        self.active_reservations = self.resources.active
        self.z_service = np.vstack((self.service_deployment, np.ones((1, 5), dtype=bool)))
        self.dags = sample_dag_tasks(self.rng, c, self.radio_rates, access_edges=self.access_edges)
        self.dag_states = {}
        self.arrivals = defaultdict(list)
        avg_rate = (c.link_rate_min_mbps + c.link_rate_max_mbps) / 2
        for dag in self.dags:
            nodes = {t.subtask_id: t for t in dag.subtasks}
            remaining = {i: max((8 * dag.edge_data_mb[i, j] / avg_rate + nodes[j].rank_u
                                 for j in t.succs), default=0.) for i, t in nodes.items()}
            state = DAGState(dag, dag.slot * c.slot_seconds, nodes,
                             dict.fromkeys(nodes, "waiting"), remaining)
            self.dag_states[dag.dag_id] = state
            self.arrivals[dag.slot].append(state)
        self._new_topology()
        self.slot_link_used = np.zeros(self.rates.shape, dtype=float)
        self.current_time = 0.
        self.slot = 0
        self.idx = 0
        self.ready = []
        self.assignments = []
        self.concurrent = defaultdict(list)
        self.pending_deliveries = {}
        self.slot_load_samples = []
        self.sampled_load = 0.
        self.total_reward = 0.
        self.missed_dags = 0
        self.conditional_rejects = self.forced_rejects = self.feasible_tasks = 0
        self.action_count_sum = 0
        self._dag_arrivals()
        self._next_ready(set())
        return self.observe()

    def _enqueue(self, state, i):
        state.status[i] = "ready"
        heappush(self.ready, (-state.nodes[i].rank_u, state.dag.dag_id, i))

    def _dag_arrivals(self):
        for state in self.arrivals.get(self.slot, ()):
            for i in state.dag.entry_ids:
                self._enqueue(state, i)

    def _finish_dag(self, state, success, changed):
        state.terminal_time = self.current_time
        state.success = success
        changed.add(state.dag.dag_id)
        if not success:
            self.missed_dags += int(bool(state.assignments))
            for i, status in state.status.items():
                if status in ("waiting", "ready"):
                    state.status[i] = "cancelled"

    def _dag_completions(self, changed):
        released = [key for key, r in self.active_reservations.items()
                    if r.release_time <= self.current_time]
        for dag_id, i in released:
            del self.active_reservations[dag_id, i]
            state = self.dag_states[dag_id]
            state.status[i] = "completed"
            if state.terminal_time is not None:
                continue
            if i in state.dag.exit_ids:
                action = state.assignments[i].action
                self.pending_deliveries[dag_id, i] = action.delivery_time_s
            else:
                for j in state.nodes[i].succs:
                    if all(state.status[p] == "completed" for p in state.nodes[j].preds):
                        self._enqueue(state, j)
        if released:
            self.resources.rebuild()
        delivered = [key for key, time in self.pending_deliveries.items()
                     if time <= self.current_time]
        for dag_id, i in delivered:
            del self.pending_deliveries[dag_id, i]
            state = self.dag_states[dag_id]
            if state.terminal_time is None and all(
                state.status[j] == "completed" and (dag_id, j) not in self.pending_deliveries
                for j in state.dag.exit_ids
            ):
                # 上传、计算、回传的浮点求和允许恰好在截止时刻交付。
                on_time = self.current_time <= state.deadline + 8 * ulp(state.deadline)
                self._finish_dag(state, on_time, changed)

    def _next_ready(self, changed):
        while True:
            while self.ready:
                _, dag_id, i = heappop(self.ready)
                state = self.dag_states[dag_id]
                if state.status[i] == "ready":
                    self.current_dag = state.dag
                    self.current_subtask = state.nodes[i]
                    return True
            events = [r.release_time for r in self.active_reservations.values()]
            events.extend(self.pending_deliveries.values())
            if self.slot >= self.cfg.episode_slots and not events:
                return False
            boundary = (self.slot + 1) * self.cfg.slot_seconds
            self.current_time = min(boundary, min(events, default=boundary))
            self.concurrent.clear()  # 时钟已推进，旧组的 f、FT 和扰动永久冻结。
            at_boundary = self.current_time == boundary
            if at_boundary:
                if self.slot < self.cfg.episode_slots:
                    sample = self.resources.load_std()
                    self.slot_load_samples.append(sample)
                    self.sampled_load += float(sample @ self.cfg.load_balance_chip_weights)
                self.slot += 1
                self.slot_link_used.fill(0)
                self._update_topology()
            self._dag_completions(changed)
            if at_boundary:
                self._dag_arrivals()

    def _dag_flow_requests(self, node):
        """按前驱编号、出口的顺序列出流，节点位置与芯片选择无关。"""
        dag, task = self.current_dag, self.current_subtask
        assignments = self.dag_states[dag.dag_id].assignments
        requests = [(p, assignments[p].action.destination, node,
                     dag.edge_data_mb[p, task.subtask_id]) for p in sorted(task.preds)]
        if not task.preds:
            requests.append((-1, dag.access_edge, node, dag.input_data_mb))
        if task.subtask_id in dag.exit_ids:
            requests.append((-2, node, dag.access_edge, dag.output_mb))
        return requests

    def _dag_route_bundles(self, requests, link_capacity):
        """只枚举最大跨节点流；其余流依次使用该组合剩余容量允许的最快路径。"""
        options = []
        cross = []
        for j, (_, source, target, mb) in enumerate(requests):
            if source == target:
                options.append([(source,)])
                continue
            cross.append(j)
            hops = self.cfg.max_hops + int(self.cloud in (source, target))
            paths = self._routes(source, target, hops)
            link_delay = self.prop + np.divide(
                mb * self.cfg.slot_seconds + .02 * self.slot_link_used, link_capacity,
                out=np.zeros_like(link_capacity), where=link_capacity > 0)
            paths.sort(key=lambda p: (sum(link_delay[a, b] for a, b in zip(p, p[1:])), p))
            options.append(paths or [()])
        if not cross:
            yield [paths[0] for paths in options], True, 0
            return
        main = max(cross, key=lambda j: requests[j][3])
        for main_path in options[main][:self.cfg.k_routes_per_destination]:
            chosen = [paths[0] for paths in options]
            used = defaultdict(float)
            feasible = True
            for j in [main] + [j for j in cross if j != main]:
                mb = requests[j][3]
                paths = [main_path] if j == main else options[j]
                for path in paths:
                    if path and all(self.slot_link_used[a, b] + used[a, b] + mb
                                    <= link_capacity[a, b] for a, b in zip(path, path[1:])):
                        chosen[j] = path
                        for arc in zip(path, path[1:]):
                            used[arc] += mb
                        break
                else:
                    feasible = False
                    break
            yield chosen, feasible, main

    def _dag_transport(self, flows, background):
        dag = self.current_dag
        assignments = self.dag_states[dag.dag_id].assignments
        incoming = outgoing = energy = 0.
        for pred, path, tier, mb in flows:
            delay = 0.
            if tier == 1:
                delay = self.cfg.pcie_dma_base_latency_s + 8 * mb / self.cfg.pcie_dma_bandwidth_mbps
                energy += mb * self.cfg.pcie_dma_energy_per_mb_j
            for a, b in zip(path, path[1:]):
                rate = float(self.rates[a, b]) * (1 - float(background[a, b]))
                transmit = 8 * mb / rate
                queue = .02 * self.slot_link_used[a, b] / (rate * self.cfg.slot_seconds / 8)
                delay += transmit + queue + self.prop[a, b]
                energy += self.forward_power[a, b] * (transmit + queue)
            if pred == -2:
                downlink = 8 * mb / self.radio_rates[dag.source]
                outgoing = delay + downlink
                energy += self.cfg.uplink_circuit_power_w * downlink
            else:
                ready_at = assignments[pred].action.finish_time_s if pred >= 0 else self.current_time
                incoming = max(incoming, max(self.current_time, ready_at) - self.current_time + delay)
        return Transport(incoming, outgoing, energy)

    def _dag_split(self, action):
        node, chip = action.destination, action.resource
        group = [a for a in self.concurrent.get((node, chip), ())
                 if self.dag_states[a.action.dag_id].terminal_time is None]
        capacity = self.resources.free_compute[node, chip] + sum(a.reservation.compute_f for a in group)
        actions = [a.action for a in group] + [action]
        shares = allocate_compute(
            [a.effective_work for a in actions],
            [self.dag_states[a.dag_id].dag.priority for a in actions],
            capacity, [a.minimum_f for a in actions], self.cfg.compute_priority_exponent)
        return group, shares

    def _dag_times(self, action, transport, degradation):
        dag = self.dag_states[action.dag_id].dag
        uplink = 8 * dag.input_data_mb / action.uplink_rate if action.uplink_rate else 0.
        radio_energy = (action.uplink_power + self.cfg.uplink_circuit_power_w) * uplink
        start = max(action.context_ready_at, action.dispatch_time_s + uplink
                    + transport.incoming + action.setup_delay_s)
        duration = action.effective_work / (action.compute_share * (1 - degradation))
        finish = start + duration
        energy = (transport.energy + radio_energy + action.setup_energy_j
                  + self.cfg.chip_compute_power_w[action.resource] * duration)
        return start, finish, finish + transport.outgoing, energy

    def _dag_estimate(self, action, share):
        state = self.dag_states[action.dag_id]
        dag, task = state.dag, state.nodes[action.subtask_id]
        action.compute_share = float(share)
        if not task.preds:
            time_for_upload = (state.deadline - action.dispatch_time_s
                               - action.transport_delay - action.setup_delay_s
                               - action.effective_work / share)
            rate = min(self.radio_rates[dag.source], 8 * dag.input_data_mb / time_for_upload)
            c = self.cfg
            solution = choose_uplink_power(
                c.radio_bandwidth_mhz, self.radio_channel_gain[dag.source],
                c.uplink_circuit_power_w, c.uplink_max_power_w, rate,
                c.dinkelbach_tolerance, c.dinkelbach_max_iterations)
            action.uplink_power = solution.power_w
            action.uplink_rate = solution.rate_mbps
            action.dinkelbach_residual = solution.residual
        start, finish, delivered, energy = self._dag_times(action, action.transport, 0.)
        action.start_time_s, action.finish_time_s, action.delivery_time_s = start, finish, delivered
        action.delay = action.estimated_delay = delivered - action.dispatch_time_s
        action.energy = action.estimated_energy = energy

    def _dag_candidate(self, node, chip, flows, path, route_ok):
        dag, task = self.current_dag, self.current_subtask
        pool = self.resources
        action = DAGAction(node, chip, path, 0., 0., 0., dag_id=dag.dag_id,
                           subtask_id=task.subtask_id, flows=flows,
                           comm_tier=max(f.comm_tier for f in flows),
                           dispatch_time_s=self.current_time)
        request = ActiveReservation.for_subtask(task, node, chip, dag.storage_gb)
        affinity = self.cfg.operator_affinity_matrix[task.op_type][chip]
        if not (route_ok and dag.allow_offload and self.z_service[node, dag.kind]
                and chip in task.compatible_chips and affinity > 0 and pool.can_allocate(request)):
            return action, False
        if chip != CPU and pool.resident_op[node, chip] != task.op_type:
            action.cold_switch = True
            if chip == FPGA:
                action.setup_delay_s = self.cfg.fpga_reconfig_delay_s
                action.setup_energy_j = self.cfg.fpga_reconfig_energy_j
            else:
                action.setup_delay_s = self.cfg.npu_context_load_delay_s
                action.setup_energy_j = self.cfg.npu_context_load_energy_j
        else:
            action.context_ready_at = pool.context_ready_at[node, chip]
        transport = self._dag_transport(flows, self.background)
        action.transport = transport
        action.transport_delay = transport.incoming + transport.outgoing
        action.transport_energy = transport.energy
        peak_upload = 0. if task.preds else 8 * dag.input_data_mb / self.radio_rates[dag.source]
        earliest = max(action.context_ready_at, self.current_time + peak_upload
                       + transport.incoming + action.setup_delay_s)
        deadline = self.dag_states[dag.dag_id].deadline
        window = deadline - transport.outgoing - earliest
        if window <= 0.:
            return action, False
        action.effective_work = task.compute_gflop / affinity
        # 相减消去固定开销时的舍入误差不能把恰好用满容量的 KKT 组屏蔽。
        action.minimum_f = action.effective_work / (window + 2 * ulp(deadline))
        _, shares = self._dag_split(action)
        if shares is None:
            return action, False
        self._dag_estimate(action, shares[-1])
        return action, True

    def _dag_candidates(self):
        dag = self.current_dag
        placements = self.dag_states[dag.dag_id].assignments
        actions, mask = [], []
        capacity = self.rates * (1 - self.background.astype(float)) * self.cfg.slot_seconds / 8
        for node in range(self.n_graph_nodes):
            requests = self._dag_flow_requests(node)
            for paths, route_ok, main in self._dag_route_bundles(requests, capacity):
                for chip in (CPU, NPU, FPGA):
                    flows = []
                    for (pred, source, target, mb), path in zip(requests, paths):
                        tier = 2 if source != target else int(
                            pred >= 0 and placements[pred].action.resource != chip)
                        flows.append(Flow(pred, path, tier, mb))
                    action, valid = self._dag_candidate(node, chip, tuple(flows), paths[main], route_ok)
                    actions.append(action)
                    mask.append(valid)
        actions.append(DAGAction(-2, -1, (), 0., 0., 0., dag_id=dag.dag_id,
                                 subtask_id=self.current_subtask.subtask_id))
        return actions, np.array(mask + [True], dtype=bool)


    def _dag_dispatch(self, action, changed):
        dag, task = self.current_dag, self.current_subtask
        group, shares = self._dag_split(action)
        # 同弧只抽一次实际拥塞，所有流共同检查的 MB 在本次承诺中一次入账。
        budget = defaultdict(float)
        for flow in action.flows:
            for arc in zip(flow.path, flow.path[1:]):
                budget[arc] += flow.data_mb
        background = self.background.astype(float)
        for arc in budget:
            noise = self.cfg.congestion_noise * (.5 + background[arc])
            background[arc] = np.clip(background[arc] + self.execution_rng.uniform(-noise, noise),
                                      0., self.cfg.congestion_realized_max)
        actual = self._dag_transport(action.flows, background)
        degradation = float(self.execution_rng.uniform(0., self.cfg.compute_degradation_max))
        r = ActiveReservation.for_subtask(task, action.destination, action.resource, dag.storage_gb)
        new = DAGAssignment(action, r, degradation, actual)
        self.active_reservations[dag.dag_id, task.subtask_id] = r
        for assignment, share in zip([*group, new], shares):
            a = assignment.action
            self._dag_estimate(a, share)
            start, finish, delivered, energy = self._dag_times(
                a, assignment.actual_transport, assignment.compute_degradation)
            a.start_time_s, a.finish_time_s, a.delivery_time_s = start, finish, delivered
            a.delay, a.energy = delivered - a.dispatch_time_s, energy
            if a.cold_switch:
                self.resources.resident_op[a.destination, a.resource] = assignment.reservation.op_type
                self.resources.context_ready_at[a.destination, a.resource] = start
            assignment.reservation.compute_f = a.compute_share
            assignment.reservation.release_time = finish
            changed.add(a.dag_id)
        self.resources.rebuild()
        for arc, mb in budget.items():
            self.slot_link_used[arc] += mb
        self.concurrent[action.destination, action.resource].append(new)
        self.assignments.append(new)
        state = self.dag_states[dag.dag_id]
        state.assignments[task.subtask_id] = new
        state.status[task.subtask_id] = "running"

    def _dag_balance(self, state):
        if not state.assignments:
            return (0., 0., 0.)
        if state.terminal_time is None:
            end = max(a.action.dispatch_time_s + a.action.estimated_delay + state.remaining_s[i]
                      for i, a in state.assignments.items())
            energy = sum(a.action.estimated_energy for a in state.assignments.values())
        else:
            end = state.terminal_time
            energy = sum(a.action.energy for a in state.assignments.values())
        c = self.cfg
        return (c.reward_beta_completion * state.success,
                -c.reward_beta_delay * (end - state.arrival) / c.reward_delay_ref,
                -c.reward_beta_energy * energy / c.reward_energy_ref)

    def _load_ledger(self):
        current = (self.resources.load_std() @ self.cfg.load_balance_chip_weights
                   if self.slot < self.cfg.episode_slots else 0.)
        return self.sampled_load + current

    def _step_dag(self, action_id):
        feasible = int(self.action_mask[:-1].sum())
        invalid = not 0 <= action_id < len(self.actions) or not self.action_mask[action_id]
        action = self.actions[-1] if invalid else self.actions[action_id]
        accepted = action.destination != -2
        state = self.dag_states[self.current_dag.dag_id]
        changed = set()
        missed_before = self.missed_dags
        load_before = self._load_ledger()
        self.feasible_tasks += bool(feasible)
        self.action_count_sum += len(self.actions)
        self.idx += 1
        penalty = 0.
        if accepted:
            self._dag_dispatch(action, changed)
        else:
            if invalid:
                penalty = self.cfg.reward_penalty_invalid
            elif feasible:
                self.conditional_rejects += 1
                penalty = self.cfg.reward_penalty_conditional_reject
            else:
                self.forced_rejects += 1
            self._finish_dag(state, False, changed)
        done = not self._next_ready(changed)
        penalty += (self.missed_dags - missed_before) * self.cfg.reward_penalty_incomplete
        parts = np.zeros(3)
        for dag_id in sorted(changed):
            item = self.dag_states[dag_id]
            balance = self._dag_balance(item)
            parts += np.subtract(balance, item.balance)
            item.balance = balance
        load_part = (-self.cfg.reward_beta_load * (self._load_ledger() - load_before)
                     / self.cfg.reward_load_std_ref_v9)
        reward = float(parts.sum() + load_part - penalty)
        self.total_reward += reward
        info = dict(dag_id=action.dag_id, subtask_id=action.subtask_id,
                    accepted=accepted, completed=state.success, destination=action.destination,
                    resource=action.resource, path=action.path, flows=action.flows,
                    comm_tier=action.comm_tier, delay=action.delay, energy=action.energy,
                    estimated_delay=action.estimated_delay, estimated_energy=action.estimated_energy,
                    setup_delay_s=action.setup_delay_s, setup_energy_j=action.setup_energy_j,
                    start_time_s=action.start_time_s, finish_time_s=action.finish_time_s,
                    delivery_time_s=action.delivery_time_s, p_up=action.uplink_power,
                    f=action.compute_share, feasible_count=feasible, invalid_action=bool(invalid),
                    reward_completion_part=parts[0], reward_delay_part=parts[1],
                    reward_energy_part=parts[2], reward_load_part=load_part,
                    reward_constraint_part=-penalty)
        return None if done else self.observe(), reward, done, info

    def _dag_network_nodes(self, edges):
        pool = self.resources
        free = {key: np.divide(value, pool.capacity[key], out=np.zeros_like(value),
                               where=pool.capacity[key] > 0) for key, value in pool.free.items()}
        speed = np.divide(pool.free_compute, pool.speed, out=np.zeros_like(pool.speed), where=pool.speed > 0)
        queues = np.divide(pool.queues, pool.queue_cap[:, None], out=np.zeros_like(speed),
                           where=pool.queue_cap[:, None] > 0)
        neighbors = self.rates > 0
        congestion = (edges[:, :, 1] * neighbors).sum(axis=1) / np.maximum(neighbors.sum(axis=1), 1)
        # 驻留编码 0 为空上下文，1..5 对应五种算子。
        resident = (pool.resident_op + 1) / len(OP_NAMES)
        return np.column_stack((
            free["cpu_cores"], free["ram_gb"], speed[:, CPU], queues[:, CPU],
            speed[:, NPU], free["npu_vram_gb"], queues[:, NPU],
            free["fpga_lut"], free["fpga_ff"], free["fpga_bram"], free["fpga_dsp"],
            speed[:, FPGA], queues[:, FPGA], resident[:, FPGA], resident[:, NPU],
            free["storage_gb"], congestion, np.arange(self.n_graph_nodes) == self.cloud,
        )).astype(np.float32)

    def _observe_dag(self):
        dag, task = self.current_dag, self.current_subtask
        state = self.dag_states[dag.dag_id]
        self.actions, self.action_mask = self._dag_candidates()
        edges = self._edge_features()
        gat_adj = self.adj.copy()
        gat_adj[:self.cloud, self.cloud] = gat_adj[self.cloud, :self.cloud] = 0
        indices = {t.subtask_id: i for i, t in enumerate(dag.subtasks)}
        dag_nodes = np.zeros((6, 15 + len(DEMAND_FIELDS)), dtype=np.float32)
        dag_adj = np.zeros((6, 6), dtype=np.float32)
        dag_edges = np.zeros((6, 6, 1), dtype=np.float32)
        status_column = {"waiting": 11, "ready": 12, "running": 13, "completed": 14}
        caps = [self.resources.capacity[key].max() for key in DEMAND_FIELDS]
        for i, subtask in enumerate(dag.subtasks):
            row = dag_nodes[i]
            row[subtask.op_type] = 1.
            row[5] = subtask.compute_gflop / dag.compute_gflop
            row[6:9] = self.cfg.operator_affinity_matrix[subtask.op_type]
            row[9:11] = (subtask.rank_u / dag.deadline_s, subtask.slack_s / dag.deadline_s)
            row[status_column[state.status[subtask.subtask_id]]] = 1.
            row[15:] = [getattr(subtask, key) / cap if cap else 0.
                        for key, cap in zip(DEMAND_FIELDS, caps)]
            dag_adj[i, i] = 1
        for (parent, child), mb in dag.edge_data_mb.items():
            i, p = indices[child], indices[parent]
            dag_adj[i, p] = 1
            dag_edges[i, p, 0] = mb / dag.input_data_mb

        targets = np.array([dag.access_edge if a.destination == -2 else a.destination
                            for a in self.actions], dtype=np.int64)
        paths = np.zeros((len(targets), self.n_graph_nodes), dtype=np.float32)
        features = np.zeros((len(targets), 12), dtype=np.float32)
        op_affinities = self.cfg.operator_affinity_matrix[task.op_type]
        max_affinity = max(max(row) for row in self.cfg.operator_affinity_matrix)
        max_setup = max(self.cfg.fpga_reconfig_delay_s, self.cfg.npu_context_load_delay_s, 1e-6)
        abs_deadline = state.arrival + dag.deadline_s
        for k, (target, a) in enumerate(zip(targets, self.actions)):
            visits = [m for f in a.flows for m in f.path] or [target]
            paths[k] = np.bincount(visits, minlength=self.n_graph_nodes) / len(visits)
            hops = max(0, len(a.path) - 1 - int(self.cloud in a.path))
            cap = self.resources.queue_cap[target]
            affinity = 0. if a.destination == -2 else op_affinities[a.resource] / max_affinity
            setup_ratio = 0. if a.destination == -2 else a.setup_delay_s / max_setup
            slack_ratio = (0. if a.destination == -2 else
                           float(np.clip((abs_deadline - a.delivery_time_s) / dag.deadline_s, -1., 1.)))
            features[k] = (a.destination == -2, affinity, target == self.cloud,
                           a.resource == CPU, a.resource == NPU, a.resource == FPGA,
                           hops / max(1, self.cfg.max_hops), a.delay / self.cfg.reward_delay_ref,
                           a.energy / 10, self.resources.queues[target].sum() / cap if cap else 0.,
                           setup_ratio, slack_ratio)
        task_vector = np.array((dag.input_data_mb / 8, dag.output_mb, task.compute_gflop / 10,
                                dag.deadline_s / self.cfg.reward_delay_ref, dag.priority / 5,
                                task.ram_gb / 2, dag.kind / 4, dag.allow_offload), dtype=np.float32)
        # 全局量只取当前已经到达/派发的信息。
        global_features = np.array((
            self.slot / self.cfg.episode_slots, self.current_time - state.arrival,
            len(state.assignments) / len(dag.subtasks), state.arrival + dag.deadline_s - self.current_time,
            *self.resources.load_std(), len(self.active_reservations), self.missed_dags,
        ), dtype=np.float32)
        return dict(nodes=self._dag_network_nodes(edges), edges=edges, adj=self.adj.copy(),
                    gat_adj=gat_adj, adjacency=gat_adj.copy(), task=task_vector,
                    source=dag.access_edge, source_node=dag.access_edge, candidate_features=features,
                    candidate_targets=targets, candidate_paths=paths, global_features=global_features,
                    dag_nodes=dag_nodes, dag_adj=dag_adj, dag_edges=dag_edges,
                    dag_node_mask=np.arange(6) < len(indices),
                    current_subtask_idx=indices[task.subtask_id], action_mask=self.action_mask.copy())

    def _dag_metrics(self):
        states = self.dag_states.values()
        count = len(states)
        successful = [s for s in states if s.success]
        accepted = sum(bool(s.assignments) for s in states)
        makespans = [s.terminal_time - s.arrival for s in successful]
        energies = [sum(a.action.energy for a in s.assignments.values()) for s in successful]
        mean_time = float(np.mean(makespans)) if successful else None
        mean_energy = float(np.mean(energies)) if successful else None
        std = np.mean(self.slot_load_samples, axis=0) if self.slot_load_samples else np.zeros(3)
        load = float(std @ self.cfg.load_balance_chip_weights)
        actions = [a.action for a in self.assignments]
        fpga = [a for a in actions if a.resource == FPGA]
        tiers = [f.comm_tier for a in actions for f in a.flows if f.pred_id >= 0]
        hops = [max(0, len(a.path) - 1 - int(self.cloud in a.path)) for a in actions]
        fpga_time = sum(a.finish_time_s - a.dispatch_time_s for a in fpga)
        priorities = [s.dag.priority for s in successful]
        miss_rate = self.missed_dags / accepted if accepted else 0.
        return dict(
            tasks=count, task_success_rate=len(successful) / count, dag_completion_rate=len(successful) / count,
            acceptance_rate=accepted / count, deadline_miss_rate=miss_rate, deadline_violation_rate=miss_rate,
            avg_delay_s=mean_time, avg_dag_makespan_s=mean_time, avg_energy_j=mean_energy,
            avg_dag_energy_j=mean_energy, total_energy_j=sum(a.energy for a in actions),
            priority_weighted_delay_s=float(np.average(makespans, weights=priorities)) if successful else None,
            mean_load_variance=load, load_balance_degree=1 / (1 + load), hetero_load_balance_degree=1 / (1 + load),
            cpu_load_std=float(std[CPU]), npu_load_std=float(std[NPU]), fpga_load_std=float(std[FPGA]),
            fpga_reconfig_rate=sum(a.cold_switch for a in fpga) / max(1, len(fpga)),
            fpga_reconfig_delay_ratio=sum(a.setup_delay_s for a in fpga) / fpga_time if fpga_time else 0.,
            comm_tier0_ratio=tiers.count(0) / max(1, len(tiers)),
            comm_tier1_ratio=tiers.count(1) / max(1, len(tiers)),
            comm_tier2_ratio=tiers.count(2) / max(1, len(tiers)),
            episode_reward=self.total_reward, objective_score=self.total_reward / count,
            reward_sum_check=self.total_reward / count, feasible_task_rate=self.feasible_tasks / max(1, self.idx),
            forced_reject_rate=self.forced_rejects / max(1, self.idx),
            conditional_reject_rate=self.conditional_rejects / max(1, self.feasible_tasks),
            mean_action_count=self.action_count_sum / max(1, self.idx), local_execution_rate=0.,
            edge_execution_rate=sum(a.destination < self.cloud for a in actions) / max(1, self.idx),
            cloud_execution_rate=sum(a.destination == self.cloud for a in actions) / max(1, self.idx),
            mean_hops_accepted=float(np.mean(hops)) if actions else None,
            mean_path_length=float(np.mean([len(a.path) for a in actions])) if actions else None,
        )


def shuffle_graph_observation(obs, permutation):
    shuffled = dict(obs)
    shuffled["adjacency"] = obs["adjacency"][permutation][:, permutation]
    shuffled["edges"] = obs["edges"][permutation][:, permutation]
    return shuffled


class ShuffledGraphEnv:
    """保持真实动作与约束不变，只扰动策略看到的图连接。"""

    def __init__(self, env, seed):
        self.env = env
        edge_nodes = np.random.default_rng(seed).permutation(env.cloud)
        self.permutation = np.append(edge_nodes, env.cloud)

    @property
    def rng(self):
        return self.env.rng

    def reset(self):
        return shuffle_graph_observation(self.env.reset(), self.permutation)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            obs = shuffle_graph_observation(obs, self.permutation)
        return obs, reward, done, info

    def metrics(self):
        return self.env.metrics()


def model_environment(env, model_type, seed):
    if model_type == "shuffled_gat":
        return ShuffledGraphEnv(env, seed)
    return env
