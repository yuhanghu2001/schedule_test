"""云–边–端多跳异构调度环境：在线任务、联合节点/路径/资源动作。"""
from dataclasses import dataclass
import numpy as np

from .config import Config
from .tasks import sample_tasks
from .topology import fluctuate_rates, fluctuate_topology, make_topology, simple_routes
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


class EdgeEnv:
    """一个 step 调度一项任务，任务按自身到达时隙推进环境。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.cloud = cfg.n_edges
        self.n_graph_nodes = cfg.n_edges + 1
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
        )
        self.radio_channel_gain = (
            2.0 ** (self.radio_rates / c.radio_bandwidth_mhz) - 1.0
        ) / c.uplink_max_power_w
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
            if self.slot % self.cfg.topology_period_slots == 0:
                self._new_topology()
            else:
                (self.rates, self.background, self.congestion_delta,
                 self.flapped_edges) = fluctuate_topology(
                    self.rng, self.cfg, self.adj, self.base_rates, self.base_background,
                    self.prop, self.forward_power, self.background, self.flapped_edges,
                )

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
            for destination in range(self.cfg.n_edges):
                if not self.service_deployment[destination, task.kind]:
                    continue
                # 同一接入边缘服务器不需要边缘间多跳。
                routes = [(source,)] if destination == source else simple_routes(
                    self.adj, source, destination, self.cfg.max_hops)
                routes.sort(key=lambda p: sum(
                    8 * task.data_mb / (self.rates[a, b] * (1 - self.background[a, b]))
                    for a, b in zip(p, p[1:])))
                for path in routes[:self.cfg.k_routes_per_destination]:
                    for q in range(3):
                        if self.support[q, task.kind]:
                            action = self._candidate(task, destination, q, path)
                            if action is not None:
                                actions.append(action)
            # 云端经接入边缘节点的广域链路转发。
            for q in range(3):
                if self.adj[source, self.cloud] and self.support[q, task.kind]:
                    action = self._candidate(task, self.cloud, q, (source, self.cloud))
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
        return node, edge

    def observe(self):
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
            edge_hops = max(0, len(route) - 1) if 0 <= action.destination < self.cloud else 0
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
            if 0 <= action.destination < self.cloud:
                self.total_hops += max(0, len(action.path) - 1)
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
