"""云–边–端多跳异构调度环境：在线任务、联合节点/路径/资源动作。"""
from dataclasses import dataclass
import numpy as np

from .config import Config
from .tasks import sample_tasks
from .topology import make_topology, simple_routes
from .continuous import allocate_compute, choose_uplink_power


@dataclass
class Action:
    destination: int  # -2: 拒绝, -1: 端侧本地, 0..M-1: 边缘, M: 云
    resource: int     # 0: CPU, 1: NPU, 2: FPGA
    path: tuple[int, ...]
    delay: float
    energy: float
    compute_share: float
    transport_delay: float = 0.0
    transport_energy: float = 0.0
    uplink_power: float = 0.0
    uplink_rate: float = 0.0


@dataclass
class Assignment:
    task_index: int
    action: Action


class EdgeEnv:
    """一个 step 调度一项任务，每 tasks_per_slot 项任务共享一个调度时隙。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.cloud = cfg.n_edges
        self.n_graph_nodes = cfg.n_edges + 1
        self.speed = np.tile(np.array([80., 120., 50.], dtype=np.float32), (cfg.n_edges, 1))
        self.speed = np.vstack((self.speed, [2000., 6000., 2000.])).astype(np.float32)
        self.mem_cap = np.array([32.] * cfg.n_edges + [128.], dtype=np.float32)
        self.storage_cap = np.array([1000.] * cfg.n_edges + [4000.], dtype=np.float32)
        self.queue_cap = np.array([60.] * cfg.n_edges + [80.], dtype=np.float32)
        self.local_speed = np.array([12., 16., 6.], dtype=np.float32)
        self.local_mem_cap = np.full(cfg.n_terminals, 8.0, dtype=np.float32)
        self.local_storage_cap = np.full(cfg.n_terminals, 128.0, dtype=np.float32)
        self.local_queue_cap = np.full(cfg.n_terminals, 12.0, dtype=np.float32)
        self.resource_power = np.array([10., 18., 12.], dtype=np.float32)
        self.support = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool)
        self.reset()

    def reset(self):
        c = self.cfg
        self.tasks = sample_tasks(self.rng, c.n_tasks, c.n_terminals, c.n_edges)
        self.radio_rates = self.rng.uniform(c.radio_rate_min_mbps, c.radio_rate_max_mbps, c.n_terminals)
        self.radio_channel_gain = (
            2.0 ** (self.radio_rates / c.radio_bandwidth_mhz) - 1.0
        ) / c.uplink_max_power_w
        self._new_topology()
        self.idx = 0
        self.slot = 0
        self.slot_resource_used = np.zeros_like(self.speed)
        self.slot_local_resource_used = np.zeros((c.n_terminals, 3), dtype=np.float32)
        self.slot_local_mem_used = np.zeros(c.n_terminals, dtype=np.float32)
        self.slot_local_storage_used = np.zeros(c.n_terminals, dtype=np.float32)
        self.slot_local_queues = np.zeros(c.n_terminals, dtype=np.float32)
        self.slot_mem_used = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_storage_used = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_queues = np.zeros(self.n_graph_nodes, dtype=np.float32)
        self.slot_link_used = np.zeros_like(self.rates)
        self.slot_assignments = []
        self.accepted = 0
        self.sum_delay = 0.
        self.sum_priority_delay = 0.
        self.sum_priority = 0.
        self.total_energy = 0.
        self.total_reward = 0.
        self.total_hops = 0
        self.total_load_variance = 0.
        self.load_samples = 0
        return self.observe()

    def _new_topology(self):
        c = self.cfg
        self.adj, self.rates, self.background, self.prop = make_topology(
            self.rng, c.n_edges, c.undirected_edge_links,
            (c.link_rate_min_mbps, c.link_rate_max_mbps),
            (c.cloud_rate_min_mbps, c.cloud_rate_max_mbps),
            (c.congestion_min, c.congestion_max),
        )
        self.forward_power = np.zeros_like(self.rates)
        for a in range(self.n_graph_nodes):
            for b in range(self.n_graph_nodes):
                if a != b and self.adj[a, b]:
                    self.forward_power[a, b] = self.rng.uniform(2., 8.)

    def _slot_capacity_mb(self, a, b):
        return self.rates[a, b] * (1 - self.background[a, b]) * self.cfg.slot_seconds / 8

    def _load_variance(self):
        remote_load = self.slot_resource_used.sum(1) / self.speed.sum(1)
        local_load = self.slot_local_resource_used.sum(1) / self.local_speed.sum()
        load = np.concatenate((local_load, remote_load))
        return float(np.var(load))

    def _local_support(self, task, resource):
        return bool(self.support[resource, task.kind] and
                    (resource == 0 or resource == 1 and task.source % 2 == 0 or
                     resource == 2 and task.source % 3 == 0))

    def _same_compute_group(self, task, destination, resource):
        group = []
        for assignment in self.slot_assignments:
            other_task = self.tasks[assignment.task_index]
            action = assignment.action
            same_local = destination == -1 and action.destination == -1 and other_task.source == task.source
            same_remote = destination >= 0 and action.destination == destination
            if action.resource == resource and (same_local or same_remote):
                group.append(assignment)
        return group

    def _transport_terms(self, task, destination, path):
        delay = task.output_mb * 8 / self.radio_rates[task.source]
        energy = self.cfg.uplink_circuit_power_w * delay
        for a, b in zip(path, path[1:]):
            if task.data_mb + self.slot_link_used[a, b] > self._slot_capacity_mb(a, b):
                return None
            if task.output_mb + self.slot_link_used[b, a] > self._slot_capacity_mb(b, a):
                return None
            upload = task.data_mb * 8 / (self.rates[a, b] * (1 - self.background[a, b]))
            upload_queue = .02 * self.slot_link_used[a, b] / self._slot_capacity_mb(a, b)
            download = task.output_mb * 8 / (self.rates[b, a] * (1 - self.background[b, a]))
            download_queue = .02 * self.slot_link_used[b, a] / self._slot_capacity_mb(b, a)
            delay += upload + upload_queue + self.prop[a, b]
            delay += download + download_queue + self.prop[b, a]
            energy += self.forward_power[a, b] * (upload + upload_queue)
            energy += self.forward_power[b, a] * (download + download_queue)
        delay += .005 * self.slot_queues[destination]
        return delay, energy

    def _completion(self, task, resource, share, transport_delay, transport_energy, local):
        compute_delay = task.compute_gflop / share
        compute_energy = self.resource_power[resource] * compute_delay
        if local:
            if compute_delay > task.deadline_s:
                return None
            return compute_delay, compute_energy, 0.0, 0.0
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
        return delay, energy, solution.power_w, solution.rate_mbps

    def _group_allocation(self, task, destination, resource, transport_delay, transport_energy):
        group = self._same_compute_group(task, destination, resource)
        group_tasks = [self.tasks[assignment.task_index] for assignment in group] + [task]
        transport_delays = [assignment.action.transport_delay for assignment in group] + [transport_delay]
        capacity = self.local_speed[resource] if destination == -1 else self.speed[destination, resource]
        minimum_shares = []
        for item, fixed_delay in zip(group_tasks, transport_delays):
            if destination >= 0:
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
            outcome = self._completion(
                other_task, resource, share, action.transport_delay,
                action.transport_energy, destination == -1,
            )
            if outcome is None:
                return None
            outcomes.append((assignment, float(share), outcome))
        candidate = self._completion(
            task, resource, shares[-1], transport_delay, transport_energy, destination == -1,
        )
        if candidate is None:
            return None
        return outcomes, float(shares[-1]), candidate

    def _candidate(self, task, destination, resource, path):
        local = destination == -1
        if local:
            source = task.source
            if self.slot_local_mem_used[source] + task.memory_gb > self.local_mem_cap[source]:
                return None
            if self.slot_local_storage_used[source] + task.storage_gb > self.local_storage_cap[source]:
                return None
            if self.slot_local_queues[source] + 1 > self.local_queue_cap[source]:
                return None
            transport_delay, transport_energy = 0.0, 0.0
        else:
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
        delay, energy, power, rate = completion
        return Action(
            destination, resource, path, delay, energy, share,
            transport_delay, transport_energy, power, rate,
        )

    def _make_candidates(self):
        task = self.tasks[self.idx]
        actions = []
        for q in range(3):
            if self._local_support(task, q):
                action = self._candidate(task, -1, q, ())
                if action is not None:
                    actions.append(action)
        if task.allow_offload:
            source = task.access_edge
            for destination in range(self.cfg.n_edges):
                # 同一接入边缘服务器不需要边缘间多跳。
                routes = [(source,)] if destination == source else simple_routes(
                    self.adj, source, destination, self.cfg.max_hops)
                routes.sort(key=lambda p: sum(
                    8 * task.data_mb / (self.rates[a, b] * (1 - self.background[a, b]))
                    for a, b in zip(p, p[1:])))
                for path in routes[:self.cfg.k_routes_per_destination]:
                    # 部分边缘节点未部署该类服务，用于模拟服务可用约束。
                    if (destination + task.kind) % 7 == 0:
                        continue
                    for q in range(3):
                        if self.support[q, task.kind]:
                            action = self._candidate(task, destination, q, path)
                            if action is not None:
                                actions.append(action)
            # 云端经接入边缘节点的广域链路转发。
            for q in range(3):
                if self.support[q, task.kind]:
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
        edge = np.zeros((n, n, 3), dtype=np.float32)
        edge[:, :, 0] = self.rates / 2000
        edge[:, :, 1] = self.background + self.slot_link_used / np.maximum(1., self.rates / 8)
        edge[:, :, 2] = self.prop / .003
        return node, edge

    def observe(self):
        task = self.tasks[self.idx]
        self.actions = self._make_candidates()
        node, edge = self._graph_features()
        cand = np.zeros((len(self.actions), 10), dtype=np.float32)
        targets = np.empty(len(self.actions), dtype=np.int64)
        paths = np.zeros((len(self.actions), self.n_graph_nodes), dtype=np.float32)
        for a, action in enumerate(self.actions):
            local = action.destination == -1
            reject = action.destination == -2
            targets[a] = task.access_edge if action.destination < 0 else action.destination
            route = action.path or (task.access_edge,)
            paths[a, list(route)] = 1 / len(route)
            cand[a, :3] = [float(reject), float(local), float(action.destination == self.cloud)]
            cand[a, 3:6] = [float(action.resource == q) for q in range(3)]
            edge_hops = max(0, len(route) - 1) if 0 <= action.destination < self.cloud else 0
            cand[a, 6] = edge_hops / self.cfg.max_hops
            cand[a, 7] = action.delay / .5
            cand[a, 8] = action.energy / 10
            if local:
                cand[a, 9] = self.slot_local_queues[task.source] / self.local_queue_cap[task.source]
            else:
                cand[a, 9] = self.slot_queues[targets[a]] / self.queue_cap[targets[a]]
        task_vector = np.array([
            task.data_mb / 8, task.output_mb, task.compute_gflop / 10,
            task.deadline_s / .5, task.priority / 5,
            task.memory_gb / 2, task.kind / 4, float(task.allow_offload),
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
        }

    def step(self, action_id: int):
        task = self.tasks[self.idx]
        action = self.actions[action_id]
        accepted = action.destination != -2
        if accepted:
            allocation = self._group_allocation(
                task, action.destination, action.resource,
                action.transport_delay, action.transport_energy,
            )
            group_outcomes, candidate_share, candidate_outcome = allocation
            for assignment, share, outcome in group_outcomes:
                other_task = self.tasks[assignment.task_index]
                previous_delay = assignment.action.delay
                previous_energy = assignment.action.energy
                delay, energy, power, rate = outcome
                self.sum_delay += delay - previous_delay
                self.sum_priority_delay += other_task.priority * (delay - previous_delay)
                self.total_energy += energy - previous_energy
                assignment.action.compute_share = share
                assignment.action.delay = delay
                assignment.action.energy = energy
                assignment.action.uplink_power = power
                assignment.action.uplink_rate = rate
            action.compute_share = candidate_share
            action.delay, action.energy, action.uplink_power, action.uplink_rate = candidate_outcome
            self.accepted += 1
            self.sum_delay += action.delay
            self.sum_priority_delay += task.priority * action.delay
            self.sum_priority += task.priority
            self.total_energy += action.energy
            if 0 <= action.destination < self.cloud:
                self.total_hops += max(0, len(action.path) - 1)
            if action.destination == -1:
                source = task.source
                self.slot_local_resource_used[source, action.resource] = sum(
                    assignment.action.compute_share for assignment in self._same_compute_group(
                        task, action.destination, action.resource)
                ) + action.compute_share
                self.slot_local_mem_used[source] += task.memory_gb
                self.slot_local_storage_used[source] += task.storage_gb
                self.slot_local_queues[source] += 1
            else:
                j = action.destination
                self.slot_resource_used[j, action.resource] = sum(
                    assignment.action.compute_share for assignment in self._same_compute_group(
                        task, action.destination, action.resource)
                ) + action.compute_share
                self.slot_mem_used[j] += task.memory_gb
                self.slot_storage_used[j] += task.storage_gb
                self.slot_queues[j] += 1
                for a, b in zip(action.path, action.path[1:]):
                    self.slot_link_used[a, b] += task.data_mb
                    self.slot_link_used[b, a] += task.output_mb
            self.slot_assignments.append(Assignment(self.idx, action))
        processed_tasks = self.idx + 1
        task_completion_rate = self.accepted / processed_tasks
        average_completion_time = self.sum_delay / self.accepted if self.accepted else 0.0
        average_task_energy = self.total_energy / self.accepted if self.accepted else 0.0
        load_imbalance = (
            self.total_load_variance + self._load_variance()
        ) / (self.load_samples + 1)
        reward = (2.0 * task_completion_rate
                  - 2.0 * average_completion_time
                  - 1.0 * average_task_energy
                  - 1.0 * load_imbalance)
        self.total_reward += reward
        self.idx += 1
        done = self.idx == len(self.tasks)
        slot_finished = done or self.idx % self.cfg.tasks_per_slot == 0
        if slot_finished:
            self.total_load_variance += self._load_variance()
            self.load_samples += 1
        if not done and self.idx % self.cfg.tasks_per_slot == 0:
            self.slot += 1
            self.slot_resource_used.fill(0)
            self.slot_local_resource_used.fill(0)
            self.slot_local_mem_used.fill(0)
            self.slot_local_storage_used.fill(0)
            self.slot_local_queues.fill(0)
            self.slot_mem_used.fill(0)
            self.slot_storage_used.fill(0)
            self.slot_queues.fill(0)
            self.slot_link_used.fill(0)
            self.slot_assignments.clear()
            if self.slot % self.cfg.topology_period_slots == 0:
                self._new_topology()
            else:
                self.background = np.where(self.rates > 0,
                    self.rng.uniform(self.cfg.congestion_min, self.cfg.congestion_max,
                                     self.background.shape), 0).astype(np.float32)
        obs = None if done else self.observe()
        return obs, float(reward), done, {"accepted": accepted, "destination": action.destination,
                                          "resource": action.resource, "path": action.path,
                                          "delay": action.delay, "energy": action.energy,
                                          "f": action.compute_share,
                                          "p_up": action.uplink_power}

    def metrics(self):
        count = len(self.tasks)
        avg_delay = self.sum_delay / self.accepted if self.accepted else None
        priority_delay = self.sum_priority_delay / self.sum_priority if self.accepted else None
        avg_energy = self.total_energy / self.accepted if self.accepted else None
        mean_hops = self.total_hops / self.accepted if self.accepted else None
        load_variance = self.total_load_variance / self.load_samples
        return {
            "tasks": count,
            "task_success_rate": self.accepted / count,
            "avg_delay_s": avg_delay,
            "priority_weighted_delay_s": priority_delay,
            "total_energy_j": self.total_energy,
            "avg_energy_j": avg_energy,
            "mean_load_variance": load_variance,
            "load_balance_degree": 1.0 / (1.0 + load_variance),
            "mean_hops_accepted": mean_hops,
            "episode_reward": self.total_reward,
        }
