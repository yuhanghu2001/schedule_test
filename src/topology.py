from collections import deque
from itertools import combinations

import numpy as np


def _activate_link(rng, cfg, a, b, adj, base_rates, base_background, prop, forward_power):
    if b == cfg.n_edges:
        rate = rng.uniform(cfg.cloud_rate_min_mbps, cfg.cloud_rate_max_mbps)
        delay = rng.uniform(0.001, 0.003)
    else:
        rate = rng.uniform(cfg.link_rate_min_mbps, cfg.link_rate_max_mbps)
        delay = rng.uniform(0.0002, 0.002)
    adj[a, b] = adj[b, a] = 1
    base_rates[a, b] = base_rates[b, a] = rate
    prop[a, b] = prop[b, a] = delay
    base_background[a, b], base_background[b, a] = rng.uniform(
        cfg.congestion_min, cfg.congestion_max, 2,
    )
    forward_power[a, b], forward_power[b, a] = rng.uniform(2., 8., 2)


def make_topology(rng, cfg):
    """按环和随机弦构建周期基线，最后一个节点为云端。"""
    n = cfg.n_edges
    adj = np.eye(n + 1, dtype=np.float32)
    base_rates = np.zeros_like(adj)
    base_background = np.zeros_like(adj)
    prop = np.zeros_like(adj)
    forward_power = np.zeros_like(adj)
    pairs = set(zip(range(n - 1), range(1, n)))
    if n > 1:
        pairs.add((0, n - 1))
    chords = sorted(set(combinations(range(n), 2)) - pairs)
    if cfg.undirected_edge_links < len(pairs):
        raise ValueError("目标无向边数不足以构建边缘环")
    extra = min(cfg.undirected_edge_links - len(pairs), len(chords))
    pairs.update(chords[i] for i in rng.choice(len(chords), extra, replace=False))
    for a, b in sorted(pairs) + [(a, n) for a in range(n)]:
        _activate_link(rng, cfg, a, b, adj, base_rates, base_background, prop, forward_power)
    return adj, base_rates, base_background, prop, forward_power


def fluctuate_rates(rng, cfg, base_rates):
    rates = base_rates.copy()
    a, b = np.nonzero(np.triu(base_rates[:cfg.n_edges, :cfg.n_edges], k=1))
    jitter = rng.uniform(-cfg.link_rate_fluctuation, cfg.link_rate_fluctuation, len(a))
    rates[a, b] = rates[b, a] = base_rates[a, b] * (1 + jitter)
    return rates


def _connected(edge_adj):
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(edge_adj[node]):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(edge_adj)


def fluctuate_topology(rng, cfg, adj, base_rates, base_background, prop, forward_power,
                       previous_background, flapped_edges):
    """更新物理链路，返回本时隙可观测速率、拥塞、趋势及周期内仍断开的边。"""
    edge_adj = adj[:cfg.n_edges, :cfg.n_edges]
    active = list(zip(*np.nonzero(np.triu(edge_adj, k=1))))
    inactive = sorted(set(combinations(range(cfg.n_edges), 2)) - set(active))
    disconnected = set(flapped_edges)
    edge_count = len(active)
    for a, b in active:
        if rng.random() >= cfg.link_flap_prob:
            continue
        edge_adj[a, b] = edge_adj[b, a] = 0
        if _connected(edge_adj):
            for matrix in (base_rates, base_background, prop, forward_power):
                matrix[a, b] = matrix[b, a] = 0
            disconnected.add((int(a), int(b)))
            edge_count -= 1
        else:
            edge_adj[a, b] = edge_adj[b, a] = 1

    # 每个空余链路名额独立尝试激活，候选只取上个时隙未连接的边。
    vacancies = min(cfg.undirected_edge_links - edge_count, len(inactive))
    additions = rng.binomial(vacancies, cfg.link_flap_prob)
    continuous = base_rates > 0
    for i in rng.choice(len(inactive), additions, replace=False):
        a, b = inactive[i]
        _activate_link(rng, cfg, a, b, adj, base_rates, base_background, prop, forward_power)
        disconnected.discard((a, b))

    background = base_background.copy()
    count = np.count_nonzero(continuous)
    noise = rng.uniform(-cfg.congestion_step_noise, cfg.congestion_step_noise, count)
    spikes = rng.random(count) < cfg.link_spike_prob
    noise[spikes] += rng.uniform(cfg.link_spike_min, cfg.link_spike_max, spikes.sum())
    background[continuous] = np.clip(
        cfg.congestion_ar_coef * previous_background[continuous]
        + (1 - cfg.congestion_ar_coef) * base_background[continuous] + noise,
        cfg.congestion_min, cfg.congestion_max,
    )
    delta = np.zeros_like(background)
    delta[continuous] = background[continuous] - previous_background[continuous]
    return fluctuate_rates(rng, cfg, base_rates), background, delta, sorted(disconnected)


def simple_routes(adjacency, source, destination, max_hops):
    """DFS 枚举边缘层简单路径，云端不作为中继节点。"""
    routes = []
    stack = [(source, (source,))]
    while stack:
        node, path = stack.pop()
        if node == destination:
            routes.append(path)
            continue
        if len(path) > max_hops:
            continue
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if neighbor not in path and neighbor != len(adjacency) - 1:
                stack.append((neighbor, path + (neighbor,)))
    return routes
