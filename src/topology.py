from collections import deque
from itertools import combinations

import numpy as np


def node_positions(n_edges, n_terminals):
    """与监控大屏共用的固定示意坐标，用于唯一最近邻接入。"""
    columns = min(4, n_edges)
    rows = (n_edges + columns - 1) // columns
    positions = []
    for index in range(n_edges):
        if index < columns:
            column, row = index, 0
        else:
            offset = index - columns
            column = columns - 1 - offset // (rows - 1)
            row = (1 + offset % (rows - 1) if (columns - 1 - column) % 2 == 0
                   else rows - 1 - offset % (rows - 1))
        x = 900.0 if columns == 1 else 300.0 + column * 1200.0 / (columns - 1)
        y = 490.0 if rows == 1 else 350.0 + row * 330.0 / (rows - 1)
        positions.append((x, y))
    terminals = [(60.0 + terminal * 1680.0 / max(1, n_terminals - 1), 815.0)
                 for terminal in range(n_terminals)]
    return np.array(positions + [(900.0, 120.0)]), np.array(terminals)


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


def edge_pairs(cfg):
    """八邻接模式只允许同一网格中的相邻节点建立物理链路。"""
    pairs = set(combinations(range(cfg.n_edges), 2))
    if cfg.edge_topology == "eight_neighbor":
        positions, _ = node_positions(cfg.n_edges, 0)
        xs, ys = sorted(set(positions[:-1, 0])), sorted(set(positions[:-1, 1]))
        cells = [(xs.index(x), ys.index(y)) for x, y in positions[:-1]]
        pairs = {(a, b) for a, b in pairs
                 if max(abs(cells[a][0] - cells[b][0]), abs(cells[a][1] - cells[b][1])) == 1}
    return pairs


def make_topology(rng, cfg):
    """构建八邻接网格或历史环形基线，仅指定网关上云。"""
    n = cfg.n_edges
    adj = np.eye(n + 1, dtype=np.float32)
    base_rates = np.zeros_like(adj)
    base_background = np.zeros_like(adj)
    prop = np.zeros_like(adj)
    forward_power = np.zeros_like(adj)
    if cfg.edge_topology == "eight_neighbor":
        pairs = edge_pairs(cfg)
    else:
        pairs = set(zip(range(n - 1), range(1, n)))
        if n > 1:
            pairs.add((0, n - 1))
        chords = sorted(edge_pairs(cfg) - pairs)
        if cfg.undirected_edge_links < len(pairs):
            raise ValueError("目标无向边数不足以构建边缘环")
        extra = min(cfg.undirected_edge_links - len(pairs), len(chords))
        pairs.update(chords[i] for i in rng.choice(len(chords), extra, replace=False))
    gateways = min(cfg.cloud_gateway_count or n, n)
    for a, b in sorted(pairs) + [(a, n) for a in range(gateways)]:
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
    allowed = edge_pairs(cfg)
    inactive = sorted(allowed - set(active))
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
    limit = len(allowed) if cfg.edge_topology == "eight_neighbor" else cfg.undirected_edge_links
    vacancies = min(limit - edge_count, len(inactive))
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
    """DFS 枚举真实简单路径，云端只允许作为最终目的地。"""
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
            if neighbor not in path and (neighbor != len(adjacency) - 1 or neighbor == destination):
                stack.append((neighbor, path + (neighbor,)))
    return routes
