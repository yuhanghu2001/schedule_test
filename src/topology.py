import numpy as np


def make_topology(rng, n_edges, link_count, link_bounds, cloud_bounds, congestion_bounds):
    """保留一个环确保连通，再随机添加若干边；最后一个节点为云端。"""
    cloud = n_edges
    n = n_edges + 1
    adjacency = np.eye(n, dtype=np.float32)
    rates = np.zeros((n, n), dtype=np.float32)
    loads = np.zeros((n, n), dtype=np.float32)
    prop = np.zeros((n, n), dtype=np.float32)
    pairs = {(min(i, (i + 1) % n_edges), max(i, (i + 1) % n_edges)) for i in range(n_edges)}
    while len(pairs) < link_count:
        a, b = rng.choice(n_edges, size=2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
    for a, b in pairs:
        rate = rng.uniform(*link_bounds)
        load = rng.uniform(*congestion_bounds)
        delay = rng.uniform(0.2, 2.0) / 1000
        adjacency[a, b] = adjacency[b, a] = 1
        rates[a, b] = rates[b, a] = rate
        loads[a, b] = loads[b, a] = load
        prop[a, b] = prop[b, a] = delay
    for a in range(n_edges):
        rate = rng.uniform(*cloud_bounds)
        adjacency[a, cloud] = adjacency[cloud, a] = 1
        rates[a, cloud] = rates[cloud, a] = rate
        loads[a, cloud] = loads[cloud, a] = rng.uniform(*congestion_bounds)
        prop[a, cloud] = prop[cloud, a] = rng.uniform(1.0, 3.0) / 1000
    return adjacency, rates, loads, prop


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
