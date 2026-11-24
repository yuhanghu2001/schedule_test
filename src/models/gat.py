"""轻量、无外部图神经网络依赖的边特征 GAT。"""
import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4):
        super().__init__()
        self.transform = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Linear(2 * out_dim + edge_dim, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ELU()

    def forward(self, x, adjacency, edge_features):
        h = self.transform(x)
        n = h.size(0)
        h_i = h[:, None, :].expand(n, n, -1)
        h_j = h[None, :, :].expand(n, n, -1)
        scores = self.attention(torch.cat((h_i, h_j, edge_features), dim=-1)).squeeze(-1)
        scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)
        if self.norm is None:  # 旧 checkpoint 的含自环注意力
            weights = torch.softmax(scores.masked_fill(adjacency == 0, -1e9), dim=1)
            return self.activation(weights @ h)
        neighbors = adjacency.clone()
        neighbors.fill_diagonal_(0)
        weights = torch.softmax(scores.masked_fill(neighbors == 0, -1e9), dim=1)
        weights = weights * neighbors.any(dim=1, keepdim=True)
        return self.activation(self.norm(h + weights @ h))


class GATEncoder(nn.Module):
    def __init__(self, node_dim=10, hidden=256, edge_dim=4):
        super().__init__()
        self.layer1 = GraphAttentionLayer(node_dim, hidden, edge_dim)
        self.layer2 = GraphAttentionLayer(hidden, hidden, edge_dim)
        self.skip = nn.Linear(node_dim, hidden)

    def forward(self, nodes, adjacency, edges):
        h = self.layer1(nodes, adjacency, edges)
        h = self.layer2(h, adjacency, edges)
        return h + self.skip(nodes) if self.skip is not None else h


class DAGGraphEncoder(nn.Module):
    """对前驱做边特征注意力聚合，返回当前子任务及整张 DAG 的表示。"""

    def __init__(self, node_dim=22, hidden=64):
        super().__init__()
        self.layer = GraphAttentionLayer(node_dim, hidden, edge_dim=1)

    def forward(self, nodes, adjacency, edges, dag_node_mask, current_subtask_idx):
        neighbors = adjacency * (dag_node_mask[:, None] & dag_node_mask[None, :])
        h = self.layer(nodes, neighbors, edges)
        return h[current_subtask_idx], h[dag_node_mask].mean(dim=0)
