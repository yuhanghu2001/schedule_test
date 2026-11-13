"""轻量、无外部图神经网络依赖的边特征 GAT。"""
import torch
from torch import nn


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=3):
        super().__init__()
        self.transform = nn.Linear(in_dim, out_dim, bias=False)
        self.attention = nn.Linear(2 * out_dim + edge_dim, 1, bias=False)
        self.activation = nn.ELU()

    def forward(self, x, adjacency, edge_features):
        h = self.transform(x)
        n = h.size(0)
        h_i = h[:, None, :].expand(n, n, -1)
        h_j = h[None, :, :].expand(n, n, -1)
        scores = self.attention(torch.cat((h_i, h_j, edge_features), dim=-1)).squeeze(-1)
        scores = torch.nn.functional.leaky_relu(scores, negative_slope=0.2)
        scores = scores.masked_fill(adjacency == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return self.activation(weights @ h)


class GATEncoder(nn.Module):
    def __init__(self, node_dim=10, hidden=256):
        super().__init__()
        self.layer1 = GraphAttentionLayer(node_dim, hidden)
        self.layer2 = GraphAttentionLayer(hidden, hidden)

    def forward(self, nodes, adjacency, edges):
        return self.layer2(self.layer1(nodes, adjacency, edges), adjacency, edges)
