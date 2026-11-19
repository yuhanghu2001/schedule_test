import unittest

import torch

from src.models.gat import GATEncoder


class ResidualGATTest(unittest.TestCase):
    def test_neighbor_residual_and_isolated_node(self):
        encoder = GATEncoder(node_dim=3, hidden=8)
        nodes = torch.eye(3)
        adjacency = torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        edges = torch.zeros(3, 3, 4)
        layer = encoder.layer1
        h = layer.transform(nodes)
        # 前两行只有一个非自身邻居，第三行仅走本节点分支。
        expected = torch.stack((h[0] + h[1], h[1] + h[0], h[2]))
        h1 = layer(nodes, adjacency, edges)
        torch.testing.assert_close(h1, layer.activation(layer.norm(expected)))
        torch.testing.assert_close(
            encoder(nodes, adjacency, edges),
            encoder.layer2(h1, adjacency, edges) + encoder.skip(nodes),
        )
        torch.testing.assert_close(adjacency.diagonal(), torch.ones(3))

    def test_second_hop_edge_features_reach_source(self):
        encoder = GATEncoder(node_dim=4, hidden=8)
        nodes = torch.eye(4)
        adjacency = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.],
                                  [0., 1., 1., 1.], [0., 0., 1., 1.]])
        edges = torch.zeros(4, 4, 4, requires_grad=True)
        encoder(nodes, adjacency, edges)[0, 0].backward()
        self.assertGreater(edges.grad[1, 2].abs().sum().item(), 0.)
        self.assertEqual(edges.grad[2, 3].abs().sum().item(), 0.)
