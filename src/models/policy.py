"""GAT-PPO：节点/路径图嵌入 + 候选动作打分 + Critic。"""
import torch
from torch import nn
from torch.distributions import Categorical
from .gat import GATEncoder


class GATPPO(nn.Module):
    def __init__(self, hidden=32, use_gat=True, global_dim=9):
        super().__init__()
        self.use_gat = use_gat
        self.gat = GATEncoder(10, hidden)
        self.flat_encoder = nn.Sequential(nn.Linear(10, hidden), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(hidden * 3 + 8 + 10 + global_dim, 64), nn.Tanh(),
                                   nn.Linear(64, 1))
        self.critic = nn.Sequential(
            nn.Linear(hidden * 2 + 8 + global_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, obs):
        device = next(self.parameters()).device
        nodes = torch.as_tensor(obs["nodes"], dtype=torch.float32, device=device)
        adjacency = torch.as_tensor(obs["adjacency"], dtype=torch.float32, device=device)
        edges = torch.as_tensor(obs["edges"], dtype=torch.float32, device=device)
        task = torch.as_tensor(obs["task"], dtype=torch.float32, device=device)
        targets = torch.as_tensor(obs["candidate_targets"], dtype=torch.long, device=device)
        paths = torch.as_tensor(obs["candidate_paths"], dtype=torch.float32, device=device)
        extras = torch.as_tensor(obs["candidate_features"], dtype=torch.float32, device=device)
        global_features = torch.as_tensor(
            obs["global_features"], dtype=torch.float32, device=device,
        )
        h = self.gat(nodes, adjacency, edges) if self.use_gat else self.flat_encoder(nodes)
        k = len(targets)
        source = h[obs["source"]].expand(k, -1)
        dest = h[targets]
        pooled = paths @ h
        task_batch = task.expand(k, -1)
        global_batch = global_features.expand(k, -1)
        logits = self.actor(
            torch.cat((source, dest, pooled, task_batch, extras, global_batch), dim=-1)
        ).squeeze(-1)
        value_features = torch.cat((h[obs["source"]], h.mean(dim=0), task, global_features))
        value = self.critic(value_features).squeeze(-1)
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs, greedy=False):
        distribution, value = self(obs)
        action = distribution.probs.argmax() if greedy else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())
