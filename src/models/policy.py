"""GAT-PPO：节点/路径图嵌入 + 候选动作打分 + Critic。"""
import torch
from torch import nn
from torch.distributions import Categorical
from ..environment import ShuffledGraphEnv, model_environment
from .gat import DAGGraphEncoder, GATEncoder


MODEL_NAMES = {
    "gatppo": "gatppo", "gatpppo": "gatppo", "ppo": "ppo",
    "gat": "gatppo", "flat": "ppo", "shuffled_gat": "shuffled_gat",
    "dual_gatppo": "dual_gatppo", "single_gatppo": "single_gatppo",
}


class DAGShuffledGraphEnv(ShuffledGraphEnv):
    """沿用已有拓扑置换，使 v5 节点特征和边特征使用相同的索引。"""

    def __init__(self, wrapped):
        self.env = wrapped.env
        self.permutation = wrapped.permutation

    def _align_nodes(self, obs):
        obs["nodes"] = obs["nodes"][self.permutation]
        obs["adj"] = obs["adj"][self.permutation][:, self.permutation]
        obs["gat_adj"] = obs["adjacency"]
        return obs

    def reset(self):
        return self._align_nodes(super().reset())

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if not done:
            self._align_nodes(obs)
        return obs, reward, done, info


class GATPPO(nn.Module):
    def __init__(self, gat_hidden=256, actor_hidden=128, critic_hidden=128,
                 use_gat=True, global_dim=9, *, model_type=None, observation_version=None):
        super().__init__()
        self.model_type = MODEL_NAMES[model_type or ("gatppo" if use_gat else "ppo")]
        self.use_gat = self.model_type != "ppo"
        self.dag_mode = (observation_version == "v5_dual_graph_dag"
                         or self.model_type in ("dual_gatppo", "single_gatppo"))
        node_dim = 18 if self.dag_mode else 10
        # v4 的两套编码器都属于历史 checkpoint，即便该策略只用其中之一。
        if self.use_gat or not self.dag_mode:
            self.gat = GATEncoder(node_dim, gat_hidden)
        if not self.use_gat or not self.dag_mode:
            self.flat_encoder = nn.Sequential(nn.Linear(node_dim, gat_hidden), nn.Tanh())
        self.dag_encoder = None
        task_dim = 8
        if self.dag_mode:
            task_dim = 22
            if self.model_type not in ("single_gatppo", "ppo"):
                self.dag_encoder = DAGGraphEncoder(task_dim, gat_hidden)
                task_dim = 2 * gat_hidden
        value_net_dim = gat_hidden if self.dag_mode else 2 * gat_hidden
        self.actor = nn.Sequential(
            nn.Linear(gat_hidden * 3 + task_dim + 10 + global_dim, actor_hidden),
            nn.Tanh(), nn.Linear(actor_hidden, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(value_net_dim + task_dim + global_dim, critic_hidden),
            nn.Tanh(), nn.Linear(critic_hidden, 1),
        )

    @classmethod
    def from_config(cls, cfg, model_type):
        return cls(cfg.gat_hidden_dim, cfg.actor_hidden_dim, cfg.critic_hidden_dim,
                   model_type=model_type, observation_version=cfg.observation_version)

    def rollout_environment(self, env, seed):
        """训练和独立推理共用此入口，已包装的环境沿用其原始置换。"""
        if isinstance(env, DAGShuffledGraphEnv):
            return env
        if not isinstance(env, ShuffledGraphEnv):
            env = model_environment(env, self.model_type, seed)
        if self.dag_mode and isinstance(env, ShuffledGraphEnv):
            return DAGShuffledGraphEnv(env)
        return env

    def forward(self, obs):
        device = next(self.parameters()).device
        nodes = torch.as_tensor(obs["nodes"], dtype=torch.float32, device=device)
        if self.use_gat:
            adjacency = torch.as_tensor(obs["gat_adj" if self.dag_mode else "adjacency"],
                                        dtype=torch.float32, device=device)
            edges = torch.as_tensor(obs["edges"], dtype=torch.float32, device=device)
            h = self.gat(nodes, adjacency, edges)
        else:
            h = self.flat_encoder(nodes)
        if self.dag_mode:
            dag_nodes = torch.as_tensor(obs["dag_nodes"], dtype=torch.float32, device=device)
            current = obs["current_subtask_idx"]
            if self.dag_encoder is None:
                task = dag_nodes[current]
            else:
                dag_adj = torch.as_tensor(obs["dag_adj"], dtype=torch.float32, device=device)
                dag_edges = torch.as_tensor(obs["dag_edges"], dtype=torch.float32, device=device)
                mask = torch.as_tensor(obs["dag_node_mask"], dtype=torch.bool, device=device)
                task = torch.cat(self.dag_encoder(dag_nodes, dag_adj, dag_edges, mask, current))
        else:
            task = torch.as_tensor(obs["task"], dtype=torch.float32, device=device)
        targets = torch.as_tensor(obs["candidate_targets"], dtype=torch.long, device=device)
        paths = torch.as_tensor(obs["candidate_paths"], dtype=torch.float32, device=device)
        extras = torch.as_tensor(obs["candidate_features"], dtype=torch.float32, device=device)
        global_features = torch.as_tensor(
            obs["global_features"], dtype=torch.float32, device=device,
        )
        k = len(targets)
        source = h[obs["source"]].expand(k, -1)
        dest = h[targets]
        pooled = paths @ h
        task_batch = task.expand(k, -1)
        global_batch = global_features.expand(k, -1)
        logits = self.actor(
            torch.cat((source, dest, pooled, task_batch, extras, global_batch), dim=-1)
        ).squeeze(-1)
        net_value = h.mean(dim=0)
        if not self.dag_mode:
            net_value = torch.cat((h[obs["source"]], net_value))
        value_features = torch.cat((net_value, task, global_features))
        value = self.critic(value_features).squeeze(-1)
        if "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask, float("-inf"))
        return Categorical(logits=logits), value

    @torch.no_grad()
    def act(self, obs, greedy=False):
        distribution, value = self(obs)
        action = distribution.probs.argmax() if greedy else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())
