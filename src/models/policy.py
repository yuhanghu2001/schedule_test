"""GAT-PPO：节点/路径图嵌入 + 候选动作打分 + Critic。"""
import numpy as np
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
        cand_dim = 12 if self.dag_mode else 10
        self.actor = nn.Sequential(
            nn.Linear(gat_hidden * 3 + task_dim + cand_dim + global_dim, actor_hidden),
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
        if "action_mask" in obs:
            mask = torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device)
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            m = valid.numel()
            valid_logits = self.actor(torch.cat((
                h[obs["source"]].expand(m, -1), h[targets[valid]], paths[valid] @ h,
                task.expand(m, -1), extras[valid], global_features.expand(m, -1),
            ), dim=-1)).squeeze(-1)
            logits = torch.full((k,), float("-inf"), dtype=valid_logits.dtype, device=device)
            logits[valid] = valid_logits
        else:
            logits = self.actor(torch.cat((
                h[obs["source"]].expand(k, -1), h[targets], paths @ h,
                task.expand(k, -1), extras, global_features.expand(k, -1),
            ), dim=-1)).squeeze(-1)
        net_value = h.mean(dim=0)
        if not self.dag_mode:
            net_value = torch.cat((h[obs["source"]], net_value))
        value = self.critic(torch.cat((net_value, task, global_features))).squeeze(-1)
        return Categorical(logits=logits), value

    def evaluate_batch(self, obs_list, actions):
        """批量前向评估 minibatch，仅对合法候选计算 Actor，与逐条 forward 数学等价。"""
        device = next(self.parameters()).device
        b = len(obs_list)
        nodes = torch.as_tensor(np.stack([o["nodes"] for o in obs_list]), dtype=torch.float32, device=device)
        if self.use_gat:
            adj_key = "gat_adj" if self.dag_mode else "adjacency"
            adj = torch.as_tensor(np.stack([o[adj_key] for o in obs_list]), dtype=torch.float32, device=device)
            edges = torch.as_tensor(np.stack([o["edges"] for o in obs_list]), dtype=torch.float32, device=device)
            h = self.gat(nodes, adj, edges)
        else:
            h = self.flat_encoder(nodes)
        if self.dag_mode:
            dag_nodes = torch.as_tensor(np.stack([o["dag_nodes"] for o in obs_list]), dtype=torch.float32, device=device)
            currents = torch.tensor([o["current_subtask_idx"] for o in obs_list], dtype=torch.long, device=device)
            if self.dag_encoder is None:
                task = dag_nodes[torch.arange(b, device=device), currents]
            else:
                dag_adj = torch.as_tensor(np.stack([o["dag_adj"] for o in obs_list]), dtype=torch.float32, device=device)
                dag_edges = torch.as_tensor(np.stack([o["dag_edges"] for o in obs_list]), dtype=torch.float32, device=device)
                dag_mask = torch.as_tensor(np.stack([o["dag_node_mask"] for o in obs_list]), dtype=torch.bool, device=device)
                cur_emb, mean_emb = self.dag_encoder(dag_nodes, dag_adj, dag_edges, dag_mask, currents)
                task = torch.cat((cur_emb, mean_emb), dim=-1)
        else:
            task = torch.as_tensor(np.stack([o["task"] for o in obs_list]), dtype=torch.float32, device=device)
        sources = torch.tensor([o["source"] for o in obs_list], dtype=torch.long, device=device)
        global_features = torch.as_tensor(np.stack([o["global_features"] for o in obs_list]), dtype=torch.float32, device=device)
        net_value = h.mean(dim=1)
        if not self.dag_mode:
            net_value = torch.cat((h[torch.arange(b, device=device), sources], net_value), dim=-1)
        values = self.critic(torch.cat((net_value, task, global_features), dim=-1)).squeeze(-1)

        batch_ids, target_parts, path_parts, extra_parts = [], [], [], []
        counts, local_actions = [], []
        for i, (obs, act) in enumerate(zip(obs_list, actions)):
            if "action_mask" in obs:
                valid = np.flatnonzero(obs["action_mask"])
                target_parts.append(obs["candidate_targets"][valid])
                path_parts.append(obs["candidate_paths"][valid])
                extra_parts.append(obs["candidate_features"][valid])
                local_actions.append(int(np.searchsorted(valid, act)))
                count = len(valid)
            else:
                target_parts.append(obs["candidate_targets"])
                path_parts.append(obs["candidate_paths"])
                extra_parts.append(obs["candidate_features"])
                local_actions.append(int(act))
                count = len(obs["candidate_targets"])
            counts.append(count)
            batch_ids.append(np.full(count, i, dtype=np.int64))

        b_idx = torch.as_tensor(np.concatenate(batch_ids), dtype=torch.long, device=device)
        t_idx = torch.as_tensor(np.concatenate(target_parts), dtype=torch.long, device=device)
        paths_cat = torch.as_tensor(np.concatenate(path_parts, axis=0), dtype=torch.float32, device=device)
        extras_cat = torch.as_tensor(np.concatenate(extra_parts, axis=0), dtype=torch.float32, device=device)
        h_cand = h[b_idx]
        pooled = torch.einsum("ci,cid->cd", paths_cat, h_cand)
        all_logits = self.actor(torch.cat((
            h[b_idx, sources[b_idx]], h[b_idx, t_idx], pooled,
            task[b_idx], extras_cat, global_features[b_idx],
        ), dim=-1)).squeeze(-1)

        log_probs, entropies, norm_entropies = [], [], []
        for logits_i, act_i, count in zip(all_logits.split(counts), local_actions, counts):
            log_p = torch.log_softmax(logits_i, dim=0)
            log_probs.append(log_p[act_i])
            ent = -(log_p.exp() * log_p).sum() if count > 1 else logits_i.new_zeros(())
            entropies.append(ent)
            norm_entropies.append(float(ent.detach() / np.log(count)) if count > 1 else 0.0)
        return torch.stack(log_probs), values, torch.stack(entropies), norm_entropies

    @torch.no_grad()
    def act(self, obs, greedy=False):
        distribution, value = self(obs)
        action = distribution.probs.argmax() if greedy else distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())
