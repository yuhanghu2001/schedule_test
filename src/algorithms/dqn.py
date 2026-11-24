"""按动态候选打分的 DQN。"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn


STATE_FIELDS = ("nodes", "task", "global_features")
DQN_FIELDS = (*STATE_FIELDS, "candidate_features", "candidate_targets", "candidate_paths")


def tensor_observation(obs, fields):
    return {"source": int(obs["source"]), **{
        field: torch.as_tensor(obs[field]) for field in fields
    }}


def batch_field(observations, field, device):
    return torch.stack([obs[field] for obs in observations]).to(device)


def exploration_level(start, end, fraction, episode_index, episodes):
    progress = min(1., episode_index / (fraction * episodes))
    return start + (end - start) * progress


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.transitions = []
        self.steps = 0

    def append(self, transition):
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.steps % self.capacity] = transition
        self.steps += 1

    def sample(self, batch_size):
        indices = np.random.choice(len(self.transitions), batch_size, replace=False)
        return [self.transitions[index] for index in indices]

    def state_dict(self):
        return {"transitions": self.transitions, "steps": self.steps}

    def load_state_dict(self, state):
        self.transitions = state["transitions"]
        self.steps = state["steps"]


class DQN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.gat_hidden_dim
        self.flat_encoder = nn.Sequential(nn.Linear(10, hidden), nn.Tanh())
        self.q_head = nn.Sequential(
            nn.Linear(3 * hidden + 8 + 10 + 9, cfg.actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(cfg.actor_hidden_dim, 1),
        )

    def forward(self, observations):
        device = self.q_head[-1].weight.device
        nodes = self.flat_encoder(batch_field(observations, "nodes", device))
        context = torch.cat((
            batch_field(observations, "task", device),
            batch_field(observations, "global_features", device),
        ), dim=1)
        features = []
        for obs, hidden, state in zip(observations, nodes, context):
            targets = obs["candidate_targets"].to(device)
            paths = obs["candidate_paths"].to(device)
            candidates = obs["candidate_features"].to(device)
            count = len(targets)
            features.append(torch.cat((
                hidden[obs["source"]].expand(count, -1),
                hidden[targets],
                paths @ hidden,
                candidates,
                state.expand(count, -1),
            ), dim=1))
        scores = self.q_head(torch.cat(features)).squeeze(-1)
        return scores.split([len(rows) for rows in features])

    @torch.no_grad()
    def act(self, obs, greedy=True):
        obs = tensor_observation(obs, DQN_FIELDS)
        (scores,) = self([obs])
        action = int(scores.argmax()) if greedy else np.random.randint(len(scores))
        return int(action), 0.0, float(scores[action])


class DQNTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.target = deepcopy(model).requires_grad_(False)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.dqn_learning_rate)
        self.replay = ReplayBuffer(cfg.dqn_buffer_size)

    def update(self):
        cfg = self.cfg
        observations, actions, rewards, next_observations = zip(
            *self.replay.sample(cfg.dqn_batch_size)
        )
        predictions = torch.stack([
            scores[action] for scores, action in zip(self.model(observations), actions)
        ])
        targets = predictions.new_tensor(rewards)
        active = [index for index, obs in enumerate(next_observations) if obs is not None]
        if active:
            states = [next_observations[index] for index in active]
            with torch.no_grad():
                next_scores = self.target(states)
                if cfg.dqn_double:
                    next_values = [target[online.argmax()] for target, online in
                                   zip(next_scores, self.model(states))]
                else:
                    next_values = [scores.max() for scores in next_scores]
                targets[active] += cfg.gamma * torch.stack(next_values)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach())

    def train_episode(self, env, episode_index, episodes):
        cfg = self.cfg
        epsilon = exploration_level(cfg.dqn_epsilon_start, cfg.dqn_epsilon_end,
                                    cfg.dqn_epsilon_decay_fraction, episode_index, episodes)
        obs = tensor_observation(env.reset(), DQN_FIELDS)
        losses = []
        done = False
        while not done:
            if np.random.random() < epsilon:
                action = int(np.random.randint(len(env.actions)))
            else:
                action, _, _ = self.model.act(obs)
            next_obs, reward, done, _ = env.step(action)
            next_state = None if done else tensor_observation(next_obs, DQN_FIELDS)
            self.replay.append((obs, action, reward, next_state))
            obs = next_state
            step = self.replay.steps
            if (step >= cfg.dqn_warmup_steps and step % cfg.dqn_train_every == 0
                    and len(self.replay.transitions) >= cfg.dqn_batch_size):
                losses.append(self.update())
            if step % cfg.dqn_target_sync == 0:
                self.target.load_state_dict(self.model.state_dict())
        update_metrics = {"epsilon": epsilon, "updates": len(losses)}
        if losses:
            update_metrics["loss"] = float(np.mean(losses))
        return update_metrics, env.metrics()

    def state_dict(self):
        return {
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
