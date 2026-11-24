"""DDPG（Wolpertinger 离散映射）：连续原型映射到最近的可行动作。"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .dqn import STATE_FIELDS, ReplayBuffer, batch_field, exploration_level, tensor_observation


DDPG_FIELDS = (*STATE_FIELDS, "candidate_features")


class FlatStateEncoder(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.nodes = nn.Sequential(nn.Linear(10, hidden), nn.Tanh())

    def forward(self, observations):
        device = self.nodes[0].weight.device
        nodes = self.nodes(batch_field(observations, "nodes", device))
        sources = [obs["source"] for obs in observations]
        return torch.cat((
            nodes[list(range(len(observations))), sources],
            nodes.mean(dim=1),
            batch_field(observations, "task", device),
            batch_field(observations, "global_features", device),
        ), dim=1)


def nearest_candidate(prototype, candidates):
    return (candidates - prototype).square().sum(dim=-1).argmin()


class DDPG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.gat_hidden_dim
        self.noise_start = cfg.ddpg_noise_start
        self.actor = nn.Sequential(
            FlatStateEncoder(hidden),
            nn.Linear(2 * hidden + 8 + 9, cfg.actor_hidden_dim), nn.Tanh(),
            nn.Linear(cfg.actor_hidden_dim, 10), nn.Tanh(),
        )
        self.critic_encoder = FlatStateEncoder(hidden)
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden + 8 + 9 + 10, cfg.critic_hidden_dim), nn.Tanh(),
            nn.Linear(cfg.critic_hidden_dim, 1),
        )
        self.register_buffer("candidate_low", torch.tensor(cfg.ddpg_candidate_low, dtype=torch.float32))
        self.register_buffer("candidate_high", torch.tensor(cfg.ddpg_candidate_high, dtype=torch.float32))

    def candidates(self, obs):
        features = obs["candidate_features"].to(self.candidate_low.device)
        return (2 * (features - self.candidate_low) /
                (self.candidate_high - self.candidate_low) - 1).clamp(-1, 1)

    def value(self, observations, actions):
        return self.critic(torch.cat((self.critic_encoder(observations), actions), dim=1)).squeeze(-1)

    @torch.no_grad()
    def select(self, obs, noise_std=0.0):
        (prototype,) = self.actor([obs])
        if noise_std:
            prototype = (prototype + noise_std * torch.randn_like(prototype)).clamp(-1, 1)
        candidates = self.candidates(obs)
        action = int(nearest_candidate(prototype, candidates))
        return action, candidates[action]

    @torch.no_grad()
    def act(self, obs, greedy=True):
        obs = tensor_observation(obs, DDPG_FIELDS)
        action, features = self.select(obs, noise_std=0.0 if greedy else self.noise_start)
        return action, 0.0, float(self.value([obs], features.unsqueeze(0)))


class DDPGTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.target = deepcopy(model).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=cfg.ddpg_actor_learning_rate)
        self.critic_parameters = [*model.critic_encoder.parameters(), *model.critic.parameters()]
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=cfg.ddpg_critic_learning_rate)
        self.replay = ReplayBuffer(cfg.ddpg_buffer_size)

    def update(self):
        cfg = self.cfg
        observations, actions, rewards, next_observations = zip(
            *self.replay.sample(cfg.ddpg_batch_size)
        )
        device = self.model.candidate_low.device
        predictions = self.model.value(observations, torch.stack(actions).to(device))
        targets = predictions.new_tensor(rewards)
        active = [index for index, obs in enumerate(next_observations) if obs is not None]
        if active:
            states = [next_observations[index] for index in active]
            with torch.no_grad():
                prototypes = self.target.actor(states)
                candidates = [self.target.candidates(obs) for obs in states]
                selected = torch.stack([
                    rows[nearest_candidate(prototype, rows)]
                    for prototype, rows in zip(prototypes, candidates)
                ])
                targets[active] += cfg.gamma * self.target.value(states, selected)

        critic_loss = nn.functional.mse_loss(predictions, targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for parameter in self.critic_parameters:
            parameter.requires_grad_(False)
        actor_loss = -self.model.value(observations, self.model.actor(observations)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for parameter in self.critic_parameters:
            parameter.requires_grad_(True)

        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.model.parameters()):
                target.lerp_(online, cfg.ddpg_tau)
        return float(actor_loss.detach()), float(critic_loss.detach())

    def train_episode(self, env, episode_index, episodes):
        cfg = self.cfg
        noise = exploration_level(cfg.ddpg_noise_start, cfg.ddpg_noise_end,
                                  cfg.ddpg_noise_decay_fraction, episode_index, episodes)
        obs = tensor_observation(env.reset(), DDPG_FIELDS)
        losses = []
        done = False
        while not done:
            action, features = self.model.select(obs, noise)
            next_obs, reward, done, _ = env.step(action)
            next_state = None if done else tensor_observation(next_obs, DDPG_FIELDS)
            self.replay.append((obs, features.cpu().clone(), reward, next_state))
            obs = next_state
            step = self.replay.steps
            if (step >= cfg.ddpg_warmup_steps and step % cfg.ddpg_train_every == 0
                    and len(self.replay.transitions) >= cfg.ddpg_batch_size):
                losses.append(self.update())
        update_metrics = {"noise_std": noise, "updates": len(losses)}
        if losses:
            actor_loss, critic_loss = np.mean(losses, axis=0)
            update_metrics.update(actor_loss=float(actor_loss), critic_loss=float(critic_loss))
        return update_metrics, env.metrics()

    def state_dict(self):
        return {
            "target": self.target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.replay.state_dict(),
        }

    def load_state_dict(self, state):
        self.target.load_state_dict(state["target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.replay.load_state_dict(state["replay"])
