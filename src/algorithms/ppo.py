"""GAE + mini-batch PPO-Clip。"""
import numpy as np
import torch
from torch import nn


class PPOTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    def collect_episode(self, env):
        obs = env.reset()
        episode = []
        done = False
        while not done:
            action, old_logprob, value = self.model.act(obs)
            next_obs, reward, done, _ = env.step(action)
            episode.append((obs, action, old_logprob, value, reward, done))
            obs = next_obs
        return episode, env.metrics()

    def update(self, episode):
        n = len(episode)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        next_adv = 0.
        next_value = 0.
        for t in reversed(range(n)):
            _, _, _, value, reward, done = episode[t]
            delta = reward + self.cfg.gamma * next_value * (1 - done) - value
            next_adv = delta + self.cfg.gamma * self.cfg.gae_lambda * next_adv * (1 - done)
            advantages[t] = next_adv
            returns[t] = advantages[t] + value
            next_value = value
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        losses = []
        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.cfg.ppo_batch_size):
                batch_losses = []
                for t in indices[start:start + self.cfg.ppo_batch_size].tolist():
                    obs, action, old_logprob, _, _, _ = episode[t]
                    distribution, value = self.model(obs)
                    action_tensor = torch.tensor(action)
                    ratio = (distribution.log_prob(action_tensor) - old_logprob).exp()
                    advantage = torch.tensor(advantages[t])
                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                    actor_loss = -torch.minimum(
                        ratio * advantage, clipped_ratio * advantage)
                    critic_loss = .5 * (value - torch.tensor(returns[t])).square()
                    batch_losses.append(
                        actor_loss + critic_loss - .01 * distribution.entropy())
                loss = torch.stack(batch_losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses))
