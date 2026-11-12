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
        policy_losses = []
        value_losses = []
        entropies = []
        normalized_entropies = []
        approximate_kls = []
        clip_fractions = []
        value_predictions = []
        gradient_norms = []
        for _ in range(self.cfg.ppo_epochs):
            indices = torch.randperm(n)
            for start in range(0, n, self.cfg.ppo_batch_size):
                batch_losses = []
                for t in indices[start:start + self.cfg.ppo_batch_size].tolist():
                    obs, action, old_logprob, _, _, _ = episode[t]
                    distribution, value = self.model(obs)
                    action_tensor = torch.tensor(action)
                    log_ratio = distribution.log_prob(action_tensor) - old_logprob
                    ratio = log_ratio.exp()
                    advantage = torch.tensor(advantages[t])
                    clipped_ratio = torch.clamp(
                        ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                    policy_loss = -torch.minimum(
                        ratio * advantage, clipped_ratio * advantage)
                    value_loss = .5 * (value - torch.tensor(returns[t])).square()
                    entropy = distribution.entropy()
                    batch_losses.append(policy_loss + self.cfg.vf_coef * value_loss
                                        - self.cfg.ent_coef * entropy)
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    action_count = distribution.probs.numel()
                    normalized_entropies.append(
                        float(entropy.detach() / np.log(action_count)) if action_count > 1 else 0.0
                    )
                    approximate_kls.append(float(((ratio - 1) - log_ratio).detach()))
                    clip_fractions.append(float(
                        ((ratio - 1).abs() > self.cfg.ppo_clip).detach()
                    ))
                    value_predictions.append(float(value.detach()))
                loss = torch.stack(batch_losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                losses.append(float(loss.item()))
                gradient_norms.append(float(gradient_norm))
        rollout_values = np.array([item[3] for item in episode], dtype=np.float32)
        return_variance = np.var(returns)
        explained_variance = (
            1.0 - np.var(returns - rollout_values) / return_variance
            if return_variance > 0 else 0.0
        )
        return {
            "loss": float(np.mean(losses)),
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "normalized_entropy": float(np.mean(normalized_entropies)),
            "approx_kl": float(np.mean(approximate_kls)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "explained_variance": float(explained_variance),
            "grad_norm_preclip": float(np.mean(gradient_norms)),
            "value_mean": float(np.mean(value_predictions)),
            "return_mean": float(np.mean(returns)),
        }
