"""GAE + mini-batch PPO-Clip。"""
import numpy as np
import torch
from torch import nn


class PPOTrainer:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.ent_coef = cfg.ent_coef
        self.device = next(model.parameters()).device
        critic_parameters = list(model.critic.parameters())
        critic_parameter_ids = {id(parameter) for parameter in critic_parameters}
        actor_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in critic_parameter_ids
        ]
        self.optimizer = torch.optim.Adam([
            {"params": actor_parameters, "lr": cfg.actor_learning_rate, "name": "actor"},
            {"params": critic_parameters, "lr": cfg.critic_learning_rate, "name": "critic"},
        ])

    def set_schedule(self, episode_index, episodes):
        progress = min(1.0, episode_index / max(1, episodes - 1))
        lr_factor = 1.0 + (self.cfg.ppo_lr_end_factor - 1.0) * progress
        self.optimizer.param_groups[0]["lr"] = self.cfg.actor_learning_rate * lr_factor
        self.optimizer.param_groups[1]["lr"] = self.cfg.critic_learning_rate * lr_factor
        self.ent_coef = self.cfg.ent_coef + (self.cfg.ppo_ent_coef_end - self.cfg.ent_coef) * progress

    def collect_episode(self, env):
        env = self.model.rollout_environment(env, self.cfg.seed)
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
                batch_idx = indices[start:start + self.cfg.ppo_batch_size].tolist()
                batch_obs = [episode[t][0] for t in batch_idx]
                batch_actions = [episode[t][1] for t in batch_idx]
                old_logprobs = torch.tensor([episode[t][2] for t in batch_idx], dtype=torch.float32, device=self.device)
                adv_batch = torch.as_tensor(advantages[batch_idx], dtype=torch.float32, device=self.device)
                ret_batch = torch.as_tensor(returns[batch_idx], dtype=torch.float32, device=self.device)
                log_probs, values, entropy_batch, norm_ent = self.model.evaluate_batch(batch_obs, batch_actions)
                log_ratio = log_probs - old_logprobs
                ratio = log_ratio.exp()
                clipped_ratio = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip)
                policy_loss = -torch.minimum(ratio * adv_batch, clipped_ratio * adv_batch)
                value_loss = 0.5 * (values - ret_batch).square()
                loss = (policy_loss + self.cfg.vf_coef * value_loss - self.ent_coef * entropy_batch).mean()
                self.optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                losses.append(float(loss.item()))
                policy_losses.extend(policy_loss.detach().cpu().tolist())
                value_losses.extend(value_loss.detach().cpu().tolist())
                entropies.extend(entropy_batch.detach().cpu().tolist())
                normalized_entropies.extend(norm_ent)
                approximate_kls.extend(((ratio - 1) - log_ratio).detach().cpu().tolist())
                clip_fractions.extend(((ratio - 1).abs() > self.cfg.ppo_clip).float().detach().cpu().tolist())
                value_predictions.extend(values.detach().cpu().tolist())
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
