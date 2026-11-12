"""图结构扰动消融。"""
import numpy as np


def shuffle_graph_observation(obs, permutation):
    shuffled = dict(obs)
    shuffled["adjacency"] = obs["adjacency"][permutation][:, permutation]
    shuffled["edges"] = obs["edges"][permutation][:, permutation]
    return shuffled


class ShuffledGraphEnv:
    """保持真实动作与约束不变，只扰动策略看到的图连接。"""

    def __init__(self, env, seed):
        self.env = env
        edge_nodes = np.random.default_rng(seed).permutation(env.cloud)
        self.permutation = np.append(edge_nodes, env.cloud)

    @property
    def rng(self):
        return self.env.rng

    def reset(self):
        return shuffle_graph_observation(self.env.reset(), self.permutation)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if not done:
            obs = shuffle_graph_observation(obs, self.permutation)
        return obs, reward, done, info

    def metrics(self):
        return self.env.metrics()


def model_environment(env, model_type, seed):
    if model_type == "shuffled_gat":
        return ShuffledGraphEnv(env, seed)
    return env
