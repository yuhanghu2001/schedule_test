"""传统调度基线。"""


def shortest_fastest_action(actions):
    feasible = list(enumerate(actions[:-1]))
    if not feasible:
        return len(actions) - 1
    return min(
        feasible,
        key=lambda item: (
            max(0, len(item[1].path) - 1),
            -item[1].compute_share,
            item[1].delay,
            item[1].energy,
        ),
    )[0]


def greedy_estimate_action(actions, cfg):
    feasible = list(enumerate(actions[:-1]))
    if not feasible:
        return len(actions) - 1
    return max(
        feasible,
        key=lambda item: (
            cfg.reward_beta_completion
            - cfg.reward_beta_delay * item[1].delay / cfg.reward_delay_ref
            - cfg.reward_beta_energy * item[1].energy / cfg.reward_energy_ref
        ),
    )[0]


def evaluate_heuristic(env, method="shortest_fastest"):
    env.reset()
    done = False
    while not done:
        if method == "shortest_fastest":
            action = shortest_fastest_action(env.actions)
        else:
            action = greedy_estimate_action(env.actions, env.cfg)
        _, _, done, _ = env.step(action)
    return env.metrics()
