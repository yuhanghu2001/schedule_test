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


def evaluate_heuristic(env):
    env.reset()
    done = False
    while not done:
        action = shortest_fastest_action(env.actions)
        _, _, done, _ = env.step(action)
    return env.metrics()
