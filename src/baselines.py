"""传统调度基线。"""
from copy import deepcopy
from itertools import islice, takewhile

import numpy as np


def feasible_indices(actions, action_mask=None):
    if action_mask is None:  # v4 只枚举可行接收动作
        return range(len(actions) - 1)
    return np.flatnonzero(action_mask[:-1]).tolist()


def shortest_fastest_action(actions, action_mask=None):
    return min(
        feasible_indices(actions, action_mask),
        key=lambda i: (
            max(0, len(actions[i].path) - 1),
            -actions[i].compute_share,
            actions[i].delay,
            actions[i].energy,
        ),
        default=len(actions) - 1,
    )


def greedy_score(action, cfg):
    return (
        cfg.reward_beta_completion
        - cfg.reward_beta_delay * action.delay / cfg.reward_delay_ref
        - cfg.reward_beta_energy * action.energy / cfg.reward_energy_ref
    )


def greedy_estimate_action(actions, cfg, action_mask=None):
    return max(feasible_indices(actions, action_mask), key=lambda i: greedy_score(actions[i], cfg),
               default=len(actions) - 1)


def random_action(env):
    indices = feasible_indices(env.actions, getattr(env, "action_mask", None))
    return int(env.rng.choice(indices)) if len(indices) else len(env.actions) - 1


def cloud_first_action(env):
    actions = env.actions
    return min(feasible_indices(actions, getattr(env, "action_mask", None)),
               key=lambda i: (actions[i].destination != env.cloud, actions[i].delay, actions[i].energy),
               default=len(actions) - 1)


def heft_hetero_action(env):
    # 就绪队列已经按 rank_u 排序，delivery_time_s 在非出口处等于 finish_time_s。
    def eft_key(i):
        action = env.actions[i]
        hops = max(0, len(action.path) - 1 - int(env.cloud in action.path))
        return action.delivery_time_s, action.energy, hops

    return min(feasible_indices(env.actions, env.action_mask), key=eft_key,
               default=len(env.actions) - 1)


def baseline_action(name, env):
    mask = getattr(env, "action_mask", None)
    if name == "shortest_fastest":
        return shortest_fastest_action(env.actions, mask)
    if name in ("greedy", "greedy_estimate"):
        return greedy_estimate_action(env.actions, env.cfg, mask)
    if name == "random":
        return random_action(env)
    if name == "cloud_first":
        return cloud_first_action(env)
    if name == "heft_hetero":
        return heft_hetero_action(env)
    raise ValueError(f"未知启发式基线: {name}")


def ranked_action(actions, cfg, rank, reject_rank):
    feasible = sorted(
        range(len(actions) - 1),
        key=lambda index: greedy_score(actions[index], cfg),
        reverse=True,
    )
    if rank == reject_rank or not feasible:
        return len(actions) - 1
    return feasible[min(rank, len(feasible) - 1)]


def slot_end(env):
    current_tasks = takewhile(lambda task: task.slot == env.slot, islice(env.tasks, env.idx, None))
    return env.idx + sum(1 for _ in current_tasks)


def slot_model(env, end, seed):
    # 截断任务列表，使末任务自然终止；从独立种子开始采样执行扰动。
    return deepcopy(env, {
        id(env.tasks): deepcopy(env.tasks[:end]),
        id(env.execution_rng): np.random.default_rng(seed),
    })


def slot_reward(env, end, seed, ranks, reject_rank):
    simulation = slot_model(env, end, seed)
    total_reward = 0.0
    for rank in ranks:
        action = ranked_action(simulation.actions, simulation.cfg, rank, reject_rank)
        _, reward, _, _ = simulation.step(action)
        total_reward += reward
    return total_reward


def genetic_plan(cfg, rng, task_count, rank_count, fitness):
    population = rng.integers(rank_count, size=(cfg.ga_population, task_count))
    population[0] = 0
    for generation in range(cfg.ga_generations):
        scores = np.array([fitness(ranks) for ranks in population])
        elite = np.argsort(-scores, kind="stable")[:cfg.ga_elite]
        if generation + 1 == cfg.ga_generations:
            return population[scores.argmax()]
        offspring = [population[index] for index in elite]
        while len(offspring) < cfg.ga_population:
            tournaments = rng.integers(len(population), size=(2, cfg.ga_tournament_size))
            parents = tournaments[np.arange(2), scores[tournaments].argmax(axis=1)]
            child = population[parents[0]].copy()
            if rng.random() < cfg.ga_crossover:
                cross = rng.random(task_count) < 0.5
                child[cross] = population[parents[1], cross]
            mutate = rng.random(task_count) < cfg.ga_mutation
            child[mutate] = rng.integers(rank_count, size=mutate.sum())
            offspring.append(child)
        population = np.stack(offspring)


def ant_colony_plan(cfg, rng, task_count, rank_count, fitness):
    pheromone = np.ones((task_count, rank_count))
    heuristic = (1 / np.arange(1, rank_count + 1)) ** cfg.aco_beta
    positions = np.arange(task_count)
    best_score = -np.inf
    for _ in range(cfg.aco_iterations):
        weights = pheromone ** cfg.aco_alpha * heuristic
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        ants = np.column_stack([
            rng.choice(rank_count, size=cfg.aco_ants, p=row) for row in probabilities
        ])
        scores = np.array([fitness(ranks) for ranks in ants])
        winner = scores.argmax()
        if scores[winner] > best_score:
            best_score = scores[winner]
            best_plan = ants[winner]
        pheromone *= 1 - cfg.aco_evaporation
        pheromone[positions, best_plan] += 1
    return best_plan


def evaluate_slot_planner(env, method):
    cfg = env.cfg
    planner, max_rank = (
        (genetic_plan, cfg.ga_max_rank) if method == "ga"
        else (ant_colony_plan, cfg.aco_max_rank)
    )
    reject_rank = max_rank + 1
    done = False
    while not done:
        end = slot_end(env)
        search_seed, model_seed = np.random.SeedSequence([cfg.seed, env.slot, 1]).spawn(2)

        def fitness(ranks):
            return slot_reward(env, end, model_seed, ranks, reject_rank)

        plan = planner(cfg, np.random.default_rng(search_seed), end - env.idx, reject_rank + 1, fitness)
        for rank in plan:
            action = ranked_action(env.actions, cfg, rank, reject_rank)
            _, _, done, _ = env.step(action)
    return env.metrics()


def evaluate_heuristic(env, method="shortest_fastest"):
    env.reset()
    if method in ("ga", "aco"):
        return evaluate_slot_planner(env, method)
    done = False
    while not done:
        action = baseline_action(method, env)
        _, _, done, _ = env.step(action)
    return env.metrics()
