"""训练与固定评估曲线。"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_LABELS = {
    "gatppo": "GAT-PPO", "ppo": "PPO", "dqn": "DQN", "ddpg": "DDPG",
    "shuffled_gat": "Shuffled GAT-PPO", "ga": "GA", "aco": "ACO",
    "greedy_estimate": "Greedy Estimate", "shortest_fastest": "Shortest Fastest",
}
CORE_METRICS = (
    ("task_success_rate", "任务完成率", "比例"),
    ("avg_delay_s", "任务平均完成时间", "秒"),
    ("avg_energy_j", "任务平均能耗", "J"),
    ("load_balance_degree", "负载均衡度", "均衡度"),
)


def method_colors(methods):
    if len(methods) <= 10:
        palette = plt.get_cmap("tab10").colors
    else:
        palette = plt.get_cmap("turbo")(np.linspace(.05, .95, len(methods)))
    return dict(zip(methods, palette))


def save_training_plot(records, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    episodes = [record["episode"] for record in records]
    series = (
        ("episode_reward", "累计奖励", "奖励", "#e41a1c"),
        ("task_success_rate", "任务完成率", "完成率", "#ff9800"),
        ("avg_delay_s", "任务平均完成时间", "秒", "#2e7d32"),
        ("avg_energy_j", "任务平均能耗", "J", "#1565c0"),
        ("load_balance_degree", "负载均衡度", "均衡度", "#7b1fa2"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for axis, (field, title, unit, color) in zip(axes.flat, series):
        values = np.asarray([record[field] for record in records], dtype=float)
        window = min(20, max(1, len(values) // 10))
        smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
        axis.plot(episodes, values, color=color, alpha=0.18, linewidth=0.9)
        axis.plot(episodes[window - 1:], smoothed, color=color, linewidth=2.0,
                  marker="o" if len(smoothed) == 1 else None, markersize=4)
        axis.set_xlim(0, max(episodes) + max(.5, max(episodes) * .02))
        axis.set_title(title)
        axis.set_xlabel("训练回合")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_evaluation_plot(records, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    episodes = [record["episode"] for record in records]
    series = (
        ("objective_score", "验证目标分数", "分数", "#e41a1c"),
        ("task_success_rate", "验证任务完成率", "完成率", "#ff9800"),
        ("avg_delay_s", "验证平均完成时间", "秒", "#2e7d32"),
        ("avg_energy_j", "验证平均任务能耗", "J", "#1565c0"),
        ("mean_load_variance", "验证负载方差", "方差", "#7b1fa2"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for axis, (field, title, unit, color) in zip(axes.flat, series):
        means = np.asarray([record[f"{field}_mean"] for record in records], dtype=float)
        stds = np.asarray([record[f"{field}_std"] for record in records], dtype=float)
        axis.plot(episodes, means, color=color, linewidth=2.0, marker="o", markersize=4)
        if len(records) == 1:
            axis.errorbar(episodes, means, yerr=stds, color=color, fmt="none", capsize=3)
        axis.fill_between(episodes, means - stds, means + stds, color=color, alpha=0.18)
        axis.set_xlim(0, max(episodes) + max(.5, max(episodes) * .02))
        axis.set_title(title)
        axis.set_xlabel("训练回合")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def load_jsonl(path):
    with open(Path(path), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def comparison_records(run_dir, models, filename):
    records = {}
    for model in models:
        path = run_dir / model / filename
        if path.is_file():
            rows = load_jsonl(path)
            if rows:
                records[model] = rows
    return records


def save_run_plots(run_dir):
    run_dir = Path(run_dir)
    training_records = load_jsonl(run_dir / "train_metrics.jsonl")
    evaluation_records = load_jsonl(run_dir / "evaluation_metrics.jsonl")
    if training_records:
        save_training_plot(training_records, run_dir / "training_metrics.png")
    if evaluation_records:
        save_evaluation_plot(evaluation_records, run_dir / "evaluation_metrics.png")


def save_comparison_plots(run_dir, models):
    """叠加已有的多算法训练和固定验证记录。"""
    run_dir = Path(run_dir)
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    colors = method_colors(models)
    plots = (
        ("train_metrics.jsonl", "training_comparison.png", "训练对比", (
            ("episode_reward", "累计奖励", "奖励"),
            ("task_success_rate", "任务完成率", "完成率"),
            ("avg_delay_s", "任务平均完成时间", "秒"),
            ("avg_energy_j", "任务平均能耗", "J"),
            ("load_balance_degree", "负载均衡度", "均衡度"),
        )),
        ("evaluation_metrics.jsonl", "evaluation_comparison.png", "固定验证对比", (
            ("objective_score", "目标分数", "分数"),
            ("task_success_rate", "任务完成率", "完成率"),
            ("avg_delay_s", "平均完成时间", "秒"),
            ("avg_energy_j", "平均任务能耗", "J"),
            ("mean_load_variance", "负载方差", "方差"),
        )),
    )
    for filename, output, title, fields in plots:
        records = comparison_records(run_dir, models, filename)
        if not records:
            continue
        figure, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
        validation = filename == "evaluation_metrics.jsonl"
        for axis, (field, name, unit) in zip(axes.flat, fields):
            for model_type, rows in records.items():
                episodes = [row["episode"] for row in rows]
                key = field + "_mean" if validation else field
                values = np.asarray([row[key] for row in rows], dtype=float)
                color = colors[model_type]
                label = METHOD_LABELS.get(model_type, model_type)
                if validation:
                    axis.plot(episodes, values, color=color, label=label, linewidth=2,
                              marker="o" if model_type == "gatppo" else "s", markersize=4)
                    std = np.asarray([row[field + "_std"] for row in rows], dtype=float)
                    axis.fill_between(episodes, values - std, values + std, color=color, alpha=.12)
                    if len(rows) == 1:
                        axis.errorbar(episodes, values, yerr=std, color=color, fmt="none", capsize=3)
                else:
                    window = min(20, max(1, len(values) // 10))
                    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
                    axis.plot(episodes, values, color=color, alpha=.18, linewidth=.9)
                    axis.plot(episodes[window-1:], smoothed, color=color,
                              label=label, linewidth=2,
                              marker="o" if len(smoothed) == 1 else None, markersize=4)
            last_episode = max(row["episode"] for rows in records.values() for row in rows)
            axis.set_xlim(0, last_episode + max(.5, last_episode * .02))
            axis.set_title(name)
            axis.set_xlabel("训练回合")
            axis.set_ylabel(unit)
            axis.legend()
            axis.grid(alpha=.25)
        axes.flat[-1].axis("off")
        axes.flat[-1].text(.1, .7, "固定验证阴影：验证种子间标准差" if validation
                         else "细线：原始回报/指标；粗线：滑动平均", transform=axes.flat[-1].transAxes)
        figure.suptitle(title)
        figure.savefig(run_dir / output, dpi=160)
        plt.close(figure)


def save_method_bar_plot(summaries, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    methods = list(summaries)
    colors = method_colors(methods)
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (field, title, unit) in zip(axes.flat, CORE_METRICS):
        for index, (method, summary) in enumerate(summaries.items()):
            if summary[f"{field}_mean"] is not None:
                axis.bar(index, summary[f"{field}_mean"], yerr=summary[f"{field}_std"],
                         color=colors[method], capsize=3)
        axis.set_xticks(range(len(methods)), [METHOD_LABELS[method] for method in methods],
                        rotation=28, ha="right")
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_rl_reward_plot(run_dir, models, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    colors = method_colors(models)
    figure, (training_axis, validation_axis) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True,
    )
    training = comparison_records(run_dir, models, "train_metrics.jsonl")
    for model, rows in training.items():
        episodes = [row["episode"] for row in rows]
        rewards = np.asarray([row["episode_reward"] for row in rows])
        smoothed = [
            rewards[max(0, index - 19):index + 1].mean() for index in range(len(rewards))
        ]
        training_axis.plot(episodes, rewards, color=colors[model], alpha=.25, linewidth=.8)
        training_axis.plot(episodes, smoothed, color=colors[model], linewidth=2,
                           marker="o" if len(rows) == 1 else None, label=METHOD_LABELS[model])
    validation = comparison_records(run_dir, models, "evaluation_metrics.jsonl")
    for model, rows in validation.items():
        episodes = [row["episode"] for row in rows]
        means = np.asarray([row["objective_score_mean"] for row in rows])
        stds = np.asarray([row["objective_score_std"] for row in rows])
        validation_axis.errorbar(episodes, means, yerr=stds, color=colors[model],
                                 marker="o", capsize=3, label=METHOD_LABELS[model])
    for axis, title, unit, records in (
        (training_axis, "训练累计奖励", "奖励", training),
        (validation_axis, "验证目标分数", "分数", validation),
    ):
        axis.set_title(title)
        axis.set_xlabel("训练回合")
        axis.set_ylabel(unit)
        axis.grid(alpha=.25)
        if records:
            axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)
