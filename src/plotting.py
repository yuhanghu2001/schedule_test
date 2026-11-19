"""训练与固定评估曲线。"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def save_run_plots(run_dir):
    run_dir = Path(run_dir)
    training_records = load_jsonl(run_dir / "train_metrics.jsonl")
    evaluation_records = load_jsonl(run_dir / "evaluation_metrics.jsonl")
    if training_records:
        save_training_plot(training_records, run_dir / "training_metrics.png")
    if evaluation_records:
        save_evaluation_plot(evaluation_records, run_dir / "evaluation_metrics.png")


def save_comparison_plots(run_dir, models):
    """叠加两算法的真实训练/固定验证记录；中断时只画已有数据。"""
    run_dir = Path(run_dir)
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    labels = {"gatppo": "GAT-PPO", "ppo": "PPO"}
    colors = {"gatppo": "#1565c0", "ppo": "#ef6c00"}
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
        records = {}
        for model_type in models:
            path = run_dir / model_type / filename
            if path.is_file():
                rows = load_jsonl(path)
                if rows:
                    records[model_type] = rows
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
                if validation:
                    axis.plot(episodes, values, color=color, label=labels[model_type], linewidth=2,
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
                              label=labels[model_type], linewidth=2,
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
        figure.suptitle(title + "（已完成回合）")
        figure.savefig(run_dir / output, dpi=160)
        plt.close(figure)
