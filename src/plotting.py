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
        axis.plot(episodes[window - 1:], smoothed, color=color, linewidth=2.0)
        axis.set_xlim(0, max(episodes))
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
        axis.plot(episodes, means, color=color, linewidth=2.0)
        axis.fill_between(episodes, means - stds, means + stds, color=color, alpha=0.18)
        axis.set_xlim(0, max(episodes))
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
