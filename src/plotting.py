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
    "dual_gatppo": "Dual GAT-PPO", "single_gatppo": "Single GAT-PPO",
    "heft_hetero": "Heterogeneous HEFT", "greedy": "Greedy", "heuristic": "Shortest Fastest",
}
CORE_METRICS = (
    ("task_success_rate", "任务完成率", "比例"),
    ("avg_delay_s", "任务平均完成时间", "秒"),
    ("avg_energy_j", "任务平均能耗", "J"),
    ("load_balance_degree", "负载均衡度", "均衡度"),
)
DAG_PLOT_METRICS = (
    ("dag_completion_rate", "DAG 完成率", "比例"),
    ("avg_dag_makespan_s", "DAG 平均完成时间", "秒"),
    ("avg_dag_energy_j", "DAG 平均能耗", "J"),
    ("hetero_load_balance_degree", "异构负载均衡度", "均衡度"),
    ("cpu_load_std", "CPU 负载标准差", "标准差"),
    ("npu_load_std", "NPU 负载标准差", "标准差"),
    ("fpga_load_std", "FPGA 负载标准差", "标准差"),
    ("fpga_reconfig_rate", "FPGA 重构率", "比例"),
)
CURVE_COLORS = (
    "#e41a1c", "#ff9800", "#2e7d32", "#1565c0", "#7b1fa2",
    "#14676c", "#9c432e", "#4d56a1", "#a95a15",
)


def curve_metrics(legacy, record, suffix=""):
    dag_core = DAG_PLOT_METRICS[:len(CORE_METRICS)]
    core = tuple(dag if dag[0] + suffix in record else field
                 for field, dag in zip(legacy[1:], dag_core))
    extra = tuple(field for field in DAG_PLOT_METRICS[len(CORE_METRICS):]
                  if field[0] + suffix in record)
    return (legacy[0], *core, *extra)


def curve_axes(fields):
    rows = (len(fields) + 1) // 2
    return plt.subplots(rows, 2, figsize=(12, 4 * rows), constrained_layout=True)


def method_colors(methods):
    if len(methods) <= 10:
        palette = plt.get_cmap("tab10").colors
    else:
        palette = plt.get_cmap("turbo")(np.linspace(.05, .95, len(methods)))
    return dict(zip(methods, palette))


def save_training_plot(records, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    episodes = [record["episode"] for record in records]
    series = curve_metrics((
        ("episode_reward", "累计奖励", "奖励"),
        ("task_success_rate", "任务完成率", "完成率"),
        *CORE_METRICS[1:],
    ), records[0])
    figure, axes = curve_axes(series)
    for axis, (field, title, unit), color in zip(axes.flat, series, CURVE_COLORS):
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
    for axis in axes.flat[len(series):]:
        axis.axis("off")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_evaluation_plot(records, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    episodes = [record["episode"] for record in records]
    series = curve_metrics((
        ("objective_score", "验证目标分数", "分数"),
        ("task_success_rate", "验证任务完成率", "完成率"),
        ("avg_delay_s", "验证平均完成时间", "秒"),
        ("avg_energy_j", "验证平均任务能耗", "J"),
        ("mean_load_variance", "验证负载方差", "方差"),
    ), records[0], "_mean")
    figure, axes = curve_axes(series)
    for axis, (field, title, unit), color in zip(axes.flat, series, CURVE_COLORS):
        means = np.asarray([record[f"{field}_mean"] for record in records], dtype=float)
        stds = np.asarray([record[f"{field}_std"] for record in records], dtype=float)
        axis.plot(episodes, means, color=color, linewidth=2.0, marker="o", markersize=4)
        if len(records) == 1 and np.isfinite(means[0]):
            axis.errorbar(episodes, means, yerr=stds, color=color, fmt="none", capsize=3)
        axis.fill_between(episodes, means - stds, means + stds, color=color, alpha=0.18)
        axis.set_xlim(0, max(episodes) + max(.5, max(episodes) * .02))
        axis.set_title(title)
        axis.set_xlabel("训练回合")
        axis.set_ylabel(unit)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(series):]:
        axis.axis("off")
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
        validation = filename == "evaluation_metrics.jsonl"
        fields = curve_metrics(fields, next(iter(records.values()))[0], "_mean" if validation else "")
        rows = (len(fields) + 2) // 2  # 保留一个面板放对比图说明。
        figure, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows), constrained_layout=True)
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
                    if len(rows) == 1 and np.isfinite(values[0]):
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
        for axis in axes.flat[len(fields):]:
            axis.axis("off")
        axes.flat[-1].text(.1, .7, "固定验证阴影：验证种子间标准差" if validation
                         else "细线：原始回报/指标；粗线：滑动平均", transform=axes.flat[-1].transAxes)
        figure.suptitle(title)
        figure.savefig(run_dir / output, dpi=160)
        plt.close(figure)


def save_method_bar_plot(summaries, output):
    plt.rcParams["font.family"] = "Noto Sans CJK SC"
    methods = list(summaries)
    colors = method_colors(methods)
    is_dag = any("dag_completion_rate_mean" in summary for summary in summaries.values())
    fields = DAG_PLOT_METRICS if is_dag else CORE_METRICS
    rows = len(fields) // 2
    figure, axes = plt.subplots(rows, 2, figsize=(13, 4.5 * rows), constrained_layout=True)
    for axis, (field, title, unit) in zip(axes.flat, fields):
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
