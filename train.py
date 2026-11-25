"""根目录极简训练入口：参数统一在文件顶部配置，命令行仅选算法与场景。

示例：
    python train.py --dual_gatppo --normal   # 常规场景（不传场景开关时也默认常规场景）
    python train.py --dual_gatppo --high     # 高拥塞场景
    python train.py --single_gatppo --high
    python train.py --dual_gatppo --ppo --high
"""
import argparse
import json
from dataclasses import asdict, replace

from src.config import load_config
from src.trainer import new_run_dir, repository_path, save_comparison_plots, train_model

# ==================== 训练参数统一配置区 ====================
EPISODES = 500
DAG_PRESET = "hetero_dag_default"                 # 异构 DAG 常规场景
DAG_HIGH_PRESET = "hetero_dag_high_congestion"    # 异构 DAG 高拥塞场景
LEGACY_PRESET = "default"                         # 传统单任务常规场景
LEGACY_HIGH_PRESET = "high_congestion"            # 传统单任务高拥塞场景
DEVICE = "auto"                                   # "auto" / "cuda" / "cpu"
THREADS = 2
SEED = None                                       # 设为整数可固定种子，None 使用预设默认种子
RESUME = None                                     # 续训 checkpoint 路径，None 表示从头训练
RUN_DIR = None                                    # 默认按北京时间自动创建 results/YYYYMMDD_HHMMSS/
# ===========================================================

MODELS = (
    "dual_gatppo",
    "single_gatppo",
    "gatppo",
    "ppo",
    "shuffled_gat",
    "dqn",
    "ddpg",
)
DAG_MODELS = {"dual_gatppo", "single_gatppo"}


def main():
    parser = argparse.ArgumentParser(description="极简训练入口，训练参数在 train.py 顶部配置")
    for name in MODELS:
        parser.add_argument(f"--{name}", action="store_true", help=f"训练 {name}")
    parser.add_argument("--normal", action="store_true", help="使用常规场景（默认）")
    parser.add_argument("--high", action="store_true", help="使用高拥塞场景")
    args = parser.parse_args()

    selected = [name for name in MODELS if getattr(args, name)] or ["dual_gatppo"]
    use_dag = any(name in DAG_MODELS for name in selected)
    if use_dag:
        preset = DAG_HIGH_PRESET if args.high else DAG_PRESET
    else:
        preset = LEGACY_HIGH_PRESET if args.high else LEGACY_PRESET
    cfg = load_config(preset)
    if SEED is not None:
        cfg = replace(cfg, seed=SEED)

    run_dir = repository_path(RUN_DIR) if RUN_DIR else new_run_dir()
    if len(selected) == 1:
        train_model(
            cfg, selected[0], run_dir, episodes=EPISODES,
            resume=repository_path(RESUME) if RESUME else None,
            device=DEVICE, threads=THREADS, config_source=preset,
            existing_dir=bool(RUN_DIR),
        )
        return

    run_dir.mkdir(parents=True, exist_ok=bool(RUN_DIR))
    with (run_dir / "comparison_config.json").open("x", encoding="utf-8") as file:
        json.dump({
            "models": selected, "config_source": preset, "config": asdict(cfg),
            "episodes": EPISODES, "device": DEVICE, "torch_threads": THREADS,
        }, file, ensure_ascii=False, indent=2)
    try:
        for model_type in selected:
            if train_model(
                cfg, model_type, run_dir / model_type, episodes=EPISODES,
                device=DEVICE, threads=THREADS, config_source=preset,
            ):
                break
    finally:
        save_comparison_plots(run_dir, selected)
    print(f"对比结果目录：{run_dir}")


if __name__ == "__main__":
    main()
