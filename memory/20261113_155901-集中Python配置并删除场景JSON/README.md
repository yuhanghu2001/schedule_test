# 集中 Python 配置并删除场景 JSON

## 对话目标

解决环境参数分散在 `src/config.py`、`src/environment.py` 和三份场景 JSON 中、需要反复跳转查看的问题。可编辑参数统一收口到 `src/config.py`，环境状态转移和奖励计算保留在 `src/environment.py`。

## 已完成事项

- 删除 `configs/default.json`、`configs/high_congestion.json` 和 `configs/small_smoke.json`。
- `Config` dataclass 保存唯一完整默认参数，并按任务规模、拓扑链路、服务/资源、时限/卸载、连续优化、奖励、GAT-PPO 分组排列。
- `load_config("default")` 直接返回完整默认配置。
- `load_config("high_congestion")` 只覆盖拓扑周期、边数和拥塞区间。
- `load_config("small_smoke")` 只覆盖任务数、拓扑周期和 mini-batch 大小。
- 训练、评估、启发式、组合实验、smoke、环境查看和全部测试统一改用 `default|high_congestion|small_smoke` 名称。
- `src/environment.py` 的奖励求和处增加论文公式注释，便于从参数直接追踪到计算。
- README、AGENTS 和实验设计文档同步指明 `src/config.py` 是唯一参数源，禁止再增加场景 JSON 参数源。

## 新的使用方式

```bash
python src/train.py --config default
python src/train.py --config high_congestion
python -m scripts.smoke --config small_smoke
```

## 验证证据

- 将新三套 Python 配置与上一提交中的三份 JSON 逐字段比较，`default`、`high_congestion`、`small_smoke` 均完全一致。
- 29项单元测试全部通过。
- smoke 使用 `small_smoke` Python 配置完整运行，临时产物自动删除。
- `src/train.py --config small_smoke --episodes 1` 在系统临时目录完成训练、固定评估、绘图和 best/last checkpoint 保存；快照记录 `config_source=small_smoke`、`n_tasks=36`，临时目录已删除。

## 实现边界

- `results/*/train_config.json`、`evaluation_config.json` 等仍然保留。它们是实验产物中的不可变配置快照，用于复现和审计，不是训练的参数来源。
- checkpoint 内的 `config` 字典仍用于续训和评估时重建当时配置，不需要手工编辑。

## 后续建议

- 调参时只查看和修改 `src/config.py`；修改奖励计算逻辑时查看 `src/environment.py` 的 `step()`。
