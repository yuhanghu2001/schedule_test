# 诊断 CPU 训练速度并推送现有改动

## 基本信息

- 时间：2026-11-12 16:43:24（北京时间）
- 执行者：Codex
- 对话主题：诊断 GAT-PPO 训练速度、确认 PyTorch/CPU/GPU 环境、说明线程上限，并提交推送全部现有改动。

## 对话目标

1. 判断训练过慢是否由 CPU 版 PyTorch 导致。
2. 确认 `torch.set_num_threads` 的可用上限。
3. 按用户要求将工作区全部现有改动提交并推送。

## 完成事项

- 确认机器为 AMD Ryzen 9 8945HX，16 个物理核心、32 个逻辑线程。
- 确认机器可见 NVIDIA GeForce RTX 5060 Laptop GPU，但 `gatppo` 环境安装的是 PyTorch 2.6.0 CPU 构建，CUDA 不可用。
- 确认训练进程使用约一个逻辑核心，GPU 无计算进程。
- 定位主要串行热点：逐任务 PPO 前向、候选动作遍历、DFS 路径枚举、Dinkelbach 和 KKT 计算。
- 说明 PyTorch 线程数最高可按 32 个逻辑线程设置，但小张量 GAT 不一定因此加速，Python/环境串行逻辑不受该参数控制。
- 确认 `results/20261112_162310/` 的 500 episode 训练已完成并生成 checkpoint、配置、指标和曲线。

## 关键修改

- `src/train.py` 当前默认训练 500 episode，并将 `torch.set_num_threads` 设为 32。
- 工作区已有的 reward 归一化修改保持不变，与数学文档、开发文档和测试一起提交。

## 验证证据

- `results/20261112_164324/`：在 `gatppo` 环境运行 11 项基础测试，全部通过。
- `results/20261112_162310/`：500 episode GAT-PPO 训练完成，含 checkpoint、配置、完整指标日志和曲线。该训练使用归一化修改前的 reward，不可作为归一化 reward 的实验结果。

## 未解决事项

- 尚未将模型和张量迁移至 CUDA，仅安装 CUDA 版 PyTorch 不会自动使用 GPU。
- PPO 更新仍是 mini-batch 内逐样本前向，环境采样仍为单环境串行。
- 32 个 PyTorch 线程是可用上限而非已验证的最优值，需实测 8、16、32 线程的单 episode 耗时。

## 后续建议

- 优先将 PPO 更新改为真正的张量批处理，并缓存每个拓扑周期内的简单路径。
- 内存充足时再引入多环境并行采样；若引入 CUDA，同时完成模型、观测张量和 checkpoint 的设备管理。
