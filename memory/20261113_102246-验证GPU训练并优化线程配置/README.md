# 验证 GPU 训练并优化线程配置

## 对话目标

核实 500 轮训练较慢是否应通过将环境全部迁移到 GPU 解决，完成 RTX 5060 可用的 PyTorch GPU 环境、代码设备支持、CPU 线程调优与实际吞吐对比。

## 已完成事项

- 确认原 `gatppo` 环境是 `PyTorch 2.6.0 cpu_generic`，无法使用 RTX 5060。
- 使用 Conda Forge 创建独立 `gatppo-gpu` 环境，安装 `PyTorch 2.14.1` CUDA 13.4 构建；实测 `torch.cuda.is_available()=True`，识别 `NVIDIA GeForce RTX 5060 Laptop GPU`、compute capability `12.0`。
- 新增 `environment-gpu.yml`，使 GPU 环境可重建，且不依赖 Anaconda 默认频道。
- GAT-PPO 观测张量、PPO 更新张量、模型、评估与 checkpoint 支持 CPU/CUDA 设备，训练配置快照记录实际设备和线程数。
- 将原先固定的 32 个 PyTorch CPU 线程改为 `--threads`，默认 4；训练、评估和组合实验的默认设备设为实测更快的 CPU。
- 保留 `--device cuda` 和 `--device auto` 用于设备对照与后续多环境批量版本。
- 同步 README 和环境开发文档，明确物理环境中的路径枚举、候选动作和约束分支仍保留在 CPU。
- 保留并验证工作区已有的显式服务部署矩阵、逐边缘异构算力矩阵及配套 deadline 松弛区间修改，未回滚或覆盖并行改动。

## 关键性能证据

当前最终配置下，统一使用 240 任务、1 次 PPO epoch，取预热后一次的采样与更新总时间：

- CPU 4 线程，PyTorch 2.6.0：`0.702s`。
- CPU 32 线程，PyTorch 2.6.0：`2.354s`。
- RTX 5060，PyTorch 2.14.1 CUDA：`2.287s`。

CPU 4 线程比 CPU 32 线程约快 3.35 倍，比当前 CUDA 串行实现约快 3.26 倍。原因是每步只有 7 个图节点，且变长候选动作必须逐任务生成；小张量 CPU→GPU 拷贝与频繁内核启动的成本超过了神经网络计算量。

## 验证证据

- `gatppo` CPU 环境：23 项单元测试全部通过。
- `gatppo-gpu` GPU 环境：23 项单元测试全部通过。
- CPU 默认入口完成 1 轮临时训练。
- CUDA 入口完成 1 轮临时训练，并从 CUDA checkpoint 成功续训到第 2 轮。
- 所有开发验证使用系统临时目录并自动删除，测试前后 `results/` 文件清单完全一致。
- `git diff --check` 通过。

## 未解决事项和实现边界

- 当前 `EdgeEnv` 是带动态多跳路径、资源预留和变长动作集的串行 Python/NumPy 环境；将这些分支逐项改写成 CUDA 不会自动获得加速，且可能改变环境动力学。
- 本次未运行新的 500 轮正式训练，不对收敛或四项优化指标做新结论。

## 后续建议与命令

当前正式 500 轮建议使用：

```bash
conda run -n gatppo python src/train.py --episodes 500 --device cpu --threads 4
```

若要让 GPU 真正发挥吞吐，下一阶段应实现多个独立 `EdgeEnv` 并行采样，将多个环境在同一调度步的图观测和候选动作 padding 后批量前向，再一次性执行 PPO 更新。
