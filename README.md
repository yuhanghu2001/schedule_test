# schedule_test — 动态多跳云边端异构任务调度

总结：用 GAT（图注意力网络）+ PPO 强化学习，在"云—边—端"三层异构计算环境中，动态决定每个任务在哪个节点执行、用哪种芯片资源、走哪条多跳路径，从而联合优化任务完成率、平均完成时间、平均任务能耗和资源负载均衡度。

解决的问题：场景是 20 个端侧设备（传感器/相机/车等）持续产生 5 类智能任务（图像识别、目标检测、视频分析、语音处理、信号计算）。任务可以：在本地端侧执行、卸载到接入的边缘服务器、经边缘间的多跳中继转发到别的区域边缘执行、云端。难点在于一个算力强的目标卸载节点，可能需要经过多跳到达，所以"选执行节点"和"选传输路径"是耦合的。GAT 负责看懂这张动态拓扑图，PPO 负责在可行动作里做联合打分。

## 结果目录规范

`results/` 只保存正式训练及其配套正式评估/组合实验的产物。正式运行按北京时间新建 `results/YYYYMMDD_HHMMSS/`，统一保存 checkpoint、日志、指标、图表和配置快照。单元测试、smoke、少轮开发验证和临时诊断使用系统临时目录，完成后自动删除，不写入 `results/`。

```text
results/
└── 20261111_223015/
    ├── best_gat_ppo.pt
    ├── last_gat_ppo.pt
    ├── train.log
    ├── train_metrics.jsonl
    ├── training_metrics.png
    ├── evaluation_metrics.jsonl
    ├── evaluation_metrics.png
    ├── metrics.json
    ├── evaluate.log
    └── config.json
```

## 训练设备与线程

RTX 50 系显卡需要支持 Blackwell 的 PyTorch 2.7 或更高版本。本机已按 `environment-gpu.yml` 创建独立的 `gatppo-gpu` 环境；GPU 对照可用 `conda run -n gatppo-gpu python src/train.py --device cuda --threads 4` 启动。`--device auto` 会在 CUDA 可用时选择 GPU，训练配置快照会记录实际设备和 CPU 线程数。

训练过程可用 `Ctrl+C` 手动中断。训练入口会先关闭并刷新 JSONL，再根据已完成回合重建 `training_metrics.png` 和已有的 `evaluation_metrics.png`，同时保存 last checkpoint。

环境中的候选动作枚举、多跳路径和物理约束仍是顺序相关的 Python/NumPy 逻辑，保留在 CPU；GAT-PPO 前向和参数更新可运行在 GPU。当前图只有7个节点，同一份240任务、1次PPO epoch的预热后基准中，CPU 4线程为0.702秒，CPU 32线程为2.354秒，RTX 5060为2.287秒，因此默认设备为CPU、默认线程数为4。GPU 主要用于后续“多环境并行 + 批量前向”版本，当前串行版本可用 `--device cuda` 做设备对照，但不建议用它跑正式500轮训练。

## 文件概览

```text
schedule_test/
├── AGENTS.md                         # 严格代码规范：简单、无防御式编程
├── README.md
├── requirements.txt
├── environment-gpu.yml               # RTX 50 系独立 GPU 环境
├── configs/
│   ├── default.json                  # 研究默认参数
│   ├── small_smoke.json              # 轻量快速运行
│   └── high_congestion.json          # 高拥塞场景
├── docs/
│   ├── 01_环境开发说明书.md             # 从建模到代码与交付验收
│   ├── 02_数学模型与C1-C24约束.md       # 24条约束的统一正确表达
│   ├── 03_实验设计与指标.md             # 场景、消融、泛化、实验协议
│   ├── 04_数据字典与实现边界.md         # 关键变量、原型简化项
│   └── 05_训练奖励与PPO对齐说明.md      # 可加奖励、全局状态与固定评估
├── memory/                           # AI 对话与工作记录
├── src/
│   ├── train.py                      # 完整训练、续训、日志和绘图入口
│   ├── config.py                     # 配置
│   ├── tasks.py                      # 任务生成
│   ├── topology.py                   # 动态连通图与简单路由
│   ├── continuous.py                 # Dinkelbach 功率与 KKT 算力子问题
│   ├── ablation.py                   # 打乱图结构消融环境
│   ├── baselines.py                  # 最短路径+最快资源启发式
│   ├── environment.py                # 状态、合法动作、更新、统计
│   ├── evaluation.py                 # 固定种子贪心评估与指标聚合
│   ├── plotting.py                   # 训练/验证曲线和横轴格式
│   ├── models/
│   │   ├── gat.py                     # 两层边特征GAT
│   │   └── policy.py                  # GAT-PPO Actor/Critic
│   └── algorithms/ppo.py             # GAE + PPO-Clip
├── scripts/
│   ├── inspect_env.py                # 查看邻接矩阵和可行动作
│   ├── smoke.py                      # 环境运行示例
│   ├── evaluate.py                   # 模型推理评估
│   ├── evaluate_heuristic.py         # 启发式评估
│   └── experiment.py                 # 多种子、消融和跨场景聚合
└── results/                          # 按 YYYYMMDD_HHMMSS 隔离每次测试产物
```

## AI 工作记录

Codex、Claude 等 AI 的每次完整工作记录保存在 `memory/`，目录名统一为 `YYYYMMDD_HHMMSS-对话内容标题/`。目录中的 `README.md` 应说明对话目标、完成事项、关键修改、验证证据、未解决事项和后续建议。例如：`memory/20261112_140000-检查代码完整性并训练GAT-PPO/`。

## 重要实现边界

- **已实现**：动态稀疏拓扑、DFS有效简单路径、正反向共享链路容量、异构资源匹配、时限过滤、逐任务奖励、shuffled mini-batch PPO、完整状态续训与独立种子评估。
- **任务时限**：每个终端先抽取实际无线速率，deadline 按任务输入上传、结果回传和最慢边缘算力计算参考处理时间，再乘配置化松弛系数；不做时隙长度截断。该参考时间仍不包含排队和多跳传输，因此不等同于完整场景的可行性保证。
- **外部调度**：默认 60% 任务设置 `must_offload`，这类任务不生成源端本地候选，只能由部署了对应服务的边缘或云端执行，或被合法拒绝。
- **连续子问题**：固定离散执行节点、异构资源和路径后，上行功率通过 Dinkelbach 法求解 Shannon 能效分式子问题；同节点同资源并发任务通过带截止时间最低算力约束的 KKT active-set 分配。两者只保证各自连续子问题最优，**不表示整个四目标混合离散–连续问题达到全局最优。**
- **模型差异**：数学文档给出完整 C1–C24 正式约束体系；代码采用“生成可行简单路径/动作”的实现方式隐式满足路由流守恒和无环条件，而非调用 MILP 求解器求完整原问题。
- **结果差异**：训练 reward 使用 `4:2:1:1` 的可加增量形式，未折扣时满足 `episode_reward/N=objective_score`；实验仍分别记录四项完整 episode 指标。
- **实验差异**：仓库提供四方法、多种子和跨场景运行入口；只有真实完成相同预算实验后才能报告均值、方差或性能提升。

阅读次序：`AGENTS.md` → `docs/01_环境开发说明书.md` → `docs/02_数学模型与C1-C24约束.md`。
