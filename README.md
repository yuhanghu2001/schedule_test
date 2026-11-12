# schedule_test — 动态多跳云边端异构任务调度

一个**可运行的研究原型**：20 个端侧节点、6 个区域边缘服务器、1 个云端服务器；动态稀疏边缘网络；5 类智能任务；CPU/NPU/FPGA 异构资源；GAT 提取拓扑/链路与邻域资源特征，PPO 联合选择任务执行节点、异构资源和多跳路径。

核心思想：围绕任务完成率、平均完成时间、平均任务能耗和资源负载均衡度四个指标进行联合优化；采用接收、服务、算力、容量、时限、路径和共享带宽等约束生成可行调度动作。

## 结果目录规范

每次运行 smoke、训练、评估或组合实验前，都必须按北京时间新建独立目录 `results/YYYYMMDD_HHMMSS/`。本次运行的 checkpoint、终端日志、评估指标、图表和配置快照统一保存在该目录中；不要把新产物直接写入 `results/` 根目录，也不要复用或覆盖历史目录。评估已有 checkpoint 时，在本次目录的日志中保留原 checkpoint 路径。

```text
results/
└── 20261111_223015/
    ├── gat_ppo.pt
    ├── train.log
    ├── train_metrics.jsonl
    ├── training_metrics.png
    ├── metrics.json
    ├── evaluate.log
    └── config.json
```

## 文件概览

```text
schedule_test/
├── AGENTS.md                         # 严格代码规范：简单、无防御式编程
├── README.md
├── requirements.txt
├── configs/
│   ├── default.json                  # 研究默认参数
│   ├── small_smoke.json              # 轻量快速运行
│   └── high_congestion.json          # 高拥塞场景
├── docs/
│   ├── 01_环境开发说明书.md             # 从建模到代码与交付验收
│   ├── 02_数学模型与C1-C24约束.md       # 24条约束的统一正确表达
│   ├── 03_实验设计与指标.md             # 场景、消融、泛化、实验协议
│   └── 04_数据字典与实现边界.md         # 关键变量、原型简化项
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
- **连续子问题**：固定离散执行节点、异构资源和路径后，上行功率通过 Dinkelbach 法求解 Shannon 能效分式子问题；同节点同资源并发任务通过带截止时间最低算力约束的 KKT active-set 分配。两者只保证各自连续子问题最优，**不表示整个四目标混合离散–连续问题达到全局最优。**
- **模型差异**：数学文档给出完整 C1–C24 正式约束体系；代码采用“生成可行简单路径/动作”的实现方式隐式满足路由流守恒和无环条件，而非调用 MILP 求解器求完整原问题。
- **结果差异**：训练 reward 按 `2:2:1:1` 使用完成率、平均完成时间、平均任务能耗和负载不均衡度的运行值；实验仍分别记录四项完整 episode 指标。
- **实验差异**：仓库提供四方法、多种子和跨场景运行入口；只有真实完成相同预算实验后才能报告均值、方差或性能提升。

阅读次序：`AGENTS.md` → `docs/01_环境开发说明书.md` → `docs/02_数学模型与C1-C24约束.md`。
