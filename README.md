# schedule_test — 动态多跳云边端异构任务调度

一个**可运行的研究原型**：20 个端侧节点、6 个区域边缘服务器、1 个云端服务器；动态稀疏边缘网络；5 类智能任务；CPU/NPU/FPGA 异构资源；GAT 提取拓扑/链路与邻域资源特征，PPO 联合选择任务执行节点、异构资源和多跳路径。

核心思想：保留四目标（任务完成率、优先级加权完成时延、总能耗、负载不均衡）；采用接收/服务/算力/容量/时延/路径/共享带宽等条件产生可行候选调度动作。

## 快速运行

在 ZIP 解压后进入 `schedule_test` 目录：

最简单的训练方式：

```bash
conda activate gatppo
python src/train.py
```

默认使用 `configs/small_smoke.json` 训练 500 个 episode，并自动创建北京时间结果目录。终端每行先显示当前 `episode`，再显示 `累计奖励`、`任务完成率`、`任务平均完成时间`（秒）、`任务平均能耗`（J）和 `负载均衡度`。训练结束后，同一结果目录会生成 `training_metrics.png`；每项指标包含浅色真实值曲线和深色滑动窗口曲线，不绘制圆点和图例。无论当前终端位于仓库根目录还是 `src/`，配置和结果路径都以仓库根目录为基准。

需要同时执行 smoke 和评估时：

```bash
conda activate gatppo
python -m pip install -r requirements.txt
# macOS / Linux
run_dir="results/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir "$run_dir"
cp configs/small_smoke.json "$run_dir/config.json"
python -m scripts.smoke --config configs/small_smoke.json --run-dir "$run_dir" | tee "$run_dir/smoke.log"
python src/train.py --config configs/small_smoke.json --episodes 3 --run-dir "$run_dir" | tee "$run_dir/train.log"
python -m scripts.evaluate --config configs/small_smoke.json --checkpoint "$run_dir/gat_ppo.pt" --run-dir "$run_dir" | tee "$run_dir/evaluate.log"
```

Windows PowerShell：

```powershell
conda activate gatppo
$beijingNow = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date).ToUniversalTime(), 'China Standard Time')
$runDir = "results/$($beijingNow.ToString('yyyyMMdd_HHmmss'))"
New-Item -ItemType Directory -Path $runDir | Out-Null
Copy-Item configs/small_smoke.json "$runDir/config.json"
python -m scripts.smoke --config configs/small_smoke.json --run-dir $runDir | Tee-Object -FilePath "$runDir/smoke.log"
python src/train.py --config configs/small_smoke.json --episodes 3 --run-dir $runDir | Tee-Object -FilePath "$runDir/train.log"
python -m scripts.evaluate --config configs/small_smoke.json --checkpoint "$runDir/gat_ppo.pt" --run-dir $runDir | Tee-Object -FilePath "$runDir/evaluate.log"
```

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

切换到主规模任务：

```bash
run_dir="results/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir "$run_dir"
cp configs/default.json "$run_dir/config.json"
python src/train.py --config configs/default.json --episodes 100 --run-dir "$run_dir" | tee "$run_dir/train.log"
```

同一动作/奖励/环境下的**无GAT消融**（普通节点编码PPO）：

```bash
run_dir="results/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir "$run_dir"
cp configs/default.json "$run_dir/config.json"
python src/train.py --config configs/default.json --episodes 100 --model flat --run-dir "$run_dir" --output flat_ppo.pt | tee "$run_dir/train.log"
python -m scripts.evaluate --config configs/default.json --checkpoint "$run_dir/flat_ppo.pt" --run-dir "$run_dir" | tee "$run_dir/evaluate.log"
```

四种方法的多随机种子聚合实验（可用不同评估配置做跨拓扑或跨拥塞测试）：

```bash
python -m scripts.experiment \
  --config configs/default.json \
  --evaluation-config configs/high_congestion.json \
  --episodes 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

该入口比较 `gat`、`flat`、`shuffled_gat` 和 `heuristic`，并在本次结果目录的 `summary.json` 中保存均值、标准差和有效样本数。它提供可复现的实验流程，不代表仓库已经跑完论文规模实验或得到性能提升结论。

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
- **结果差异**：实验指标与训练奖励分开记录；固定权重奖励是四目标的训练代理，不声称是加权目标的严格等价实现。
- **实验差异**：仓库提供四方法、多种子和跨场景运行入口；只有真实完成相同预算实验后才能报告均值、方差或性能提升。

阅读次序：`AGENTS.md` → `docs/01_环境开发说明书.md` → `docs/02_数学模型与C1-C24约束.md` → 运行示例。
