# 对齐算力与联动任务时限

## 对话目标

将云、边、端 CPU/NPU/FPGA 算力改为实验配置字段，把边缘和云端算力提升到论文同量级；按任务输入量、计算量和配置化松弛系数联动生成 deadline，并以 0.95 秒保险截断。同步数学模型、开发文档、数据字典、实验配置和基本测试，完成 smoke、空时隙单任务判定与 50 轮短训练验证。

## 已完成事项

- 先修改 docs/02_数学模型与C1-C24约束.md，再修改代码与配置。
- 新增 edge_speed=[200,350,200]、cloud_speed=[8000,12000,8000]、local_speed=[12,16,6]，单位均为 GFLOP/s。
- EdgeEnv 不再硬编码云边端算力，直接从 Config 读取三组速度。
- 新增 deadline_ceiling_s=0.95；deadline 实现为 min((min_uplink + min_compute) * slack, deadline_ceiling_s)。
- 三套 JSON 配置、环境开发说明、数据字典和任务采样测试已同步。
- 文档明确：1 秒时隙是相对原文“连续时间＋队列”模型的简化，deadline 上限保证不跨时隙，当前未实现跨时隙剩余工作量。
- 保留并验证了工作区原有的 GPU/线程配置改动，未覆盖这些现有修改。

## 关键修改

deadline 参考时间为：

    min_uplink = data_mb * 8 / radio_rate_max_mbps
    min_compute = compute_gflop / min(edge_speed)
    deadline_s = min(
        (min_uplink + min_compute) * uniform(deadline_slack_min, deadline_slack_max),
        deadline_ceiling_s,
    )

默认 radio_rate_max_mbps=180、min(edge_speed)=200、deadline_slack_min=1.2、deadline_slack_max=2.5、deadline_ceiling_s=0.95。

## 验证证据

- /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover -s tests -v：22 项测试全部通过。
- PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python scripts/smoke.py：36 任务 smoke 完整运行，临时产物自动删除。
- 空时隙、参考最优无线速率 180 Mbps 下，600 任务中 562 项至少有一个可行动作，参考可行率为 93.67%。
- 随机实际信道 60–180 Mbps 下，同口径运行 600 任务的可行率为 74.00%。该数字与参考最优条件不同，不应混用。
- 50 轮 small_smoke GAT-PPO 临时训练：训练轨迹前 10 轮平均完成率 55.00%，后 10 轮 62.50%；固定 5 个验证种子的未训练贪心完成率 55.00%，第 50 轮为 57.78%；目标函数均值从 0.8191 升至 1.0232。临时训练目录已自动删除，未写入 results/。
- git diff --check 通过，三套 JSON 均通过语法解析。

## 未解决事项和实现边界

- 50 轮短训练只是开发验证；虽然训练轨迹和固定验证均有改善，但不足以声称统计显著提升。
- deadline 参考式使用 180 Mbps 最优无线速率，而运行时信道速率在 60–180 Mbps 间抽样；因此参考可行率不等于实际场景可行率。
- 本次未执行正式多种子训练，未生成或修改任何 results/ 正式结果。
- 当前验证环境为 PyTorch 2.6 CPU；工作区原有改动要求 PyTorch 2.7+ 以支持 RTX 50 系显卡，本次未在 CUDA/Blackwell 设备上验证。

## 后续建议

- 正式结论需在论文规模配置上运行多随机种子实验，比较固定验证集的完成率、时延、能耗、负载不均衡度和置信区间。
- 对 deadline 松弛度做消融时，应同时报告参考最优无线条件下的单任务可行率和随机实际信道下的运行可行率。
