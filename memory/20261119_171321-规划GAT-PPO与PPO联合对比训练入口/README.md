# 规划 GAT-PPO 与 PPO 联合对比训练入口

## 目标
用户要求先给方案，讨论新写脚本、修改 ablation.py 或 train.py 参数实现两算法一起跑。

## 核对发现
- src/train.py 已支持 --model gat/flat/shuffled_gat，承担单模型训练、验证、best/last、续训与曲线。
- src/ablation.py 已有 run_ablation 主入口，固定 GAT-PPO/gat 与 PPO/flat，按算法/种子顺序调用 train.py，评估 best 并生成 summary.json。
- src/models/policy.py 中 flat 是普通节点 MLP 编码加同一 PPO Actor/Critic，仍使用相同任务、候选路径和动作特征。
- scripts/experiment.py 也可 --methods gat flat，但当前评估 last，与 ablation 的 best 口径不同。
- 实验文档要求验证种子1042..1046选择best，最终独立测试2042..2051，正式完整对比至少10训练种子。
- 当前 ablation 最终每个训练种子仅测试一个 cfg.evaluation_seed+seed，没有双算法叠加图。

## 推荐方案
复用 src/ablation.py 作为两算法实验入口，保持 train.py 单模型功能，保留图打乱消融模块的既有接口。补充统一独立测试种子列表、最终多种子评估及对比图/表；先按每个训练种子汇总测试结果，再统计训练种子间均值/标准差。默认一条命令顺序完成两个算法，真正并发可作为明确标注的新可选参数，最多同时两个训练进程，避免一次启动全部种子。输出仍按北京时间 results 时间戳归档，各算法/训练种子目录独立。

## 验证证据
读取训练、消融、策略、评估、组合实验与实验协议实现。实际执行 conda run -n gatppo python -m src.ablation --help，确认当前参数可用。未启动训练、评估或新增 results。

## 未解决事项与后续建议
本轮只提供方案，未修改业务代码。多测试种子参数、叠加对比图和真正并发参数均为建议新增，不能当作当前已实现功能。提交推送未执行。
