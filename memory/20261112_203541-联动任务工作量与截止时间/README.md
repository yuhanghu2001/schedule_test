# 联动任务工作量与截止时间

## 目标

将 `src/tasks.py` 中与工作量无关的 deadline 独立抽样，改为根据输入数据量、计算量和可配置松弛系数联动生成，并同步代码、配置、测试与文档。

## 完成事项

- 先更新数学模型假设，增加工作量联动 deadline 公式和实现边界。
- `sample_tasks` 先抽样 `data_mb` 和 `compute_gflop`，再根据最快无线上行参考时间、最慢边缘计算参考时间及松弛系数生成 `deadline_s`。
- 新增 `deadline_slack_min=1.2` 和 `deadline_slack_max=2.5`，并同步三套 JSON 配置。
- 环境使用配置的最大上行速率和实际最小边缘算力传入任务采样函数。
- 候选动作时延和任务 deadline 观测统一改用 `reward_delay_ref` 归一化，不再硬编码 `0.5`。
- 同步 `AGENTS.md`、`README.md` 和环境、实验、数据字典文档，清除“0.5 秒是最大 deadline”和“deadline 独立抽样”的旧表述。
- 新增任务采样回归测试，验证每个任务的实际松弛倍数均在配置区间内。

## 关键修改

`deadline_s = (data_mb * 8 / radio_rate_max_mbps + compute_gflop / min_edge_compute_gflops) * slack`，其中 `slack ~ U[deadline_slack_min, deadline_slack_max]`。

该参考时间没有包含下行回传、排队、多跳中继和信道恶化，因此用于避免 deadline 与工作量脱节，不声称保证所有任务实际可行。

## 验证证据

- `python -m unittest discover -s tests -v`：22 项测试全部通过。
- `python -m scripts.smoke`：36 任务 smoke 运行完成，临时产物已自动删除。
- 测试前后 `results/` 文件清单一致，未新增、删除或覆盖正式训练结果。
- `git diff --check` 通过，三套 JSON 配置均通过语法检查。

## 未解决事项

- 本次未进行新的正式训练，因此不对修改后的最终完成率或收敛效果作实验结论。

## 后续建议

- 正式训练时固定其他配置，对 `deadline_slack_min/max` 开展松弛度消融，并同时比较完成率、平均完成时延、平均任务能耗和负载不均衡度。
