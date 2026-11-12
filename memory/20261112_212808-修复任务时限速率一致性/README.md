# 修复任务时限速率一致性

## 对话目标

修复 deadline 生成使用全局最快无线速率、而环境执行使用源终端实际速率造成的口径不一致；将结果回传纳入 deadline 预算，删除 0.95 秒截断，并把随机实际信道下的空时隙单任务可行率提高到 98% 以上。

## 已完成事项

- 先更新数学模型，再修改任务生成、环境初始化、配置、测试及其他说明文档。
- EdgeEnv.reset 先为每个终端抽取 60–180 Mbps 的实际无线速率，再将整个速率数组传给 sample_tasks。
- sample_tasks 使用任务源终端的实际速率分别计算输入上传和结果回传时间，并加入最慢边缘算力下的计算时间。
- 删除 deadline_ceiling_s 配置字段、三套 JSON 配置项及任务生成中的截断逻辑。
- deadline 允许超过 1 秒；时隙只作为当前原型的共享资源记账窗口，不再作为 deadline 上限。
- 同步更新 AGENTS.md、README、数学模型、环境开发说明、数据字典和任务采样测试。

## 关键修改

新的 deadline 生成式为：

    radio_rate = radio_rates[source]
    min_uplink = data_mb * 8 / radio_rate
    min_download = output_mb * 8 / radio_rate
    min_compute = compute_gflop / min(edge_speed)
    deadline_s = (
        min_uplink + min_download + min_compute
    ) * uniform(deadline_slack_min, deadline_slack_max)

## 验证证据

- /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover -s tests -v：22 项测试全部通过。
- PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python scripts/smoke.py：36 个任务完整运行，任务完成率 88.89%，运行态可行任务率 100%，临时产物自动删除。
- 默认配置、600 个任务、每个任务独占空时隙资源时，592 个任务至少存在一个可行动作，可行率为 98.67%。
- 同一批任务中 100 个 deadline 大于 1 秒，deadline 最大值为 2.218827 秒，证明 0.95 秒截断已移除。
- git diff --check 通过，三套 JSON 配置均通过语法解析。
- 未生成或修改 results/ 下的正式训练结果。

## 未解决事项和实现边界

- 仍有 8/600 个空时隙任务不可行，原因来自 allow_offload、本地算力和服务支持等核心物理约束；本次没有放宽这些约束。
- deadline 预算包含源端无线输入、无线结果回传和最慢边缘计算，但仍不包含排队、多跳传输及动态拓扑开销。
- deadline 可以超过时隙长度，但当前环境未实现跨时隙剩余工作量和持续资源占用；时隙仍是简化的资源记账窗口。
- 本次只做环境、测试和 smoke 验证，没有进行新的正式训练，因此不报告算法性能结论。

## 后续建议

- 正式实验同时报告空时隙单任务可行率、运行态可行率和强制拒绝率，区分任务生成可行性与同槽资源竞争。
- 若后续需要严格模拟 deadline 超过时隙的任务，应单独实现跨时隙剩余工作量、链路占用和计算占用状态。
