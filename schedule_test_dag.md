# `schedule_test` 异构 DAG 协同调度升级开发任务清单

> **基线仓库**：[yuhanghu2001/schedule_test](https://github.com/yuhanghu2001/schedule_test)（`main` 分支，提交 `9020d74`）  
> **改造原则**：保留现有 12 边缘 + 1 云 + 29 条动态多跳链路（E0–E3 网关约束、AR(1) 拥塞漂移、时隙通断）与 5 类业务名称，在不推翻底层网络仿真与上位机数据库骨架的前提下，将系统升级为**异构 DAG 工作流 + 三维细粒度芯片资源 + 算子亲和/重构代价 + 双图 GAT-PPO + 联合无效动作掩码**架构。

---

## 一、现状与目标架构映射总览

| 改造层次 | 当前仓库实现（基线 `9020d74`） | 目标升级形态 | 涉及核心文件 |
| :--- | :--- | :--- | :--- |
| **任务层** | [`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py) 中 `Task` 为单点独立任务（5 类业务 `kind` 0~4，无内部子任务依赖） | 保留 5 类业务名称，每类升级为 4~6 个子任务的异构 `TaskDAG` 模板，含算子标签、兼容芯片集、传输数据量、向上关键度 `rank_u` 与松弛度 `Slack` | [`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py), [`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py) |
| **资源与拓扑层** | [`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py) 仅维护 `edge_speed_matrix` (12×3) 与粗粒度 `memory_gb`/`storage_gb` | 保留 4×3 八邻域 29 链路拓扑；引入 CPU/NPU/FPGA 三维细粒度专属资源池 + 时间片在途预留/归还 + 接入/中层/网关异构配比梯度 | [`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py), [`src/topology.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/topology.py), [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py) |
| **映射与代价层** | [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py) 用固定 `support` (3×5) 和 `[10, 18, 12]` 功耗，无亲和系数、无重构时延、无片内跨芯片通信 | 算子-芯片亲和系数 $\alpha(\tau_i, u)$ + FPGA 比特流重构 / NPU 上下文加载 $T_{setup}$ + 三级通信（同芯/片内 PCIe/跨节点多跳） | [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py), [`src/continuous.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/continuous.py) |
| **算法层** | [`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py) 为单网络图 2 层 GAT（节点 10 维、边 4 维）；[`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py) 仅对预筛候选列表打分 | 双图 GAT（网络拓扑 GAT 扩展节点特征 + 新增轻量任务 DAG-GAT）+ 联合动作 `(m, u, p)` + Actor 输出层 Invalid Action Masking（$-\infty$ 屏蔽） | [`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py), [`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py), [`src/algorithms/ppo.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/algorithms/ppo.py), [`src/baselines.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/baselines.py) |
| **目标与奖励层** | `reward_version="v8_dynamic_bottleneck"` 按单任务完成率/时延/能耗与单维边缘方差计算 | `reward_version="v9_hetero_dag"`：DAG 整体 Makespan 与违约率 + 分芯片计算/重构/三级传输能耗 + CPU/NPU/FPGA 分维度负载标准差加权和 | [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py), [`src/evaluation.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/evaluation.py), [`src/train.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/train.py) |
| **平台与文档层** | `backend/` 5 张 MySQL 表 + FastAPI 路由 + `frontend/` Vue 3 界面 + `docs/` 7 份说明书 | 保持 5 张表向后兼容，扩展 DAG 子任务轨迹与三类芯片负载字段；更新数学模型与实验脚本 | [`backend/schema.sql`](https://github.com/yuhanghu2001/schedule_test/blob/main/backend/schema.sql), `backend/services/`, `frontend/src/`, `scripts/`, `docs/` |

**推荐开发顺序**：`Phase 0（配置基线）` $\rightarrow$ `Phase 1（DAG 任务层）` $\rightarrow$ `Phase 2（细粒度资源与梯度）` $\rightarrow$ `Phase 3（亲和度、重构与三级通信）` $\rightarrow$ `Phase 4（双图 GAT-PPO 与掩码）` $\rightarrow$ `Phase 5（奖励、指标与基线评估）` $\rightarrow$ `Phase 6（前后端兼容与文档同步）`。

---

## 二、分阶段详细开发任务清单

### Phase 0：配置扩展与版本兼容基线

**目标**：在不破坏原有 `"default"` / `"high_congestion"` / `"small_smoke"` 预设的前提下，在 [`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py) 中建立异构 DAG 场景的配置开关与核心物理参数。

- [ ] **P0.1 新增架构与版本开关字段**（[`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py)）
  - 在 `Config` dataclass 中新增：`enable_dag_mode: bool = True`、`observation_version: str = "v5_dual_graph_dag"`、`reward_version: str = "v9_hetero_dag"`。
  - 保留旧版 `observation_version="v4_edge_only_4d_edges"` 与 `reward_version="v8_dynamic_bottleneck"` 的读取兼容，方便后续跑消融实验与回归对比。
- [ ] **P0.2 新增异构芯片三维资源与配比梯度参数**（[`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py)）
  - 定义节点分层角色映射（对应现有 4×3 八邻域拓扑）：
    - 底层接入边缘（E10、E9、E6、E5，承接 24 个终端接入）：高 CPU 核数 + 高 FPGA（LUT/FF/BRAM/DSP）配置、低 NPU 配比。
    - 中层区域边缘（中间排节点）：高 NPU 算力与显存配置，专攻高并发推理。
    - 上层网关（E0–E3）与云端：高并发大算力 NPU + 多核 CPU，用于重型分支与过载兜底。
  - 新增默认资源上限配置字典/元组：CPU（核心数、内存 GB）、NPU（等效算力 GFLOPs、显存 GB）、FPGA（LUT、FF、BRAM、DSP 及重构槽位）。
- [ ] **P0.3 新增算子亲和度、切换开销与三级通信参数**（[`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py)）
  - 定义算子类型枚举（如 `CONTROL_PARSE`、`STREAM_FILTER_ROI`、`FFT_FIR`、`TENSOR_INFER`、`AGGREGATE_TRACK`）及其对 `{CPU, NPU, FPGA}` 的亲和度矩阵 `operator_affinity_matrix`。
  - 新增切换开销参数：`fpga_reconfig_delay_s`、`fpga_reconfig_energy_j`、`npu_context_load_delay_s`、`npu_context_load_energy_j`。
  - 新增片内跨芯片通信参数：`pcie_dma_bandwidth_mbps`、`pcie_dma_base_latency_s`、`pcie_dma_energy_per_mb_j`。
  - 新增分芯片功耗系数（区分 CPU/NPU/FPGA 单位计算能耗）与三类芯片负载均衡权重 `(w_cpu, w_npu, w_fpga)`。
- [ ] **P0.4 新增异构 DAG 场景预设**（[`src/config.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/config.py)）
  - 在 `load_config(name)` 中新增 `"hetero_dag_default"`、`"hetero_dag_high_congestion"`、`"hetero_dag_smoke"` 三个预设，确保现有脚本通过 `--preset` 可一键切换。

---

### Phase 1：任务层升级——5 类异构 DAG 模板与图属性计算

**目标**：改造 [`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py)，将 5 类单点独立任务升级为 4~6 个子任务的异构 DAG，并计算调度所需的结构先验（`rank_u`、`Slack`）。

- [ ] **P1.1 设计 `SubTask` 与 `TaskDAG` 数据结构**（[`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py)）
  - 定义 `@dataclass class SubTask`：包含 `dag_id`、`subtask_id`、`op_type`（算子类型 $\tau_i$）、`compatible_chips`（兼容芯片集 $\mathcal{U}(t_i) \subseteq \{0:\text{CPU}, 1:\text{NPU}, 2:\text{FPGA}\}$）、`compute_gflop`（$W_i$）、细粒度资源需求（`cpu_cores`、`ram_gb`、`npu_vram_gb`、`fpga_lut`、`fpga_dsp`、`fpga_bram`）、`preds`、`succs`、`rank_u`、`slack_s`。
  - 定义 `@dataclass class TaskDAG`：保留原 `Task` 的顶层业务字段（`dag_id`、`source`、`access_edge`、`kind` 0~4、`priority`、`deadline_s`、`slot`），新增 `subtasks: list[SubTask]`、`edge_data_mb: dict[tuple[int, int], float]`（刻画前驱到后继的数据压缩/放大）、`entry_ids`、`exit_ids`、`input_data_mb`。
- [ ] **P1.2 实现 5 类业务异构 DAG 模板生成器**（[`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py)）
  - **视频分析 / 目标检测 DAG（5~6 节点）**：`关键帧抽取 (CPU)` $\rightarrow$ `2~3 路并行去噪/滤波/ROI 生成 (FPGA 优先，CPU 兜底，输出数据量显著压缩)` $\rightarrow$ `神经网络推理 (NPU 优先，FPGA 次选，CPU 兜底)` $\rightarrow$ `目标跟踪与结果汇聚 (CPU)`。
  - **信号计算 / 语音处理 DAG（4~5 节点）**：`协议解析与分帧 (CPU)` $\rightarrow$ `并行 FFT/FIR 实时流式滤波 (FPGA 最优，CPU 兜底)` $\rightarrow$ `特征识别模型推理 (NPU 最优，CPU 兜底)` $\rightarrow$ `状态聚合决策 (CPU)`。
  - **图像识别 DAG（4 节点）**：`图像解码与分片 (CPU)` $\rightarrow$ `2 路并行锐化/增强 (FPGA/CPU)` $\rightarrow$ `分类推理与结果合并 (NPU/CPU)`。
  - 为每条依赖边配置合理的数据量缩放比例（例如原始流 4~8 MB $\rightarrow$ 滤波/ROI 后压缩至 0.5~1.5 MB $\rightarrow$ 推理特征结果压缩至 0.05~0.2 MB），凸显“就近预处理降维再多跳传输”的收益。
- [ ] **P1.3 实现 DAG 拓扑先验特征计算**（[`src/tasks.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/tasks.py)）
  - 实现 `compute_dag_priorities(dag, avg_chip_speeds, avg_link_rate)`：
    - 反向拓扑遍历计算向上关键度 `rank_u`（HEFT 风格：节点平均执行时间 + 最大后继通信与 `rank_u` 之和）。
    - 计算各分支的最早开始时间（EST）、最晚开始时间（LST）及松弛时间 `slack_s = LST - EST`，供后续 DAG-GAT 节点特征与奖励塑形使用。
  - 升级 `sample_tasks(...)`：支持按 `cfg.enable_dag_mode` 返回 `list[TaskDAG]`，并基于 DAG 关键路径最小理论完成时间 $\times$ `[deadline_slack_min, deadline_slack_max]` 计算合理的 `deadline_s`。
- [ ] **P1.4 编写 Phase 1 单元测试**（`tests/`）
  - 验证 5 类 DAG 均为无环连通图（拓扑排序成功）、兼容芯片集非空、`rank_u` 沿依赖边单调递减、`deadline_s` 大于理想关键路径耗时。

---

### Phase 2：资源与拓扑层升级——三维细粒度资源池与动态预留

**目标**：在保留 [`src/topology.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/topology.py) 中 12 边 + 1 云 + 29 条动态链路的前提下，改造 [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py) 中的节点资源建模与在途资源管理。

- [ ] **P2.1 封装三维细粒度节点资源状态类**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py) 或新建 `src/resources.py`）
  - 为 13 个计算节点（12 边缘 + 1 云）分别维护三类芯片的容量上限与当前可用余量：
    - **CPU**：`cpu_cores_cap / free`、`ram_gb_cap / free`、`cpu_speed_gflops`、`cpu_queue_len`
    - **NPU**：`npu_speed_gflops`、`npu_vram_cap / free`、`npu_resident_model`（可选）、`npu_queue_len`
    - **FPGA**：`fpga_speed_gflops`、`fpga_lut_cap / free`、`fpga_ff_cap / free`、`fpga_bram_cap / free`、`fpga_dsp_cap / free`、`fpga_resident_op_id`（当前驻留算子 ID）、`fpga_queue_len`
  - 校验节点异构配比梯度初始化逻辑：确保底层接入边缘（E10/E9/E6/E5）、中层区域边缘、上层网关（E0–E3）与云端各自拥有差异化硬件画像，避免 12 个边缘同质化。
- [ ] **P2.2 实现时间片/事件驱动的在途资源预留与归还机制**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)）
  - 引入在途占用记录表（如 `ActiveReservation(node, chip, release_time, cpu_cores, ram_gb, npu_vram_gb, fpga_lut, fpga_ff, fpga_bram, fpga_dsp, op_type)`）。
  - 在子任务分配时立即扣减对应芯片的细粒度余量并写入预留记录；在推进仿真时钟或进入下一时隙时，自动扫描 `release_time <= current_time` 的记录并归还对应芯片资源。
  - 实现细粒度资源可行性校验接口 `can_allocate(node, chip, subtask) -> bool`，直接供环境转移与无效动作掩码复用。
- [ ] **P2.3 保持底层网络拓扑模块稳定**（[`src/topology.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/topology.py)）
  - 保持 `edge_pairs`、`make_topology`、`fluctuate_topology`、`simple_routes` 的核心逻辑与云网关限制（仅 E0–E3 直连云端、云不中继）不变。
  - 仅补充一个辅助函数（如 `node_tier_roles(n_edges)`），返回每个边缘节点的拓扑层级标签（接入层/中层/网关层），供配置初始化与前端可视化复用。

---

### Phase 3：映射、代价与三级通信层升级

**目标**：改造 [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py) 与 [`src/continuous.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/continuous.py)，建立算子亲和执行时间、异构切换开销与三级通信统一计算流水线。

- [ ] **P3.1 实现算子亲和度与多态执行时间/能耗计算**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)）
  - 按公式实现子任务执行时间计算：
    $$T_{exec}(t_i, m, u) = \frac{W_i}{C_{m,u} \cdot \alpha(\tau_i, u)} + T_{setup}(t_i, m, u)$$
    - 当 $u = \text{FPGA}$ 且节点 $m$ 当前 `fpga_resident_op_id == tau_i` 时，$T_{setup} = 0$，否则叠加 `fpga_reconfig_delay_s` 并计入 `fpga_reconfig_energy_j`，调度生效后更新 `fpga_resident_op_id = tau_i`。
    - 当 $u = \text{NPU}$ 时，按需叠加模型上下文加载时延 `npu_context_load_delay_s` 与加载能耗。
  - 适配 [`src/continuous.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/continuous.py) 中的 `allocate_compute`（KKT 注水法）：当同一时隙内多个子任务落到同一 `(node m, chip u)` 时，结合有效工作量 $W_i / \alpha(\tau_i, u)$ 进行并发算力切分与排队完成时间更新。
- [ ] **P3.2 实现三级通信时延与能耗模型**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)）
  - 封装 `estimate_subtask_comm(subtask, target_node, target_chip, path, dag_state)`，处理 DAG 入口子任务与中间依赖边两类通信：
    - **入口子任务（端侧源 $\rightarrow$ 接入边缘 $\rightarrow$ 目标节点）**：复用现有 [`src/continuous.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/continuous.py) 的 Dinkelbach 上行功率优化（`choose_uplink_power`）+ 现有多跳路径传输模型。
    - **DAG 内部前驱 $t_p \rightarrow$ 当前子任务 $t_i$ 的三级通信**：
      1. **同节点同芯片**（$m_p = m_i, u_p = u_i$）：通信时延 $= 0$，通信能耗 $= 0$。
      2. **同节点跨芯片**（$m_p = m_i, u_p \neq u_i$）：按 `edge_data_mb[(p, i)] / pcie_dma_bandwidth_mbps + pcie_dma_base_latency_s` 计算片内 PCIe/DMA 搬运时延，并计入拷贝能耗。
      3. **跨节点**（$m_p \neq m_i$）：直接调用现有 1~4 跳动态边缘链路模型（含时隙通断、AR(1) 拥塞、带宽抖动与中继转发能耗），取所有前驱到达时间的最大值计算当前子任务最早开始时间：
         $$\text{EST}(t_i) = \max_{p \in \text{pred}(t_i)} \left( \text{FT}(t_p) + T_{comm}(p \rightarrow i) \right)$$
- [ ] **P3.3 改造 `EdgeEnv` 的 DAG 步进调度循环**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)）
  - 将原“每步调度 1 个独立 `Task`”改造为“维护当前时隙就绪子任务队列（Ready Queue，按 `rank_u` 降序或关键度排序），每步弹出 1 个就绪 `SubTask` 执行 `(node m, chip u, path p)` 决策”。
  - 扩展 `Action` 与 `Assignment` dataclass：补充 `dag_id`、`subtask_id`、`comm_tier`（0/1/2）、`setup_delay_s`、`setup_energy_j`、`start_time_s`、`finish_time_s`。
  - 当 DAG 所有退出节点（`exit_ids`）完成时结算整图 `makespan_s = max(FT) - arrival_time` 与是否按期完成；若任一关键子任务被拒或不可行，按策略标记整图违约并释放其下游未执行节点。
- [ ] **P3.4 编写 Phase 2–3 环境仿真冒烟测试**（`tests/`）
  - 校验同节点同芯通信为 0、同节点跨芯产生 PCIe 时延、跨节点复用多跳拥塞；校验 FPGA 连续执行同类算子时 `setup_delay_s == 0`、切换算子时触发重构时延。

---

### Phase 4：算法层升级——双图 GAT-PPO 与联合无效动作掩码

**目标**：升级 [`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py)、[`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py) 与 [`src/algorithms/ppo.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/algorithms/ppo.py)，构建极具辨识度的双图 GAT-PPO，并引入严格 Invalid Action Masking。

- [ ] **P4.1 升级网络拓扑 GAT 输入特征**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py), [`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py)）
  - 保留 [`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py) 中 4 维边特征 `GraphAttentionLayer`、`GATEncoder` 与 `gat_adj`（云端不中继）。
  - 将 `observe()` 中 `nodes` 特征从 `(N, 10)` 扩展为细粒度三维异构向量，包含：`[CPU 核/内存余量率与队列, NPU 算力/显存余量率与队列, FPGA LUT/DSP/BRAM 余量率与队列, 当前 FPGA 驻留算子 ID (One-Hot 或可学习 Embedding), 邻居平均拥塞, 云节点标志]`。
- [ ] **P4.2 新增轻量级任务 DAG-GAT 编码器**（[`src/models/gat.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/gat.py) 或新建 `src/models/dag_gat.py`）
  - 在 `observe()` 中构造当前 DAG 图张量：
    - `dag_nodes` `(M, F_dag)`：`[算子类型 One-Hot, 归一化计算量, 三类芯片亲和度(3维), 归一化 rank_u, 归一化 Slack, 子任务状态(待就绪/当前就绪/已完成), 细粒度资源需求]`。
    - `dag_edges` `(M, M, 1)`：归一化依赖传输数据量（前驱 $\rightarrow$ 后继有向边）。
    - `dag_adj` `(M, M)`：有向邻接矩阵（可同时保留正向/反向通道或双向加方向标志，帮助捕捉上游已分配信息与下游关键路径压力）。
    - `current_subtask_idx`：当前待调度子任务在 DAG 中的索引。
  - 实现轻量 1~2 层 `DAGGraphEncoder`（带边特征注意力 + 残差 + `LayerNorm`），输出当前子任务嵌入 `subtask_emb` 与整图汇聚嵌入 `dag_mean_emb`。
- [ ] **P4.3 实现联合动作空间与 Invalid Action Masking**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py), [`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py)）
  - 将动作空间组织为 `(target_node m, target_chip u, multi_hop_path p)`（外加 1 个兜底拒接/延迟动作防止极端死锁），由环境同步返回布尔掩码张量 `action_mask`。
  - 在 `EdgeEnv.build_action_mask(...)` 中严格屏蔽三类无效动作（对应位置置 `False`）：
    1. 前驱未完成或芯片架构不兼容（$u \notin \mathcal{U}(t_i)$）；
    2. 目标节点 $m$ 上芯片 $u$ 的细粒度资源（内存/显存/LUT/FF/BRAM/DSP）或队列余量不足；
    3. 当前时隙因动态断链不可达的路径 $p$（以及违反 E0–E3 云网关约束的路径）。
  - 在 `GATPPO`（[`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py)）的 Actor 输入中融合 `[source_emb, target_emb, path_emb, subtask_emb, dag_mean_emb, cand_feat, global_feat]`，在输出 logits 上执行 `logits = logits.masked_fill(~action_mask, -1e9)`（等效 $-\infty$）后再构建 `Categorical(logits=logits)`，保证采样概率严格为 0 且不产生无效探索梯度。
  - 同步升级 Critic 输入为 `[net_graph_mean, dag_mean_emb, subtask_emb, global_feat]`。
- [ ] **P4.4 升级 `PPOTrainer` 轨迹缓存与掩码更新**（[`src/algorithms/ppo.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/algorithms/ppo.py)）
  - 在 `collect_episode(env)` 中保存 `dag_nodes`、`dag_edges`、`dag_adj`、`current_subtask_idx` 与 `action_mask`。
  - 在 `update(episode)` 计算新策略 `log_prob` 与 `entropy` 时，重新传入对应的 `action_mask`，并检查无有效候选时的数值稳定性（防 `NaN` 梯度）。
- [ ] **P4.5 升级消融模型与启发式基线**（[`src/models/policy.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/models/policy.py), [`src/baselines.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/baselines.py)）
  - 在 `MODEL_NAMES` 中保留/新增消融变体：`"dual_gatppo"`（完整双图 GAT-PPO + 掩码）、`"single_gatppo"`（仅网络图 GAT，无 DAG-GAT）、`"ppo"`（扁平 MLP + 掩码）、`"shuffled_gat"`。
  - 在 [`src/baselines.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/baselines.py) 中升级 `shortest_fastest_action`、`greedy_estimate_action` 支持 `action_mask` 与三级通信估计，并新增异构 HEFT 启发式基线（按 `rank_u` + 最小 EFT 选 `(m, u, p)`），作为强对比基线。

---

### Phase 5：优化目标、奖励函数与评估体系升级

**目标**：升级 [`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)、[`src/evaluation.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/evaluation.py)、[`src/train.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/train.py) 及 `scripts/`，将 4 大指标全面升级为异构 DAG 感知口径。

- [ ] **P5.1 升级多目标奖励函数（`reward_version="v9_hetero_dag"`）**（[`src/environment.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/environment.py)）
  - **完成率与时延项**：
    - 子任务级过程塑形（Dense Shaping）：对当前子任务的增量耗时与松弛度消耗（关键路径 `slack_s == 0` 给予更高权重）施加负奖励，引导 PPO 把稀缺 NPU/FPGA 留给关键路径、把高松弛度分支弹性回落给空闲 CPU。
    - DAG 终态结算：当 DAG 最后一个子任务完成时，依据整图 `makespan_s <= deadline_s` 发放完成率奖励或截止期违约惩罚。
  - **异构能耗项**：汇总当前子任务在所选芯片上的动态计算能耗（区分 CPU/NPU/FPGA 功耗系数）+ FPGA 比特流重构能耗 / NPU 加载能耗 + 三级通信能耗（片内 PCIe 或多跳转发）。
  - **分芯片三维负载均衡项**：将原单维资源负载方差改为 CPU / NPU / FPGA 三类芯片分别计算全网节点负载标准差再加权求和：
    $$D_{load} = w_{\text{CPU}} \cdot \sigma(\text{Load}_{\text{CPU}}) + w_{\text{NPU}} \cdot \sigma(\text{Load}_{\text{NPU}}) + w_{\text{FPGA}} \cdot \sigma(\text{Load}_{\text{FPGA}})$$
    避免“NPU 爆满、CPU 空转”带来的虚假均衡。
- [ ] **P5.2 升级评估指标与日志归档**（[`src/evaluation.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/evaluation.py), [`src/train.py`](https://github.com/yuhanghu2001/schedule_test/blob/main/src/train.py)）
  - 在 `EVALUATION_METRICS` 中保持原有核心字段名兼容（或平滑映射），并新增异构 DAG 专属指标：
    - `dag_completion_rate`、`avg_dag_makespan_s`、`deadline_violation_rate`
    - `avg_dag_energy_j`、`fpga_reconfig_rate`、`fpga_reconfig_delay_ratio`
    - `cpu_load_std`、`npu_load_std`、`fpga_load_std`、`hetero_load_balance_degree`
    - `comm_tier0_ratio`（同芯）、`comm_tier1_ratio`（片内跨芯）、`comm_tier2_ratio`（跨节点多跳）
  - 更新 `src/plotting.py`，在 `training_metrics.png` 与 `evaluation_metrics.png` 中新增三类芯片负载均衡对比曲线与 FPGA 重构率曲线。
- [ ] **P5.3 更新 CLI 脚本与回归验证流程**（`scripts/`）
  - 检查并适配 `scripts/smoke.py`、`scripts/inspect_env.py`、`scripts/evaluate.py`、`scripts/evaluate_heuristic.py`、`scripts/experiment.py`，确保 `--preset hetero_dag_smoke` 能在数分钟内跑通完整训练 + 评估 + 绘图闭环。

---

### Phase 6：上位机平台兼容与文档同步

**目标**：确保 Vue 3 + FastAPI + MySQL 上位机平台平滑承接异构 DAG 调度展示，并同步更新仓库文档。

- [ ] **P6.1 后端 Schema 与数据库向后兼容**（[`backend/schema.sql`](https://github.com/yuhanghu2001/schedule_test/blob/main/backend/schema.sql), `backend/schemas.py`）
  - 保留 `env_configs`、`simulation_snapshots`、`training_runs`、`evaluation_records`、`training_metrics` 5 张主表结构，利用 JSON 字段（配置快照与指标快照）承载新增的 DAG 模板参数与三维芯片负载指标，避免破坏历史实验记录查询。
  - 在 `backend/schemas.py` 中补充 `SubTaskAssignmentSchema`、`ChipResourceUsageSchema` 与新增评估字段的可选 Pydantic 模型定义。
- [ ] **P6.2 后端仿真与训练服务适配**（`backend/services/sim_service.py`, `backend/services/train_service.py`, `backend/routers/`）
  - 在仿真步进接口返回的快照中附带：当前节点 CPU/NPU/FPGA 三维余量、FPGA 当前驻留算子 ID、以及当前 DAG 的子任务映射与三级通信路径。
  - 确保训练任务管理器支持选择 `"dual_gatppo"` 与 `"hetero_dag_*"` 预设。
- [ ] **P6.3 前端可视化增量适配**（`frontend/src/`）
  - 节点状态卡片/悬浮提示中，将单一算力条拆分为 **CPU / NPU / FPGA 三条细粒度进度条 + FPGA 驻留算子标签**。
  - 任务调度详情区保留原有 5 类业务名称筛选，点击单条任务可展开其 4~6 个子任务的 DAG 拓扑/甘特流向（标注所在节点、芯片、三级通信类型与重构耗时）。
  - 评估对比图表区新增分芯片负载标准差与 DAG Makespan 对比维度。
- [ ] **P6.4 同步技术文档与论文数学模型**（`docs/`, [README.md](https://www.google.com/search?q=https://github.com/yuhanghu2001/schedule_test/blob/main/README.md)）
  - 更新 `docs/01_环境开发说明书.md`、`docs/02_数学模型与C1-C24约束.md`（补充 DAG 前驱时序约束、三维细粒度资源约束、亲和度与重构时延公式、分芯片负载均衡定义）及双图 GAT-PPO 架构说明文档，保证代码与开题/学位论文公式严格一致。

---

## 三、关键里程碑验收标准与常见避坑提示

| 阶段 | 验收命令 / 检查点 | 常见避坑点 |
| :--- | :--- | :--- |
| **Phase 1–3 完成** | 运行 `pytest tests/` 与 `python scripts/inspect_env.py --preset hetero_dag_smoke` | **掩码全 False 死锁**：极端拥塞或全网资源耗尽时，必须保留 1 个始终合法的兜底动作（如 `REJECT` 或云端兜底），否则 `Categorical(logits=-inf)` 会触发 `NaN`。 |
| **Phase 4 完成** | 运行 `python scripts/smoke.py --preset hetero_dag_smoke --model dual_gatppo` | **Batch 内变长 DAG 对齐**：5 类 DAG 节点数在 4~6 之间变化，PPO `update()` 打包 mini-batch 时需对 `dag_nodes`/`dag_adj` 按 `max_dag_nodes=6` 做零填充并配合 `dag_node_mask` 池化，避免维度报错。 |
| **Phase 5 完成** | 运行 `scripts/experiment.py` 对比 `dual_gatppo`、`single_gatppo`、`ppo`、`heft_hetero`、`greedy` | **稀疏奖励收敛慢**：不要只在 DAG 最后一个子任务给奖励；结合关键路径 `rank_u` / `slack_s` 做步级密集塑形，同时将过程奖励与终态完成奖励量级对齐，避免 Critic 估值震荡。 |
| **Phase 6 完成** | 启动 `scripts/dev.py`，验证 `/api/health`、`8000` 端口接口与 `8888` 端口前端拓扑渲染 | **历史 Checkpoint 兼容**：在 `save_checkpoint()` 中写入 `observation_version` 与 `model_name` 元数据，前端/评估脚本加载旧版 `v4` 权重时自动路由到原单图结构或给出清晰版本提示。 |
