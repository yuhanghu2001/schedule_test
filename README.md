# schedule_test — 动态多跳云边端异构任务调度

总结：用 GAT（图注意力网络）+ PPO 强化学习，在"云—边—端"三层异构计算环境中，动态决定每个任务在哪个节点执行、用哪种芯片资源、走哪条多跳路径，从而联合优化任务完成率、平均完成时间、平均任务能耗和资源负载均衡度。

解决的问题：场景默认是 24 个端侧任务源（传感器/相机/车等，仅产生任务并通过无线接入边缘，不执行本地计算）持续产生 5 类智能任务（图像识别、目标检测、视频分析、语音处理、信号计算），卸载到 12 个区域边缘服务器或 1 个云端执行。难点在于目标异构节点往往需经过 1–4 跳动态波动的边缘链路（时隙级通断、AR(1) 拥塞漂移与尖峰、带宽抖动）到达，因此"选执行节点"和"选多跳传输路径"强耦合。4 维边特征 GAT 负责编码动态拓扑与拥塞趋势，PPO 负责在可行动作里做联合打分。当前三套预设仅前 4 个边缘（默认 E0–E3）具有云端广域连接，其余边缘需先多跳到达网关再上云；云端不作为边缘中继。边缘网络仅连接上下左右和四个对角的相邻节点，默认四列三行共有 29 条边缘链路。端侧按示意位置唯一接入最近边缘，当前底部 24 个终端分别接入 E10/E9/E6/E5，不向上层节点重复拉无线连线。

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

## 前端、后端和 MySQL 一键启动

在仓库根目录操作。首次安装 JavaScript 依赖：

```bash
npm install
```

后端使用 Conda 的 `gatppo` 环境；首次使用时安装 Python 依赖（已有环境无需重新创建）：

```bash
conda create -n gatppo python=3.11
conda run -n gatppo python -m pip install -r requirements.txt -r backend/requirements.txt
```

之后每次只需运行：

```bash
npm start
```

`npm run dev` 与 `npm start` 等效，在 `frontend/` 目录运行也可以。启动命令依次准备项目专用 MySQL、启动 FastAPI 和 Vite，无需手动激活 Conda 环境或执行 sudo。按 `Ctrl+C` 会先停止应用服务，再关闭 MySQL，数据库文件保留；FastAPI 或 Vite 退出时也会清理本次启动的其它服务。

- 浏览器打开 **http://localhost:8888** 进入深蓝三栏系统总览；远程访问用 `http://服务器IP:8888`。
- 后端接口文档：http://localhost:8000/docs。
- 前端的 `/api` 和 `/ws` 请求通过 Vite 转发至本机 `8000` 端口。
- 前端固定使用 `8888`，端口占用时会报错，避免自动切换后访问错地址。
- `npm run build` 构建前端静态产物，不启动服务。

**MySQL 自动准备**：首次 `npm start` 通过 conda-forge 安装固定版本 MySQL 8.4.2 到 `.local/mysql-env/`（需要联网，不修改 `gatppo` 依赖），初始化 `.local/mysql-data/`，生成随机数据库密码及 `backend/.env`，自动创建 `schedule_test`、应用账户和 `backend/schema.sql` 的五张表。当前机器已完成安装和初始化，后续启动复用已有文件。

- MySQL 仅监听 `127.0.0.1:3306`；无需单独安装系统服务。
- 应用连接参数：`backend/.env`；本地管理员连接文件：`.local/mysql-admin.cnf`。两者权限为 `600`，不会输出密码。
- 数据文件：`.local/mysql-data/`；数据库日志：`.local/mysql.log`。数据库目录、独立 Conda 环境和凭据均已被 Git 忽略。
- 停止和重启保留库表及记录；备份时保留 `.local/` 和 `backend/.env`，不要单独删除连接凭据。
- 访问 http://localhost:8888/api/health，`db_connected: true` 表示数据库连接成功。
- 已有进程占用端口时，在原启动终端按 `Ctrl+C` 后再运行，启动脚本不会杀掉其它实例。

总览右侧“任务产生”集中显示整轮任务数、完成数、已调度业务分布及当前待调度任务的来源和工作量，左侧概况只保留四项调度 KPI。总览先按场景预览任务和拓扑，手动选择调度算法，选择 GAT-PPO 或 PPO 时再选模型权重批次，点击“任务加载”，再点击“开始推演”逐任务自动运行至本轮结束。“暂停推演”会切换为“继续推演”，继续时沿用当前任务进度。更换算法后需要重新加载任务。总览支持节点详情和任务路径高亮。主导航为“系统总览、模型训练、调度日志、拓扑监控、节点管理”，训练工作台和实验记录导航入口已移除，原页面路由仍保留；完整沙盘 `/sandbox` 与算法评估 `/compare` 保留原路由。独立拓扑监控页 `/topology` 采用参考图的点阵地图、透视网格、发光平台及云端/边缘服务区域布局，从环境预览快照展示全部节点与真实链路。悬停链路可查看速率、拥塞及双向 1 MB 单跳估算时延。云和边缘节点标出 CPU、NPU、FPGA 算力并高亮优势资源；云下方标注 C0，边缘节点上下左右以水平/竖直线连接，四个对角以斜线连接，不绘制跨格或折返连线。系统总览的拓扑面板复用同一完整场景，按面板等比缩小，保留背景、全部节点及真实连线；选中任务表记录可高亮其实际路径，推演操作继续在系统总览中使用。节点管理可编辑端侧无线速率、边缘/云端算力、内存、存储、队列容量和服务部署，保存到本机环境草稿后用于新建推演及训练。组件、指标口径及实际截图见 [上位机系统说明](docs/07_上位机可视化调度系统设计.md#5-深蓝监控总览2026-11-19)。

左侧“网络资源状态与调度瓶颈”根据总览快照展示边缘节点综合算力利用率、网络链路状态及利用率趋势，并提醒节点高负载、链路拥塞和断链。点击节点可查看详细使用情况。指标口径见[网络资源状态与调度瓶颈](docs/07_上位机可视化调度系统设计.md#10-网络资源状态与调度瓶颈)。

主导航“模型训练”进入 `/logs`。可按北京时间实验批次选择同组模型，在训练回合与固定种子验证之间切换，按回合范围查询结果对比、指标趋势、日志明细与累计奖励对比。页面读取 `results/` 中已有日志。统计口径见[模型训练日志](docs/07_上位机可视化调度系统设计.md#11-模型训练日志)。

总览右侧“任务场景”和拓扑操作栏“加载场景”共享同一个场景。切换场景会重新预览该场景的任务与拓扑，旧推演须重新加载。任务产生区展示基础到达率、基准拥塞和速率，悬停拓扑中的单条链路可查看双向 1 MB 单跳估算时延。

总览完成推演后弹出提示，点击“查看调度日志”可进入 `/dispatch-logs`。同一环境和种子下，各算法最近一次完成的结果可直接比较。页面展示各项最佳值、结果表、归一化雷达图、指标对比和按已调度任务数绘制的趋势。统计口径见[调度日志](docs/07_上位机可视化调度系统设计.md#12-调度日志)。

## 训练设备与线程

当前小图逐任务训练在 CPU 上显著快于 GPU，**默认使用 `gatppo` 环境与 `--device cpu --threads 2`**（`conda run -n gatppo python src/train.py`）。如需在 RTX 50 系显卡上做 GPU 对照，可使用独立的 `gatppo-gpu` 环境（`conda run -n gatppo-gpu python src/train.py --device cuda --threads 2`）。`--device auto` 会在 CUDA 可用时选择 GPU，训练配置快照会记录实际设备和 CPU 线程数。

训练过程可用 `Ctrl+C` 手动中断。训练入口会先关闭并刷新 JSONL，再根据已完成回合重建 `training_metrics.png` 和已有的 `evaluation_metrics.png`，同时保存 last checkpoint。

环境中的候选动作枚举、多跳路径和物理约束仍是顺序相关的 Python/NumPy 逻辑，保留在 CPU；当前默认模型为两层256维GAT、128维Actor/Critic头，每轮含20个时隙并更新5个PPO epoch，因图规模小且逐样本更新，CPU（2线程）单轮约 2s，快于 GPU 的频繁拷贝与核函数启动开销。

## GAT-PPO / PPO 统一训练入口

在仓库根目录运行，`--episodes` 是每个算法的训练回合数：

```bash
# 完整 GAT-PPO
conda run --no-capture-output -n gatppo python src/train.py --model gatppo --episodes 1000

# 单独 PPO（MLP 节点编码，不执行 GAT 消息传播）
conda run --no-capture-output -n gatppo python src/train.py --model ppo --episodes 1000

# 一次运行两者：相同配置/种子/回合预算，依次独立训练并生成对比图
conda run --no-capture-output -n gatppo python src/train.py --model gatppo and ppo --episodes 1000
```

`--model "gatppo and ppo"` 也可用；`gatpppo` 作为完整算法的拼写别名。历史 `gat`、`flat` 名称及对应权重仍可读取。`ablation.py` 已删除，原图观测打乱包装移入 `environment.py`，已有 `shuffled_gat` 实验保持可运行。

单算法使用 `results/YYYYMMDD_HHMMSS/`，保存 `best_gatppo.pt` / `last_gatppo.pt` 或 `best_ppo.pt` / `last_ppo.pt`、配置、JSONL 日志及原有曲线。双算法输出：

```text
results/YYYYMMDD_HHMMSS/
├── comparison_config.json
├── gatppo/                         # GAT-PPO 的权重、配置、日志和曲线
├── ppo/                            # PPO 的权重、配置、日志和曲线
├── training_comparison.png         # 两算法训练五指标叠加图
└── evaluation_comparison.png       # 两算法固定验证五指标叠加图
```

训练图含累计奖励、完成率、时延、能耗和负载均衡度；验证图含目标分数、完成率、时延、能耗和负载方差，阴影是同一模型的固定验证种子间标准差。它不表示多次训练种子间置信区间，也不替代独立最终测试。

可用 `--config default/high_congestion/small_smoke`（等价参数名 `--preset`）、`--seed`、`--device`、`--threads` 设置参数。单算法续训使用 `--resume 路径 --episodes 总回合数`；`--output` 和 `--resume` 只用于单算法。Ctrl+C 保留已完成回合与 last 权重，双算法模式仅对已有记录绘图，不伪造未运行算法的数据。开发验证必须指定系统临时目录 `--run-dir`，不写入正式 results。

### 异构 DAG 协同调度

异构 DAG 模式通过 `enable_dag_mode` 开启，使用 `hetero_dag_default`、`hetero_dag_high_congestion`、`hetero_dag_smoke` 三个预设，分别继承传统三个场景的负载与网络参数。每个任务从图像识别、目标检测、视频分析、语音处理、信号计算五类 DAG 模板生成；子任务具备 CPU/NPU/FPGA 兼容性和核数、RAM、显存及 FPGA LUT/FF/BRAM/DSP 需求，并受算子亲和度、芯片层级画像、FPGA 重构、NPU 上下文载入和三级内部通信（同节点同芯 / 同节点跨芯 PCIe/DMA / 跨节点多跳）影响。deadline 根据源端实际无线速率、DAG 最长依赖路径工作量和配置化松弛系数生成。

训练模型为 `dual_gatppo`（网络与任务 DAG 双图编码）、`single_gatppo`（仅网络图 GAT 编码，无 DAG-GAT）、`ppo` 和 `shuffled_gat`。策略使用无效动作掩码。推荐直接通过根目录极简入口 `python train.py --dual_gatppo` 启动（训练轮数、场景预设、设备等参数统一在 `train.py` 顶部配置）；底层模块入口也支持 `--preset hetero_dag_default --model dual_gatppo --episodes 500`。DAG 评估增加 `dag_completion_rate`、`avg_dag_makespan_s`、`deadline_violation_rate`、`avg_dag_energy_j`、FPGA 重构、CPU/NPU/FPGA 三维负载及通信层级指标；`heft_hetero` 是对应异构 HEFT 启发式基线。相关实现与指标定义见 [环境开发说明](docs/01_环境开发说明书.md#异构-dag-模式enable_dag_modetrue)、[实验设计](docs/03_实验设计与指标.md#异构-dag-升级) 和 [奖励说明](docs/05_训练奖励与ppo对齐说明.md#5-异构-dag-的-v9-奖励与观测)。目前不据此宣称任何未完成的训练效果。

## 文件概览

```text
schedule_test/
├── AGENTS.md                         # 全员共享硬规则：简单、无防御式编程与执行边界
├── GEMINI.md                         # Gemini 大脑入口（自动引入 Codex 调度规则）
├── CLAUDE.md                         # Claude 大脑入口（自动引入 AGENTS 与 Codex 调度规则）
├── README.md
├── requirements.txt
├── environment-gpu.yml               # RTX 50 系独立 GPU 环境
├── docs/
│   ├── 01_环境开发说明书.md             # 从建模到代码与交付验收
│   ├── 02_数学模型与C1-C24约束.md       # 24条约束的统一正确表达
│   ├── 03_实验设计与指标.md             # 场景、消融、泛化、实验协议
│   ├── 04_数据字典与实现边界.md         # 关键变量、原型简化项
│   ├── 05_训练奖励与PPO对齐说明.md      # 可加奖励、全局状态与固定评估
│   ├── 06_多智能体调度与Codex委派规则.md # 大脑调度 Codex 的模型、推理档位与审核闭环
│   └── 07_上位机可视化调度系统设计.md   # Vue3+Element Plus+FastAPI+MySQL 上位机架构
├── memory/                           # AI 对话与工作记录
├── backend/                          # FastAPI + MySQL 上位机后端（直接复用 src/）
│   ├── app.py                        # FastAPI 入口、健康检查与路由挂载
│   ├── db.py                         # SQLAlchemy + PyMySQL 5 张表 ORM
│   ├── schema.sql                    # MySQL 建表脚本
│   ├── services/                     # 仿真推演与后台训练/WebSocket 服务
│   └── routers/                      # 环境配置、训练监控与多策略评估接口
├── frontend/                         # Vue 3 + Vite + Element Plus + ECharts 前端
│   └── src/                          # 训练工作台、推演沙盘、对比实验室、实验档案与拓扑画布
├── src/
│   ├── train.py                      # GAT-PPO/PPO 单算法、双算法对比、续训与绘图入口
│   ├── config.py                     # 全部参数与传统/DAG 六套 Python 场景配置
│   ├── tasks.py                      # 传统任务与五类异构 DAG 模板生成
│   ├── resources.py                  # DAG 模式三维细粒度芯片资源池与预留
│   ├── topology.py                   # 动态连通图、层级角色与简单路由
│   ├── continuous.py                 # Dinkelbach 功率与 KKT 算力子问题
│   ├── baselines.py                  # 最短最快与估计一步奖励贪心基线
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

只有用户明确要求写入或更新 memory 时，Codex、Claude 等 AI 才在 `memory/` 留档或更新索引；完成任务、提交或推送代码不自动生成工作记录。获准留档时，目录名统一为 `YYYYMMDD_HHMMSS-对话内容标题/`。目录中的 `README.md` 应说明对话目标、完成事项、关键修改、验证证据、未解决事项和后续建议。例如：`memory/20261112_140000-检查代码完整性并训练GAT-PPO/`。

## 重要实现边界

- **已实现**：非平稳突发/热点任务流、12 边缘动态稀疏拓扑、时隙级非桥接链路通断、AR(1) 拥塞演化与尖峰、带宽抖动、4 维边特征 GAT、DFS 有效简单路径、正反向共享链路容量、跨时隙资源占用、拥塞关联执行不确定性、异构资源匹配、估计时限过滤、增量奖励、shuffled mini-batch PPO、完整状态续训、独立种子评估以及前后端分离可视化上位机调度系统。
- **任务时限**：每个终端先抽取实际无线速率，deadline 按任务输入上传、结果回传和最慢边缘算力（120 GFLOP/s）计算参考处理时间，再乘配置化松弛系数；不做时隙长度截断。该参考时间仍不包含排队和多跳传输，因此不等同于完整场景的可行性保证。
- **纯边缘–云卸载**：端侧仅产生任务与无线收发（`a_i=1, m_i^{off}=1`），所有接收任务由部署了对应服务的区域边缘或云端执行，或被合法拒绝。
- **连续子问题**：固定离散执行节点、异构资源和路径后，上行功率通过 Dinkelbach 法求解 Shannon 能效分式子问题；同节点同资源并发任务通过带截止时间最低算力约束的 KKT active-set 分配。两者只保证各自连续子问题最优，**不表示整个四目标混合离散–连续问题达到全局最优。**
- **模型差异**：数学文档给出完整 C1–C24 正式约束体系；代码采用“生成可行简单路径/动作”的实现方式隐式满足路由流守恒和无环条件，而非调用 MILP 求解器求完整原问题。
- **结果差异**：训练 reward 使用 `beta_1*delta-y*(beta_2*t+beta_3*E)-beta_4*D_load-Omega` 公式，四项系数比例为 `2:1:1:1`；任务项使用接收后的实际时延/能耗，负载项按时隙内边缘负载方差增量结算。未折扣时满足 `episode_reward/N=objective_score`，成功率与接收率分别报告。
- **实验差异**：仓库提供四方法、多种子和跨场景运行入口；只有真实完成相同预算实验后才能报告均值、方差或性能提升。

阅读次序：`AGENTS.md` → `docs/01_环境开发说明书.md` → `docs/02_数学模型与C1-C24约束.md`。
