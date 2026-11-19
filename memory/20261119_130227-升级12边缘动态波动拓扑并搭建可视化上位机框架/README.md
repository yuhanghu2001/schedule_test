# 20261119_130227 — 升级 12 边缘动态波动拓扑并搭建可视化上位机框架

## 1. 对话目标

1. 严格按照 `docs/06_多智能体调度与Codex委派规则.md`（Gemini 作为大脑设计、拆解、审核与验证，Codex 作为执行手按模型与推理档位落地代码），完成两项核心任务。
2. **任务一（发挥 GAT 作用的动态大拓扑与环境重构）**：将端侧改为纯任务产生源（无本地计算），上层边缘节点由 6 扩至 12（24 个端侧、18 条稀疏边、稀疏异构服务部署），并引入周期拓扑重构与时隙级链路动态通断（Link Flap）、AR(1) 拥塞漂移与尖峰、带宽抖动、拥塞关联执行扰动及 4 维 GAT 边特征。
3. **任务二（前后端分离可视化上位机调度系统框架）**：基于 Vue 3 + Vite + Element Plus + ECharts + `localStorage`（前端）与 FastAPI + SQLAlchemy/PyMySQL + MySQL（后端），搭建包含模型训练工作台、仿真推演沙盘、多算法对比实验室、实验记录与模型资产管理的完整上位机框架。

## 2. 已完成事项与大脑-Codex调度记录

1. **阶段 1A — 数学模型与文档先行（`gpt-6.2-astra`，`xhigh`）**：
   - 保持 C1–C18 与 C19–C24 约束编号和 `x/y/g/f/p_up/z/z_service` 变量语义不变，修订 `docs/02_数学模型与C1-C24约束.md` 并同步 `docs/01`、`03`、`04`、`05`。
   - 定义 v6 场景：`N=24` 纯任务源端侧、`M=12` 区域边缘（4 类异构画像与 12×5 稀疏服务矩阵，每类服务 4 副本，直接接入命中率 33.3%）、`C=1` 云端、最慢边缘算力 `120 GFLOP/s`、周期重构 + 非桥接链路保连通动态通断 + AR(1) 拥塞与尖峰 + 带宽抖动 + 4 维边特征 `[归一化当前带宽, 有效拥塞率, 归一化传播时延, Δρ]`。
2. **阶段 1B — 核心环境与 GAT 代码改造及 `D_ref` 实测重标定（`gpt-6.2-astra`，`high`）**：
   - 更新 `src/config.py`、`src/topology.py`、`src/tasks.py`、`src/environment.py`、`src/models/gat.py`、`scripts/inspect_env.py` 与 `tests/`（新增 `tests/test_topology.py`）。
   - 分离任务级执行扰动随机流 `execution_rng`，确保同一随机种子下不同策略共享完全一致的动态拓扑与链路轨迹。
   - 在 `default` 场景、种子 `1042–1046`（共 100 时隙、2,386 到达任务、1,721 成功任务）上运行随机非拒绝策略，实测 $\bar D^{rand}=0.08608214$、$\bar t^{rand}=0.82071578\,\mathrm{s}$，按 $\lambda_0=18.0$ 标定得 **`D_ref = 0.0058`** 并回填配置与文档。
3. **阶段 2A — FastAPI + MySQL 后端服务与接口测试（`gpt-6.2-sol`，`high`）**：
   - 按 `docs/07_上位机可视化调度系统设计.md` 搭建 `backend/`（`app.py`、`config.py`、`db.py`、`schema.sql`、`schemas.py`、`.env.example`、`requirements.txt`、`services/sim_service.py`、`services/train_service.py`、`routers/env.py`、`routers/training.py`、`routers/experiments.py`）及 `tests/test_backend_api.py`。
   - 实现 24 个 REST 接口与 `WS /ws/training/{run_id}` 实时训练流，直接复用 `src/` 核心环境与 PPO 引擎，并兼容历史 3 维与新版 4 维 Checkpoint 推演。
4. **阶段 2B — Vue 3 + Element Plus + ECharts + `localStorage` 前端工程（`gpt-6.1-terra`，`high`）**：
   - 搭建 `frontend/`，实现 `TopologyGraph.vue`（云–边–端三层 SVG 拓扑、拥塞热力、虚线瞬断链路、流光多跳路径高亮、节点拖拽与 `localStorage` 坐标保存）、`NodeResourcePanel.vue`（CPU/NPU/FPGA 跨时隙冻结 vs 本时隙新占分段算力条、内存/存储/队列水位、服务标签与在途任务）、`TaskFlowTable.vue`、`MetricCharts.vue` 以及四大视图 `TrainingWorkbench.vue`、`SimulationSandbox.vue`、`CompareLab.vue`、`ExperimentHistory.vue`。

## 3. 关键代码和文档修改

- `docs/01`–`05`、`docs/07_上位机可视化调度系统设计.md`、`README.md`、`AGENTS.md`、`.gitignore`
- `src/config.py`、`src/topology.py`、`src/tasks.py`、`src/environment.py`、`src/models/gat.py`、`scripts/inspect_env.py`
- `backend/`（FastAPI + SQLAlchemy/PyMySQL 5 张表 + 仿真/训练服务 + REST/WS 路由）
- `frontend/`（Vue 3 + Vite + Element Plus + Pinia + Vue Router + Axios + ECharts + `localStorage`）
- `tests/`（新增 `tests/test_topology.py`、`tests/test_backend_api.py` 并更新其余 9 个测试文件）

## 4. 验证命令与结果证据

- **全量单元测试（43 项，含环境动力学、拓扑波动、GAT 4 维观测与 FastAPI 后端接口）**：
  ```bash
  MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover tests
  ```
  结果：`Ran 43 tests in 4.997s — OK`。
- **Smoke 验证（临时目录，自动清理）**：
  ```bash
  MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python scripts/smoke.py
  ```
  结果：20 项任务完成率 `0.80`、接收率 `0.95`、本地/边缘/云执行率 `0.0 / 0.85 / 0.10`、平均跳数 `2.37`、平均路径长度 `3.47`，`episode_reward / N == objective_score == -1.820921`；`results/` 无任何新增或污染。
- **前端依赖安装与 Vite 生产构建验证**：
  ```bash
  npm --prefix /home/hyh/schedule_test/frontend install && npm --prefix /home/hyh/schedule_test/frontend run build
  ```
  结果：`✓ 2243 modules transformed. ✓ built in 7.73s`。

## 5. 尚未解决的事项和实现边界

1. 本机 WSL 环境暂未启动 MySQL 实例（`/api/health` 中 `db_connected: false`）。前端已支持在未连接 MySQL 时直接浏览磁盘 `results/*/` 历史训练与曲线、运行仿真预览/沙盘推演/多策略评估和启动训练；配置 `backend/.env` 并执行 `backend/schema.sql` 后即可启用 MySQL 持久化与 `results/` 一键同步入库。
2. 新 v6 12 边缘动态波动场景尚未跑完整 1000 轮正式训练实验（遵守 `AGENTS.md` 第 11 条，未向 `results/` 写入开发期短轮产物）。

## 6. 后续建议与常用运行命令

- **启动上位机后端（`gatppo` 环境）**：
  ```bash
  PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
  ```
- **启动上位机前端开发服务器（自动代理 `/api` 与 `/ws` 到 `8000`）**：
  ```bash
  npm --prefix frontend run dev
  ```
- **在 v6 新场景下正式对比 1000 轮 `GAT-PPO` 与单纯 `PPO`**：
  ```bash
  conda run -n gatppo python src/ablation.py --config default --episodes 1000 --seeds 42
  ```
