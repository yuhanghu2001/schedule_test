# 多智能体调度与 Codex 委派规则

本文档是 `schedule_test` 仓库的**大脑（Gemini / Claude）→ 执行手（Codex CLI）**协作规范，由根目录 `GEMINI.md` 与 `CLAUDE.md` 自动引入生效。所有参与本仓库的 AI（含大脑与 Codex）同时必须遵守 `AGENTS.md` 中的全部硬规则。

---

## 1. 角色分工与协作边界

- **大脑（Gemini / Claude）**：负责需求理解、数学建模与方案决策、系统架构设计、任务拆解、选择 Codex 模型与推理档位、下发任务书、逐行审核改动与独立复跑验证、向用户汇报、按用户反馈发起二次调度，以及整轮结束后的中文 Git 提交推送；`memory/` 留档只在用户明确要求时执行。
- **执行手（Codex CLI）**：负责在大脑限定的文件与规格范围内执行代码实现、局部重构、测试补全、问题定位、前后端页面与接口开发、UI 截图审校与样式修复。
- **防套娃与防越权边界**：
  - Codex **不得**再嵌套调用 `codex` 或其他外部代理（除非大脑显式启用 `ultra`）。
  - Codex **不得**自行执行 `git commit` / `git push`，**不得**自行写 `memory/` 留档，**不得**将临时测试或 smoke 输出写入 `results/`。
- **大脑直接处理（无需派发 Codex）的场景**：
  - 纯问答解释、方案探讨、架构/数学决策讨论；
  - 1–3 行显而易见的微小修改；
  - 少量文件只读查看、用户明确要求的 `memory/` 留档、Git 状态检查与提交推送。
  - 其余实质性编码、多文件改动、复杂排错、测试补充及前后端开发，统一按本规则派发 Codex 执行并审核。

---

## 2. 本机 Codex 可用模型与特长定位

基于本机 Codex CLI（`v0.170.0`）可用模型，采用 **4 个主力模型 + 1 个额度降级备选**：

| 模型 slug | 核心长处与定位 | 支持 `model_reasoning_effort` | 适用主战场 |
| :--- | :--- | :--- | :--- |
| `gpt-6.2-astra` | 旗舰强推理，数学严谨性与复杂代码精度最高（500k 上下文） | `low` / `medium` / `high` / `xhigh` / `max` / `ultra` | 数学推导核验、C1–C24 约束对齐、Dinkelbach/KKT 连续优化、GAT-PPO 梯度/收敛疑难诊断、高风险核心改造 |
| `gpt-6.2-sol` | 主力高速编码模型，工程落地与速度最均衡（500k 上下文） | `low` / `medium` / `high` / `xhigh` / `max` / `ultra` | 仿真环境常规开发、基线/消融/评估脚本、后端 FastAPI/WebSocket 服务、常规测试与单模块重构 |
| `gpt-6.1-terra` | 视觉理解、界面布局与计算机交互专长（500k 上下文） | `low` / `medium` / `high` / `xhigh` / `max` / `ultra` | 上位机可视化前端页面、动态拓扑图/多跳路径动效/四目标仪表盘开发，以及配合 `-i <截图>` 做视觉验收与样式修复 |
| `gpt-6-luna` | 轻量高速低成本模型（272k 上下文） | `low` / `medium` / `high` / `xhigh` / `max`（不支持 `ultra`） | 简单脚本、配置常量微调、日志/JSONL 指标提取、文档格式同步、代码检索摘要、模板化样板改动 |
| `gpt-6.1-sol` *(备选)* | 上一代高性价比编码模型，复杂任务表现接近 Astra | `low` / `medium` / `high` / `xhigh` / `max` / `ultra` | `gpt-6.2-astra` 或 `gpt-6.2-sol` 触发限速或额度紧张时的直接平替 |

---

## 3. 推理程度（`model_reasoning_effort`）分级标准

- **`low`**：路径完全明确、无歧义的机械性改动或信息提取（改常量、摘录日志、小修文档/样式）。
- **`medium`**：日常默认主力档位。规格清晰的单模块功能实现、标准 API 接口、常规前端组件、常规单元测试。
- **`high`**：存在跨模块耦合、状态转移约束或非平凡排错（环境动力学改动、PPO 训练/评估流程改造、前后端协议联调、复杂前端交互状态）。
- **`xhigh`**：要求强数学严密性或高风险架构正确性（C1–C24 约束与代码严格对齐、Dinkelbach/KKT 边界与数值稳定性、奖励不变量排查、训练不收敛根因诊断、核心架构审校）。
- **`max`**：仅作为**升级兜底档位**，在 `xhigh` 仍无法攻克的极难 Bug 或隐蔽数值问题时启用，默认不作为首轮选项。
- **`ultra`**：会触发 Codex 内部自动拆分子代理。**默认禁用**——大脑已负责任务拆解与审核，避免双层套娃与不可控耗时；仅在用户明确要求整包托管大型独立子任务时使用。

---

## 4. `schedule_test` 全场景路由矩阵

### 4.1 数学建模、仿真环境与调度算法

| 任务场景 | 典型文件 / 内容 | 推荐模型 | 推理程度 | 沙箱 `-s` | 关键约束提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 数学公式推导核验、C1–C24 约束审计、奖励公式与量纲标定 | `docs/02_数学模型与C1-C24约束.md`、`docs/05_训练奖励与PPO对齐说明.md` | `gpt-6.2-astra` | `xhigh`（简单符号核对用 `high`） | `read-only`（审计） / `workspace-write`（改文档） | 改环境动力学前必须先改 `docs/02` 并经大脑确认 |
| 连续子问题优化（Dinkelbach 上行功率 / KKT 并发算力） | `src/continuous.py`、`tests/test_continuous.py` | `gpt-6.2-astra` | `xhigh` | `workspace-write` | 严禁把启发式/局部最优误写成全局最优 |
| 仿真环境动力学核心改造（时隙延续、多跳路由、不确定性、奖励结算） | `src/environment.py`、`src/topology.py`、`src/tasks.py` | `gpt-6.2-astra`（强耦合）或 `gpt-6.2-sol`（局部新增） | `high`–`xhigh` | `workspace-write` | 核验 `x/y/g/f/p_up/z/z_service` 语义、`deadline_s` 联动、奖励 `2:1:1:1` |
| GAT / PPO 网络结构与训练更新逻辑改造 | `src/models/gat.py`、`src/models/policy.py`、`src/algorithms/ppo.py` | `gpt-6.2-astra` | `high`–`xhigh` | `workspace-write` | 不引入 PyG、Ray、Gym 等重型依赖 |
| 训练不收敛、指标异常、KL/梯度问题深度诊断 | `results/*/train_metrics.jsonl`、`src/algorithms/ppo.py`、`src/environment.py` | `gpt-6.2-astra` | `xhigh`（卡壳升 `max`） | 先 `read-only` 定位，再 `workspace-write` 修复 | 必须基于真实日志与代码证据，不盲改超参 |
| 启发式基线、消融实验、评估聚合、训练曲线绘图 | `src/baselines.py`、`src/trainer.py`、`src/evaluation.py`、`src/plotting.py`、`scripts/` | `gpt-6.2-sol` | `medium`（跨文件接口调整用 `high`） | `workspace-write` | smoke 与开发验证只用系统临时目录，不写 `results/` |
| 参数配置微调、常规单测补充、实验结果统计提取 | `src/config.py`、`tests/`、`results/*/metrics.json` | `gpt-6-luna`（统计/微调）或 `gpt-6.2-sol`（补单测） | `low`–`medium` | `read-only` / `workspace-write` | 仅写核心功能测试，不引入复杂 mock 框架 |

### 4.2 前后端分离可视化上位机调度系统（后续规划）

| 任务场景 | 典型内容 | 推荐模型 | 推理程度 | 沙箱 `-s` | 关键约束提醒 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 系统架构与前后端接口契约（REST / WebSocket 推演流 / 离线回放与在线交互） | 接口 Schema、事件协议、状态机设计 | 大脑主导；需细化或核验时调 `gpt-6.2-astra` | `high`–`xhigh` | `read-only` / `workspace-write` | 保持最简直接架构，直接复用 `src/` 调度与环境引擎，不复制仿真逻辑 |
| 后端服务开发（封装模型推理、仿真步进、历史 `results/` 回放与控制 API） | FastAPI / WebSocket 路由、状态序列化、推演控制 | `gpt-6.2-sol` | `medium`（并发推演/实时流用 `high`） | `workspace-write` | 不写冗余防御式校验层、鉴权或多余抽象 |
| 前端核心可视化界面（云–边–端动态多跳拓扑图、链路拥塞热力、多跳路径高亮、四目标实时仪表盘与控制栏） | 页面布局、拓扑图与图表组件、交互联动 | `gpt-6.1-terra`（视觉布局/图交互）或 `gpt-6.2-sol`（复杂状态/数据层） | `medium`–`high` | `workspace-write` | 视觉呈现交 `gpt-6.1-terra`，重逻辑数据层可交 `gpt-6.2-sol`；严禁添加装饰性小图标，必须沿用 `dashboard.css` 深蓝大屏风格 |
| 前端视觉验收、排版微调与渲染 Bug 修复 | 对照页面截图检查遮挡、层级、配色、响应适配 | `gpt-6.1-terra` | `medium`–`high` | `read-only`（审图） / `workspace-write`（修复） | 必须配合 `-i <截图路径>` 直接输入页面截图，核查无装饰小图标且风格与现有页面一致 |
| 前后端联调与端到端排错（字段对齐、推演卡顿、WebSocket 同步异常） | 跨端定位与修复 | `gpt-6.2-sol`（常规）或 `gpt-6.2-astra`（复杂时序并发） | `high` | `workspace-write` | — |
| 静态文案、类型同步、简单样式/配置调整 | 常量、配色、小修布局 | `gpt-6-luna` | `low` | `workspace-write` | — |

### 4.3 通用辅助

| 任务场景 | 推荐模型 | 推理程度 | 沙箱 `-s` |
| :--- | :--- | :--- | :--- |
| 代码第二视角独立审查（`codex exec review`） | `gpt-6.2-astra`（核心算法）或 `gpt-6.2-sol`（常规） | `high` | `read-only` |
| 仓库快速检索、大文件摘要、文档格式同步 | `gpt-6-luna` | `low` | `read-only` / `workspace-write` |

---

## 5. 标准 CLI 调用规范

### 5.1 首次派发（写代码 / 改文档）

```bash
codex exec \
  -C /home/hyh/schedule_test \
  -m <model_slug> \
  -c model_reasoning_effort="<effort>" \
  -s workspace-write \
  -o /tmp/codex_last_msg.txt \
  "<结构化任务书>"
```

### 5.2 只读诊断 / 数学核验 / 代码审查

将 `-s workspace-write` 改为 `-s read-only`；代码审查可直接用 `codex exec review -m <model_slug> -c model_reasoning_effort="high"`。

### 5.3 前端视觉审校 / 样式修复（附截图）

```bash
codex exec \
  -C /home/hyh/schedule_test \
  -m gpt-6.1-terra \
  -c model_reasoning_effort="medium" \
  -s workspace-write \
  -i /tmp/ui_screenshot.png \
  -o /tmp/codex_last_msg.txt \
  "<视觉检查与修复要求>"
```

### 5.4 审核打回 / 迭代返工

优先复用上次会话上下文，减少重复读码：

```bash
codex exec resume --last \
  -m <model_slug> \
  -c model_reasoning_effort="<effort>" \
  -o /tmp/codex_last_msg.txt \
  "<具体缺陷清单与修改要求>"
```

若需要彻底换思路或高风险大规模改造，再新开会话（必要时加 `--worktree` 隔离，审核通过后合并）。任务书较长时，通过标准输入（`codex exec ... -`）传入，避免 shell 转义风险；所有临时文件统一写入 `/tmp/`，不得污染仓库或 `results/`。

---

## 6. 任务书契约（大脑 → Codex）

大脑每次派发任务给 Codex，任务书必须精炼写清 6 项：

1. **目标**：要解决的具体问题与验收标准。
2. **文件边界**：允许新增/修改的文件清单，禁止触碰的范围。
3. **硬约束**：点明与本次任务直接相关的 `AGENTS.md` 规则（无防御式代码、最简实现、变量语义、先改数学文档、不污染 `results/` 等）。
4. **规格与契约**：大脑已定稿的数学公式、接口签名、数据结构或 UI 交互要求。
5. **自测命令**：指定完成后运行的验证命令（如 `MPLCONFIGDIR=/tmp/schedule_test_mpl PYTHONPATH=. /home/hyh/miniconda3/envs/gatppo/bin/python -m unittest discover tests` 或前端构建命令），并清理临时目录。
6. **禁止越权**：明确要求**不执行 `git commit` / `git push`、不写 `memory/`、不嵌套调用 `codex`**；结束时只汇报改动文件、核心逻辑、验证输出与潜在风险。

---

## 7. 大脑审核、升级与向用户汇报闭环

Codex 每次执行完毕，大脑必须按以下闭环收口，**禁止直接转述 Codex 结论而不复核**：

1. **Diff 与规范审核**：运行 `git status` 和 `git diff`，逐行核查：
   - 是否仅修改授权文件，有无残留临时文件或写入 `results/`；
   - 是否违反 `AGENTS.md`（防御性 `try/except`、冗余 `if` 兜底、过度抽象、重型依赖）；
   - 公式、约束编号（C1–C24）、变量语义（`x/y/g/f/p_up/z/z_service`）与 `docs/` 是否同步；
   - 是否虚报最优性或实验指标。
2. **独立复跑验证**：大脑自行复跑关键单测 / 临时目录 smoke / 前后端构建，拿到真实命令输出。
3. **打回与自动升级**：
   - 未通过审核 → 给出具体文件、行号、缺陷和目标行为，用 `codex exec resume --last` 打回返工。
   - **升级机制**：同一子任务连续返工 **2 轮**仍未达标，按 `gpt-6-luna → gpt-6.2-sol → gpt-6.2-astra` 升级模型或上调一档推理程度（最高至 `xhigh`，极难问题再升 `max`）；若仍卡壳，由大脑直接接管修复并向用户说明原因。
4. **向用户汇报**：简洁呈现四项内容：
   - 核心改动与结果摘要（附可点击文件链接）；
   - 本次调度的 Codex 模型与推理档位（含返工/升级记录）；
   - 大脑审核结论与真实测试/构建验证证据；
   - 遗留边界与需用户确认/调整的事项。
5. **用户反馈再调度与最终收口**：
   - 用户提出调整意见后，大脑重新拆解目标再次调度 Codex 并重复审核闭环；
   - 整轮任务确认收尾后，由**大脑**检查 `git status`，使用中文 commit message 提交全部改动并推送。
   - 只有用户明确要求写入或更新 memory 时，才由**大脑**统一在 `memory/YYYYMMDD_HHMMSS-对话内容标题/README.md` 留档并更新 `memory/README.md`；任务完成或提交推送本身不代表留档授权。
