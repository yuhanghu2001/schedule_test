# 20261119_105544 — 建立大脑调度 Codex 委派规则

## 1. 对话目标

1. 确认本机环境中 Gemini / Claude 通过命令行直接调度 Codex CLI 及动态调节推理程度（`model_reasoning_effort`）的可行性。
2. 结合本机 Codex CLI 实际可用模型与能力定位，以及 `schedule_test` 仓库当前研究内容（数学公式推导、仿真环境建模、GAT-PPO 算法训练调试）与后续规划（前后端分离可视化上位机调度系统），建立正式的多智能体调度与审核闭环规则。
3. 解决规则存放位置问题，兼顾 Gemini、Claude 与 Codex 三方工具的自动加载机制，避免 Codex 误读大脑规则导致角色冲突。

## 2. 已完成事项

1. **核验本机 Codex CLI 环境与可用模型**：
   - 验证 `/home/hyh/.local/bin/codex`（指向 standalone `v0.170.0`）及其 `codex exec` 子命令与参数（`-m`、`-c model_reasoning_effort=...`、`-s`、`resume --last`、`review`、`-i`、`-o`、`--worktree`）。
   - 检查 `~/.codex/config.toml` 与 `~/.codex/models_cache.json`，确认当前可用主力模型为 `gpt-6.2-astra`、`gpt-6.2-sol`、`gpt-6.1-terra`、`gpt-6-luna`（及备选 `gpt-6.1-sol`），以及各自支持的推理档位（`low`/`medium`/`high`/`xhigh`/`max`/`ultra`）。
2. **确立“独立规则文档 + 大脑专属入口 + 共享硬规则”三层结构**：
   - 不把“调度 Codex”的大脑规则写入 `AGENTS.md`，避免 Codex 运行时读取 `AGENTS.md` 产生嵌套调度与角色冲突。
   - 统一规则源写入 `docs/06_多智能体调度与Codex委派规则.md`。
   - 根目录新增 `GEMINI.md`（使用 Antigravity `@[...](...)` 引入）与 `CLAUDE.md`（使用 `@AGENTS.md` 与 `@docs/...` 引入），确保 Gemini 与 Claude 会话自动以大脑身份运行。
   - 仅在 `AGENTS.md` 新增第 16 条（明确执行手边界：不嵌套调用代理、不擅自提交/留档）并同步目录职责与 `README.md` 文件概览。

## 3. 关键代码和文档修改

- 新增 `docs/06_多智能体调度与Codex委派规则.md`：完整定义大脑/执行手分工、4+1 模型特长矩阵、6 级推理程度使用原则、`schedule_test` 算法研发与上位机前后端全场景路由表、`codex exec` 调用模板（含 `-i` 截图视觉审校与 `resume --last` 返工）、6 要素任务书契约以及大脑 Diff 审核/独立验证/2 轮升级/向用户汇报闭环。
- 新增 `GEMINI.md`：Gemini / Antigravity 自动加载的大脑入口规则。
- 新增 `CLAUDE.md`：Claude Code 自动加载的大脑入口规则。
- 修改 `AGENTS.md`：补充第 16 条多智能体执行边界，并在目录职责补充 `GEMINI.md` / `CLAUDE.md`。
- 修改 `README.md`：在文件概览树中同步新增文档与入口文件。

## 4. 验证命令与结果证据

- `codex exec --help`：退出码 `0`，确认支持 `-m`、`-c`、`-s`、`-i`、`-o`、`--worktree`、`resume`、`review`。
- `wc -c GEMINI.md CLAUDE.md AGENTS.md docs/06_多智能体调度与Codex委派规则.md`：
  - `GEMINI.md`（594 B）+ `docs/06_多智能体调度与Codex委派规则.md`（13,361 B）= `13,955 B`，远低于 Antigravity 单规则文件 `24,000 B` 展开上限。
- `git status`：仅包含本次新增与同步的规则/文档及 `memory/` 记录，未触碰 `src/`、`tests/` 或 `results/`。

## 5. 尚未解决的事项和实现边界

1. 首次实际通过 Gemini/Claude 终端执行 `codex exec` 时需用户批准外部命令权限（勾选始终允许 `codex exec` 后可自动调度）。
2. 若后续本机 Codex 模型清单更新，需同步修订 `docs/06_多智能体调度与Codex委派规则.md` 第 2 节与第 4 节路由表。

## 6. 后续建议与常用运行命令

- 用一次小任务实测闭环（例如让大脑派 `gpt-6-luna(low)` 做只读检查，验证 `codex exec` 端到端链路）：
  ```bash
  codex exec -C /home/hyh/schedule_test -m gpt-6-luna -c model_reasoning_effort="low" -s read-only "列出 src/ 下各模块的核心职责并汇总为 5 行以内"
  ```
