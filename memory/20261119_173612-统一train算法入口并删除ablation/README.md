# 统一 train 算法入口并删除 ablation

## 目标
按用户指定在 src/train.py 实现 --model gatppo、--model ppo、--model gatppo and ppo，双算法完成后绘制对比图，并删除 src/ablation.py。

## 完成事项与关键修改
- train.py 改为可导入的 train_one/main 入口，默认完整 GAT-PPO，支持完整算法拼写别名 gatpppo、带引号/不带引号的 gatppo and ppo，保留单算法续训和中断收尾。
- 双算法顺序执行，分别重置相同随机种子、实例化独立环境/模型/优化器，使用同配置、回合数和固定验证种子；父目录保存 comparison_config，子目录 gatppo/ppo 保存配置、日志、best/last 与原有曲线。
- plotting.py 新增两算法训练和固定验证叠加图，分别为 training_comparison.png、evaluation_comparison.png。验证阴影取固定验证种子标准差，单验证点用数据点/误差线，短训和中断不生成伪造数据。
- 新权重文件为 best_gatppo.pt / last_gatppo.pt 和 best_ppo.pt / last_ppo.pt。
- 删除 ablation.py；原观测打乱函数和包装逐段迁入 environment.py，不改仿真状态转移、任务时限、动作约束或奖励。原 shuffled_gat 模式继续支持。
- policy.py 定义已使用模型名映射，历史 gat/flat 权重、旧命令可识别；评估、后端训练和推演同步新名称。
- 前端工作台、草稿和历史回填同步 gatppo/ppo；后端递归发现双算法子目录权重/配置，历史曲线路由支持嵌套运行路径。
- README、AGENTS 目录职责、开发说明、实验协议和上位机说明同步；未引入新算法依赖或新的训练脚本。

## 验证证据
- CLI --help、临时目录 small_smoke 双算法各2回合通过，两个模型的预算/配置/验证种子一致，权重和两张对比图存在，实际查看图像确认两算法图例和数据正确。
- 检查权重变化：GAT-PPO 更新 GAT、普通 MLP 分支不变；PPO 更新 MLP、GAT 分支权重与初始化一致。
- gatpppo 别名1回合、带引号双算法1回合通过。
- PPO 2回合 checkpoint 续训至第3回合，与连续3回合训练全部模型张量完全一致。
- 临时 legacy_flat.pt（由本次短训模型设置历史类型名构成）评估通过；新 GAT-PPO 和历史 flat 类型在真实 SimManager 中单步推理通过。
- 迁移后的图打乱观测保留节点/动作，并只按排列变化邻接；物理 step 奖励与未包装环境一致。
- TestClient 临时 manager 验证嵌套路径 pair/ppo 曲线返回2条训练记录，API 普通 PPO 训练1回合完成；未写真实数据库或正式 results。
- 双算法中断测试：首个训练回合后发送 SIGINT，保存已有 GAT-PPO last 和训练对比图，不产生未执行 PPO 的虚假记录。
- npm run build、git diff --check 通过；未新增装饰图标。
- 检查实际服务训练/仿真会话数均0后重启 npm start，加载新名称和模块位置；前端、健康检查与 OpenAPI 返回200，db_connected:true，API 默认模型 gatppo。
- 所有训练与评估验证产物位于系统临时目录 /tmp/schedule-train-*，完成后删除。未运行正式长训、未新增 results 产物。初次临时验证脚本缺少仓库搜索路径，补正后通过；此为验证脚本路径问题。

## 未解决事项与后续使用
本次功能已完成，未执行正式长训或性能对比结论。开发服务已重启并保持运行，可直接刷新前端，不需重复 npm start。普通训练在 gatppo 环境执行 python src/train.py --model gatppo / ppo；两算法使用 --model gatppo and ppo。单算法 --resume 续训；--resume/--output 不用于双算法。当前未提交或推送，保留前次方案记录改动。
