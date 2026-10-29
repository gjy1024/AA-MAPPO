# 实现审计与维护验收

历史检查对象：公开仓库 `gjy1024/AA-MAPPO` 的commit `e40522cf0376d8ec5ef5c17df3ced24114d55cf3`。检查日期：2026-10-28。该记录来自只读静态检查，未据此运行原论文完整训练。

这些发现描述该公开快照，不推断原论文真正使用的历史实验代码。后续已在用户提供的本地项目中找到更完整的材料：原始源码、四算法核心代码快照、历史日志、权重及截图。本地AA快照中的单token结构仍然存在；其他差异需按具体快照继续核对。修复状态必须由实际变更和验证记录填写。

## 1. 历史公开快照的问题

| 编号 | 静态检查事实 | 影响 | 维护版验收标准 | 当前状态 |
| --- | --- | --- | --- | --- |
| A01 | `algorithms/utils/mlp.py`先映射整个观测，再`unsqueeze(1)`得到单token MHA | 权重恒为1，不能进行多实体关注分配 | 明确实体token与形状；重新训练受影响模型 | 维护版`aamappo/models.py`已改为多实体token；完成512步短链路，完整训练尚未完成 |
| A02 | Actor与Critic均调用该MLPBase | 双侧包含MHA，不代表双侧已实现实体注意力 | 分别定义Actor与Critic的信息边界和聚合 | 维护版支持`none/actor/critic/both`；当前全部实体可见，未实现缺失实体mask |
| A03 | README所列`train_isac_aa_mappo.sh`、`train_isac_standard_mappo.sh`及`train/train_isac.py`未出现在文件树中 | 无法照这些命令复现 | 使用真实入口，安装到评估端到端验收 | 根README和复现指南改用`python -m aamappo`；提供实际smoke产物 |
| A04 | `scripts/train_mpe.sh`调用缺失的`train/train_mpe.py` | 现有Shell入口不完整 | 不继续推荐缺失入口 | 原代码保留，维护版CLI取代推荐入口；不声称修复了所有旧脚本 |
| A05 | `train/train.py`只显式接受`mappo`和`rmappo`；README示例使用`aa_mappo` | 示例参数与入口不匹配 | 算法选择、配置和错误提示一致 | 新CLI用`--attention-mode`选择消融，与旧算法名分开 |
| A06 | `envs/env_core.py`奖励含通信、感知代理及逐机碰撞惩罚，未见速度代价 | 与论文复合奖励描述不完全对应 | 明确逐机/全局口径、权重、公式和评估指标 | 维护版实现共享团队复合奖励与分项指标；新增聚合与参数假设已披露，未混用旧结果 |
| A07 | 历史环境中碰撞每对每机扣20，论文式(20)为全局事件扣50 | 奖励语义和规模不同 | 保留历史定义并为维护版明确选择 | 维护版发生任一碰撞事件扣一次团队惩罚50；仍是软惩罚 |
| A08 | 动作空间为无界Box；环境只截断速度 | 不能据此声称实现有界加速度 | 明确动作边界、缩放及二范数约束 | 维护版tanh动作再投影单位圆并缩放；统一时间步；历史实现保留 |
| A09 | 环境`seed`方法为空；入口虽设置全局NumPy/Torch种子，但逐环境seed本身不起作用 | 并行环境独立随机性需验证 | 使用独立随机数生成器 | 维护版每个环境使用`np.random.default_rng`；当前训练器是单环境参考实现 |
| A10 | 原公开快照主要提供6张PNG，未见原始日志、权重与多种子结果表 | 单凭原公开材料不能重建结果 | 恢复并标识真实材料 | 本地恢复4组日志、24份检查点和16张截图，已归档并生成来源校验；仍无完整多种子统计 |
| A11 | 未见依赖锁定文件和LICENSE正文；README存在MIT徽章 | 安装和复用条件不完整 | 固定测试依赖并保留上游归属 | 已提供`requirements-tested-cpu.txt`、LICENSE、第三方通知和上游MIT声明；直接依赖文件不是完整传递锁 |
| A12 | README仅给出2025年GitHub杂项引用，未分列2026正式论文引用；API未见Release和Tag | 引用与版本追踪不足 | 添加正式引用和软件版本记录 | 已补正式论文信息和`CITATION.cff`；Release和软件DOI不因文件齐备而视为已发布 |

## 2. 本地恢复材料与维护版的关系

| 材料 | 已确认的范围 | 应如何对外说明 |
| --- | --- | --- |
| 原源码与四算法快照 | 已发现，保留历史版本；AA单token问题仍在 | 不静默覆盖原件，修正后另列维护实现 |
| 四组历史训练日志 | `6G_ISAC_Attention`、`Standard_MAPPO`、`No_Clip_MAPPO`、`Naive_Reward_MAPPO`的run1记录已发现 | 作为legacy真实历史曲线，不解释成多种子均值 |
| 历史模型权重 | Actor/Critic检查点已发现 | 补充快照、配置、保存时刻和哈希映射后使用 |
| 截图及轨迹图片 | 已发现 | 不能从PNG恢复全部原始状态或统计不确定性 |
| 新多实体注意力实现 | 属于发表后维护工作，随附512步真实短训练与重新加载评估 | 完整性能结论必须重新训练；smoke不等于论文性能复现 |
| 交互教学数据 | 规则控制和数值示例 | 标明illustrative，不进入算法性能对比 |

## 3. 物理模型还需逐项核对

- 距离是否包含固定高度，单位是否一致。
- 速度和位置积分是否采用同一个时间步。
- 用户是否移动、如何移动，相关状态是否可观测。
- 通信是否为SNR、是否存在真实干扰，用户关联及带宽如何处理。
- 感知项是代理奖励、SNR、CRB还是位置误差，不混用名称。
- 边界裁剪前的越界事件是否记录；碰撞是逐对、逐机还是团队事件。
- 训练与测试的奖励口径、终止规则和时间上限是否一致。

## 4. 维护版验证记录

| 验证项 | 实际命令/记录 | 结果 | 限制 |
| --- | --- | --- | --- |
| 安装与导入 | `requirements-tested-cpu.txt`和根目录[验证记录](../VALIDATION.md) | 已记录本地CPU环境 | 未验证所有操作系统与硬件 |
| 实体token与归一化 | `tests/`；`aamappo/models.py` | 测试覆盖多候选权重、扰动及梯度等；完整结果见根记录 | 全实体可见，没有缺失实体mask |
| Actor/Critic前向及梯度 | `python -m unittest discover -s tests -v` | 后端报告15项单元检查通过，范围见根记录 | 不是性能对比 |
| 环境边界、单位和seed | `tests/`；`aamappo/environment.py` | 单位、边界与随机性采用明确维护实现 | 单环境、理想仿真 |
| 短训练—评估—导出 | `artifacts/maintenance-smoke-v2/` | 当前最新验收记录：512环境步、训练seed7、3个评估场景；具体结果见根验证记录 | 不代表论文性能复现；旧`maintenance-smoke/`为帧对齐改进前快照 |
| 数据平台接入 | `site/data/manifest.json`与`scripts/check_site.py` | 历史日志、维护版评估、教学回放分别接入 | 浏览器交互验收范围见根记录 |
| 多种子主对比 | 待填写 | 未据此声明完成 | 需完整训练预算 |
| 干净环境独立复跑 | 待填写 | 未据此声明完成 | 需独立执行者记录 |

审计发现修复后，保留原始问题及版本信息，不删除历史。此次维护产物以源文件SHA-256记录身份，训练记录中的`git_commit`为null时不补写虚构commit。结构或奖励修改后的新结果应明确标记维护版本，不能重新解释为原论文原始数据。

## 5. 可追溯链接

- [历史注意力实现](https://github.com/gjy1024/AA-MAPPO/blob/e40522cf0376d8ec5ef5c17df3ced24114d55cf3/algorithms/utils/mlp.py)
- [历史环境实现](https://github.com/gjy1024/AA-MAPPO/blob/e40522cf0376d8ec5ef5c17df3ced24114d55cf3/envs/env_core.py)
- [历史动作空间](https://github.com/gjy1024/AA-MAPPO/blob/e40522cf0376d8ec5ef5c17df3ced24114d55cf3/envs/env_continuous.py)
- [历史训练入口](https://github.com/gjy1024/AA-MAPPO/blob/e40522cf0376d8ec5ef5c17df3ced24114d55cf3/train/train.py)
- [历史Shell脚本](https://github.com/gjy1024/AA-MAPPO/blob/e40522cf0376d8ec5ef5c17df3ced24114d55cf3/scripts/train_mpe.sh)

相关文档：[方法说明](METHOD.md)、[复现指南](REPRODUCIBILITY.md)、[实验协议](EXPERIMENTS.md)、[局限性](LIMITATIONS.md)。
