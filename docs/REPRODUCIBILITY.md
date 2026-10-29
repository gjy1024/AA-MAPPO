# 复现指南与验收记录

目标是让其他人能够从干净环境安装项目、评估明确来源的模型、重建结果。代码可以运行、训练可以完成、论文主要结论得到复现，是三个不同层级。

## 当前状态

本地原始材料已经找到：包括四类算法的核心代码快照、训练日志、模型权重和截图。它们比先前公开仓库的材料更完整，但“材料存在”不自动证明模型与每张论文图的对应关系、种子设置或统计充分性。

本维护项目区分三类来源：历史训练日志（legacy）、维护版短链路验证（maintenance smoke）和规则控制/数值教学示例（illustrative）。维护版已经生成512个环境步的短训练产物，并完成保存、重新加载和3个场景种子的确定性评估。该记录只验证运行闭环，不声明完成原论文性能复现或新的多训练种子实验。完整验收范围见根目录[验证记录](../VALIDATION.md)。

| 层级 | 需要完成的工作 | 通过标准 |
| --- | --- | --- |
| 基础检查 | 依赖安装、配置解析、网络前向、环境步进 | 命令退出正常，维度和数值检查通过 |
| 短链路运行 | 维护版短训练或加载指定模型，随后独立评估和导出 | 输出带来源信息的日志、轨迹和指标；只证明管线可用 |
| 完整训练 | 明确配置下的完整训练预算 | 每次运行保存配置、种子、版本、权重和日志 |
| 结果复现 | 多个独立训练种子、公平基线、统一评估 | 能报告结果分布和与论文的差异，而非仅一张相似图 |

## 1. 环境要求

本地短训练记录使用Windows、Python3.10.5、PyTorch2.7.0+cpu和NumPy2.2.6，训练器当前为CPU参考实现。[requirements-tested-cpu.txt](../requirements-tested-cpu.txt)固定了直接依赖版本，并非包含所有传递依赖的完整锁文件。论文报告的Python3.8是历史设置，不应与维护版验证环境混为一谈；Linux自动检查是否通过，应看实际GitHub Actions记录。

以下命令均在仓库根目录运行。Windows PowerShell示例：

```powershell
python -c "import sys; assert sys.version_info[:2] == (3, 10), 'Select Python 3.10 first'"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-tested-cpu.txt
python -m aamappo --help
```

macOS/Linux将激活命令替换为`source .venv/bin/activate`。使用新的运行目录，避免混合实验或覆盖原记录。

### 只查看平台

```bash
python scripts/serve.py --port 8000
```

浏览器访问`http://localhost:8000`。网站本身不依赖PyTorch；直接打开`site/index.html`可以使用内置教学交互，但读取历史JSON档案需通过HTTP服务器。

### 自检、短训练和重新加载评估

```bash
python -m unittest discover -s tests -v
python -m aamappo smoke --config configs/maintenance.json --steps 512 --out runs/smoke-001
python -m aamappo evaluate --checkpoint runs/smoke-001/checkpoint.pt --seeds 1001 1002 1003 --out runs/smoke-001/reloaded-evaluation.json
```

`smoke`会真实训练，不是打印模拟结果；512步仍然只用于短链路验收。也可以直接评估随仓库提供的维护版检查点：

```bash
python -m aamappo evaluate --checkpoint artifacts/maintenance-smoke-v2/checkpoint.pt --seeds 1001 1002 1003 --out runs/bundled-checkpoint-evaluation.json
```

检查点加载器只接受维护版格式，拒绝历史架构权重。只加载可信来源的检查点。评估JSON可直接导入网站；动画显示首个场景，`episodes`和`summary`包含所有评估场景。

### 完整训练与消融计划

```bash
python -m aamappo train --config configs/maintenance.json --attention-mode both --seed 1 --steps 2000000 --out runs/both-seed1
python -m aamappo sweep --config configs/maintenance.json --seeds 1 2 3 4 5 --steps 2000000 --dry-run
```

四种注意力模式为`none / actor / critic / both`。`sweep`默认只打印计划，显式加`--execute`才启动全部训练；不要同时使用`--execute`和`--dry-run`。4×5次完整训练属于后续实验，不因提供命令而视为完成。

维护版真正不使用裁剪的对照由配置`no_clip: true`控制，距离型奖励由`reward_mode: "naive"`控制。它们与历史No_Clip和Naive记录分别命名和报告。

### 教学数据与历史档案

```bash
python -m aamappo demo --config configs/maintenance.json --seed 7 --out runs/teaching-demo.json
python scripts/build_site_data.py --evaluation artifacts/maintenance-smoke-v2/evaluation.json
python scripts/check_site.py
```

`demo`生成规则控制轨迹，来源标识为`illustrative`，不是训练后的AA-MAPPO策略。需要重新导入本人的原始材料时，使用实际来源路径运行：

```bash
python scripts/import_legacy.py --source "原始AA-MAPPO_Project目录"
```

原始材料目录不是仓库发布后必然存在的路径；直接使用仓库已归档数据时，无需重复导入。上述导入和站点数据构建会更新对应归档/展示文件，维护前先检查工作区差异。

## 2. 历史实验材料怎样使用

原始材料包含 `6G_ISAC_Attention`、`Standard_MAPPO`、`No_Clip_MAPPO` 和 `Naive_Reward_MAPPO` 的 `run1` 记录。导入时保留原始JSON、检查点和来源清单，不改写原始数值。平台可以重新绘制这些历史训练曲线，但不能把单个`run1`解释成多个独立种子的均值。

需要继续核对：每个运行对应哪一份核心代码、完整配置和种子；截图对应哪条日志；权重保存时刻及选择规则；是否有独立评估轨迹。原始目录中的绝对路径键可以在展示数据中规范化，但须保留映射和原文件校验信息。

维护版修正了实体token等实现选择后，使用新的命名空间和检查点格式。历史权重不能直接冒充维护版模型；结构或奖励改变后需要重新训练。

## 3. 一次运行必须保存什么

建议为每次运行分配唯一目录，例如 `results/<experiment>/<variant>/seed-<id>/`。实际目录结构由命令行接口确定，文档必须与代码保持一致。

- 完整解析后的配置，而不仅是用户传入的覆盖参数。
- Git commit、运行日期、执行命令、依赖版本和硬件。
- 训练种子与评估场景种子；并行环境应有独立随机数生成器。
- 训练步数口径：环境步、智能体步和优化步分别说明。
- 原始逐回合或逐评估点日志；平滑曲线不能替代原始数据。
- 模型检查点及选择规则，例如固定训练预算后的模型或验证集选择。
- 评估策略是否确定性、评估回合数、终止规则、评价指标定义。
- 轨迹坐标、分项奖励及安全事件，而不仅是 PNG。

## 4. 平台数据接口

平台用 JSON 展示评估轨迹。最小结构如下；该段说明结构，不是实验数据：

```json
{
  "schema_version": "1.0",
  "kind": "evaluation",
  "provenance": {
    "status": "evaluation",
    "policy": "实际策略名",
    "seed": 0,
    "checkpoint": "实际检查点路径或标识",
    "notes": "版本、实验范围与已知限制"
  },
  "config": {"map_size": 100, "safe_distance": 2, "dt": 0.1},
  "users": [[40, 60]],
  "trajectory": [],
  "summary": {}
}
```

每个轨迹帧应有 `step`、`time`、`positions`、`velocities` 和 `metrics`。指标命名与单位应在导出器和 [实验协议](EXPERIMENTS.md) 中保持一致。`trajectory` 为空的数据只用于说明接口，不能作为可播放的评估记录。

仓库当前使用的来源状态：

| 来源状态 | 含义 | 能否支撑算法性能结论 |
| --- | --- | --- |
| `illustrative` | 手工、规则控制或教学参数生成 | 否 |
| `maintenance-smoke` | 维护版给定检查点的评估；随附数据来自512步训练 | 短训练不支持算法性能结论 |
| 单模型评估 | 给定检查点在指定场景的评估 | 仅支撑这一模型和场景的表现 |
| 多种子实验 | 多次独立训练后统一评估 | 可以在设计范围内形成统计证据 |
| `historical-single-run` | 本地恢复并保留来源的原始训练记录 | 支撑该运行的记录；与论文图及配置的精确对应仍需核对 |

当前导出器对维护版模型统一使用`maintenance-smoke`标签，即使用户自行进行了更长训练，也不能仅凭这个标签断定预算；请查看`trained_environment_steps`、配置与provenance。正式多种子研究发布时，应补充能描述实验范围的来源记录。

当前交付以`artifacts/maintenance-smoke-v2/`作为最新短训练验收记录，训练种子为7、环境步数为512、完成回合为2，评估场景种子为1001、1002、1003。3个评估场景均来自同一训练模型，不是3个独立训练种子。网页回放有201帧，即初始状态加200步；记录范围见根目录[验证记录](../VALIDATION.md)。

旧目录`artifacts/maintenance-smoke/`保留为改进注意力权重与轨迹帧对齐之前的验收快照，不作为当前网页数据。旧记录不删除，也不把新实现的源码身份写回旧provenance；查看或复跑当前维护版时使用v2记录。

## 5. 独立复跑记录

请其他人只使用仓库文档，在干净环境复跑。记录实际过程，不要将作者自己的成功运行写成独立验证。

| 字段 | 记录 |
| --- | --- |
| 执行者与日期 | 待填写 |
| 操作系统与硬件 | 待填写 |
| Git commit / Release | 待填写 |
| 安装命令与结果 | 待填写 |
| 运行命令与退出状态 | 待填写 |
| 模型和数据来源 | 待填写 |
| 生成文件及指标 | 待填写 |
| 与预期差异 | 待填写 |
| 尚未完成内容 | 待填写 |

## 6. 问题报告

报告问题时附上系统、commit、配置、精确命令和最短错误输出。不要上传访问令牌、账号凭据或私人数据。对缺失检查点、安装冲突、结果偏差分别建立 Issue；修复后记录是否影响历史结果。

规范参考：[NeurIPS 代码与数据指南](https://neurips.cc/public/guides/CodeSubmissionPolicy)、[IEEE 可复现研究指南](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/)。
