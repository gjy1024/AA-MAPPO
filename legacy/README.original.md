# AA-MAPPO: Attention-Augmented MAPPO (融合注意力机制的多智能体近端策略优化)

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

本项目是研究论文/报告 **《6G 背景下基于注意力机制和强化学习的无人机集群路径规划方法研究》** 的官方代码实现。

面对 6G 通感一体化（ISAC）范式下的多目标严苛耦合与复杂电磁干扰，本项目基于经典的 **MAPPO** 算法，创新性地引入了 **多头注意力机制 (Multi-Head Attention)**。在“集中式训练、分布式执行”（CTDE）架构下，使无人机智能体能够从高维局部观测中自适应筛选关键环境特征。代码库原生支持并验证了算法在通信容量、感知精度、飞行能耗与防碰撞等多目标联合优化上的帕累托最优性能。

## 🌟 核心特性 (Features)

- **多头注意力特征提取 (Multi-Head Attention)**: 在 Actor 和 Critic 网络前端植入基于 `Query-Key-Value` 的自注意力模块（默认 4-Heads），赋予无人机动态感知与“自动划重点”的能力，自主过滤环境噪声并聚焦高价值通感目标，彻底解决多机博弈中的信息冗余与过载。
- **高保真 ISAC 联合物理模型**: 摒弃传统简化模型，底层环境严格基于 6G 毫米波大尺度衰落特性与雷达双程衰减机制。内置精心设计的复合奖励函数，融合了通信香农容量、感知定位克拉美-罗下界 (CRB)、防碰撞底线以及精细的旋翼飞行能耗惩罚。
- **稳健的 PPO 剪切机制**: 采用同策略（On-policy）近端策略优化，利用截断概率比（Clipping）机制严格限制无人机策略更新的步长，彻底消除了传统多智能体算法（如 MADDPG）常见的轨迹剧烈抖动与策略崩溃现象。
- **完备的消融实验基准**: 代码原生支持报告中提及的所有基线与消融算法，包括 `Standard MAPPO`（退化特征提取）、`No_Clip MAPPO`（移除剪切保护）以及 `Naive Reward MAPPO`（传统单一奖励模型），方便研究者一键复现对比实验。

## 📁 目录结构 (Directory Structure)

```text
AA-MAPPO/
├── algorithms/     # 核心算法模块（包含 AA-MAPPO 的多头注意力特征提取器及 PPO 优化逻辑）
├── envs/           # 6G ISAC 场景环境封装（内置无人机运动学、物理信道与多目标奖励计算）
├── runner/         # 运行调度器（负责多线程环境数据 Rollout 以及驱动网络前向/反向传播）
├── scripts/        # 一键执行的 Shell 训练脚本 (含 AA-MAPPO 及各消融实验基准)
├── train/          # 训练主入口（实例化环境、算法与 Runner，启动主循环）
├── utils/          # 通用工具（日志记录、奖励平滑缩放、优势函数 GAE 计算等辅助组件）
└── config.py       # 全局超参数与运行配置中心

# 1. 创建并激活虚拟环境
conda create -n aa-mappo python=3.8 -y
conda activate aa-mappo

# 2. 安装 PyTorch (请根据您的硬件及 CUDA 版本调整)
pip install torch torchvision torchaudio

# 3. 安装基础科学计算与强化学习依赖
pip install numpy tensorboard gym wandb

cd scripts
# 运行本文提出的核心算法: AA-MAPPO
bash train_isac_aa_mappo.sh

# 运行消融实验基准: Standard MAPPO (验证注意力的必要性)
bash train_isac_standard_mappo.sh

python train/train_isac.py \
    --env_name "ISAC_UAV_Swarm" \
    --algorithm_name "aa_mappo" \
    --experiment_name "uav_3_users_2" \
    --num_env_steps 2000000 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --num_heads 4 \
    --clip_param 0.2 \
    --gamma 0.99

tensorboard --logdir=./Results_TRAJ_Standard_MAPPO/

@misc{aamappo2025,
  title={6G 背景下基于注意力机制和强化学习的无人机集群路径规划方法研究},
  author={gjy1024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/gjy1024/AA-MAPPO}},
  year={2025}
}

