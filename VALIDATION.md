# 本次实际验收

日期：2026-10-29。平台：Windows，Python3.10.5，PyTorch2.7.0+cpu，NumPy2.2.6。

| 检查 | 实际结果 |
| --- | --- |
| `python -m unittest discover -s tests -v` | 17项通过，含实体注意力Q/K/V梯度、单token回归、环境边界/seed、tanh密度、GAE、PPO、模型往返与示范初始化 |
| `python -m aamappo smoke --config configs/maintenance.json --steps 512 --out artifacts/maintenance-smoke` | 完成512个环境步，输出配置、来源、训练指标、模型与评估JSON |
| `python -m aamappo evaluate --checkpoint artifacts/maintenance-smoke/checkpoint.pt --seeds 1001 1002 1003 --out artifacts/maintenance-smoke/reloaded-evaluation.json` | 模型重新加载并完成3个评估场景 |
| `python -m aamappo sweep --config configs/maintenance.json --seeds 1 2 --steps 1024 --dry-run` | 打印4模式×2种子共8条训练命令，不启动长训练 |
| `node --check site/app.js` | JavaScript语法通过 |
| `python scripts/check_site.py` | 静态链接、唯一ID、JSON、manifest与28个历史数据文件校验通过 |

512步模型仍会出现碰撞/越界，不能当作性能成果。页面提供该模型是为了公开展示真实维护验收，而不是挑选成功航迹。三场景统计是一个固定策略的评估分布，不是三次独立训练。

浏览器已验证播放/暂停/拖动/选机、运行切换、注意力教学、训练/执行切换、历史算法/平滑/对数控件、非法JSON拒绝；手机版页面宽度无整体溢出。Matplotlib图使用等比例坐标、不平滑轨迹，训练诊断保留原始点和注明窗口的均线。

后续原配置30k、长rollout30k和示范初始化探索已运行。按验证集固定长rollout检查点，再执行20场景独立测试：通信1.14060、含碰撞0/20、触边0/20、平均回报291.02；规则参考回报742.30仍更强。所有150条逐场景记录无损归档。详情见[探索协议](docs/PILOT_STUDY.md)。

独立同学干净环境复跑、完整2M×多训练seed消融、论文逐图对应与真实飞行验证尚未完成。
