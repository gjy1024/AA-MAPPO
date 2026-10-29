# 参与维护

先阅读 [证据与局限](docs/LIMITATIONS.md) 和 [实验协议](docs/EXPERIMENTS.md)。

1. 新建分支，说明问题、触发条件、预期行为。
2. 修改算法时记录配置变化；不要覆盖历史模型或把旧日志归到新架构。
3. 运行 `python -m unittest discover -s tests -v`、`python scripts/check_site.py`。
4. 核心算法修改再运行 `python -m aamappo smoke --out runs/smoke-review`。
5. 提交PR，附运行命令、环境版本、结果文件和局限。

请勿提交凭据、个人环境文件、大量重复训练产物或未获许可的出版PDF。
发现结果不一致时保留原始数据，记录差异与原因，不静默删除不利运行。
声明“复现”“优于”“安全”“定位精度”时必须指向相应证据及指标定义。
