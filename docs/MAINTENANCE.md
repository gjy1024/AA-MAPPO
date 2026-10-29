# GitHub维护与发布

目标是在一个仓库维护代码、实验协议、项目网站和可追溯结果。外部服务账号、访问令牌和私人数据不进入仓库。

## 1. 三类版本

| 版本类型 | 保存内容 | 维护原则 |
| --- | --- | --- |
| 论文对应版本 | 原始完整代码、配置、权重、日志和图表映射 | 用明确标签或Release冻结，不用后续修正覆盖原始证据 |
| 维护版本 | 入口修复、实体注意力实现、测试、导出、文档与平台 | 每次变更记录影响；影响方法时重新训练 |
| 教学示例 | 规则控制轨迹、注意力数值示例、交互讲解 | 明确标识教学来源，不能混入正式实验对比 |

本地已恢复四组历史日志与权重。归档时保留到`artifacts/legacy/`并附来源清单；网站可使用规范化后的曲线数据。当前维护版交付使用`artifacts/maintenance-smoke-v2/`，上一份`artifacts/maintenance-smoke/`保留为注意力帧对齐改进前的验收快照，不作为当前网页数据。维护版smoke输出与illustrative教学数据分别保存、分别标识。具体文件名由导入和导出程序决定，不能仅靠目录名推断来源。

如果找不到某项原始实验材料，记录“暂不可获取”，不要以维护版重新生成的数据冒充原数据。

## 2. 网站与GitHub Pages

网站源文件位于 `site/`，采用静态HTML、CSS和JavaScript，便于与代码一起审阅和版本管理。网站不需要登录、数据库或线上训练服务。发布前检查站点所有数据来源标签与链接。

GitHub Pages通过[`pages.yml`](../.github/workflows/pages.yml)上传并部署`site/`，核心检查由[`ci.yml`](../.github/workflows/ci.yml)负责。首次配置在仓库`Settings → Pages → Build and deployment → Source`选择`GitHub Actions`，再推送相应变更或在Actions中手动运行`Research Studio Pages`。部署需要仓库权限，且以实际Actions成功记录和页面响应为准；仅有工作流文件或README链接，不代表页面已经上线。

修改展示数据后，应同步更新来源清单与验证记录。项目根目录README提供论文、平台、源码、复现指南和实验协议入口。对网页的修订也保留commit。

## 3. 建议的自动检查

- 解析配置与示例JSON，检查字段、形状和非有限值。
- 环境重置和步进，确认单位、边界与可重复性。
- Actor/Critic前向、损失与短训练链路。
- 确认注意力候选实体数大于1，检查权重归一化；当前为全部实体可见，未来引入缺失实体后再增加mask测试。
- 静态站点链接、JSON导入、轨迹回放及响应式布局。

自动检查通过证明的是被检查的功能，不代表已复现论文主要结论。完整训练和统计实验另存实际记录。

## 4. 发布一个可引用版本

发布前确认：

1. README命令在干净环境中运行通过。
2. 论文—代码—图表映射和配置完整。
3. 许可及第三方归属已核查，`CITATION.cff`与正式论文信息一致。
4. Release说明标出已验证功能、已知限制和结果覆盖范围。
5. 检查点和大文件不依赖会失效的私人临时链接。
6. 对外链接、网站数据及示例来源与Release一致。

Release应关联确定的commit。可使用诸如 `v0.1.0` 的版本号，但“论文对应”必须有原始证据支持，不能仅通过名称宣布。

## 5. 软件DOI与论文DOI

软件版本可以按需要使用Zenodo归档，取得持久标识，并在元数据中关联原论文DOI。该步骤需要仓库拥有者自行启用连接或授权，不在文档中声称已经完成。

论文DOI用于引用论文；软件版本DOI用于引用固定软件快照。不要为已经有DOI的同一论文文件随意制造第二个论文标识。软件归档本身不代表研究结论已经通过独立评审。

## 6. 许可、归属与学术引用

公开可查看的仓库不等于具备完整开源许可。本维护包已补充[`LICENSE`](../LICENSE)、[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)，并在`legacy/licenses/`保留所引用上游MIT声明；原代码头部归属继续保留。新增外部代码和素材时仍需逐项核查，不能仅靠README徽章推断其权限。

`CITATION.cff`可以同时描述软件，并将论文指定为首选引用。论文正文、软件、数据和网站素材的权限分别检查。正式排版PDF的公开分享按对应出版政策执行；仓库可以始终提供论文DOI链接。

## 7. Issue与变更记录

Issue建议分为安装、运行、结果偏差、文档、平台和研究建议。问题报告包含commit、环境、配置、命令与错误输出。

变更记录至少说明：改了什么、为什么改、怎样验证、是否改变模型或环境语义、是否需要重训、是否影响已有结果。修复Bug时保留必要历史，不静默重写旧结论。

## 8. 定期维护

不必为了提交频率而修改项目。每次依赖或接口变化时执行相关检查；每次正式实验后补齐配置和原始记录；每次发布时检查链接与来源。后续新论文应说明与原论文差异，并按目标出版物要求进行引用和披露。

官方参考：[GitHub Pages文档](https://docs.github.com/en/pages)、[GitHub引用文件](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)、[Zenodo软件归档](https://help.zenodo.org/docs/github/)、[IEEE会议论文分享](https://conferences.ieeeauthorcenter.ieee.org/get-published/post-your-paper/)。
