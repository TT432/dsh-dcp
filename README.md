# dsh-dcp：LLM-DCP（Dynamic Compressing Prompts）的 dsh 插件

以工具形式实现 **LLM-DCP**（[arXiv:2504.11004](https://ar5iv.labs.arxiv.org/html/2504.11004)）：
把 prompt/上下文压缩建模为 MDP，用 LLM 充当 DCP-Agent 多步迭代压缩，保留关键信息。

> 历史说明：早期版本曾包含 DCP (SOSP'25, arXiv:2510.10620) 的 dcp_plan / dcp_benchmark 工具，
> 现已移除；本插件当前只提供 dcp_compress（相关研究资料见 _research\dcp-official\）。

## 工具

- **dcp_compress**（LLM-DCP）：用 ctx.llm 的可配置 LLM 充当 DCP-Agent，按论文 MDP 框架
  **多步迭代压缩**长 prompt/上下文：逐步删除冗余、保留关键信息（实体/数字/标识符/指令要点），
  目标压缩率随轨迹收紧（HPC 课程思想）。适合在调用长上下文模型前压缩提示词/文档/会话上下文。

## 安装

要求 dsh **0.2.1-rc.2 或更高的 0.2.x**（`peerDependencies` 声明为
`@deepseek-ai/dsh-tools` / `@deepseek-ai/dsh-llm` `^0.2.1-rc.2`）。0.1.x 运行时会在启动期被
dsh 的版本门禁禁用，请升级 dsh 后再装本版本。

```powershell
# 0) 先在插件目录安装依赖（必须）：profile 通过 link: 引用插件，Node ESM 会
#    realpath 到本目录解析 @deepseek-ai/* 导入；没有本地 node_modules 会报
#    ERR_MODULE_NOT_FOUND: Cannot find package '@deepseek-ai/dsh-tools'
npm install

# 1) 安装到 web profile（profile 是 pnpm workspace 根，需 -w 标志）
dsh plugin --profile web add -w <path/to/dcp-plugin>

# 2) 在 ~\.dsh\profiles\web\cordis.patch.yml 追加：
# - insert:
#     - id: dsh-dcp
#       name: 'dsh-dcp'
```

新装依赖后需重启 web 服务；改 patch 会热重载。

## 测试

```powershell
# LLM-DCP 压缩核心 + dcp_compress 工具接线（mock LLM / 真实 BlockAssembler）
node --test test\compress-tests.mjs test\tool-tests.mjs
```

## 实现说明

论文把 prompt 压缩建模为 MDP（状态=压缩中 prompt；动作=逐 token 保留/删除；奖励=压缩率+关键信息保留−输出分布差异−越界惩罚），
DCP-Agent 是逐 token 二分类策略网络（xlm-roberta-large + PPO + HPC 课程训练）。
本插件在 dsh 中的等价实现（**LLM-as-DCP-Agent**）：
- 用 ctx.llm 的可配置 LLM 充当策略网络，按 MDP 框架迭代多步（每步基于上一步的动态上下文）；
- 目标压缩率随轨迹逐步收紧（HPC 课程思想：从宽松到严格）；
- 关键信息保留：压缩指令显式要求保留实体/数字/标识符/指令 + 用户可指定 must-keep 要点；
- 每步校验（空输出/无进展/无减少）与提前终止（达标即停，避免过度压缩）。
- 论文"不依赖黑盒 LLM 训练"的目标在推理端由 LLM 直接承担；dsh 未暴露 tokenizer，
  token 数为字符近似估算（英文≈4 字符/token，CJK 加权），压缩率控制以字符比例为准。
- 与 dsh 0.2.x 的 LLM 契约对齐：一次性压缩请求使用无身份的 `RequestUserInput`
  （`{role:'user', content}`，不伪造持久消息 id/source；`MessageSource` 没有 `plugin` 这个 kind）；
  不再设置 `purpose`（0.2.x 是 `compaction | session-title` 的封闭联合）；
  分片消费走真实的 `BlockAssembler`（`block-start` / `text-delta` / `block-end` / `usage` / `finish`）。
- 官方仓库（Fhujinwu/DCP）在论文接收后才会更新代码，本实现基于论文 §IV 的方法重写。

## 参考

- LLM-DCP：https://arxiv.org/abs/2504.11004 · 官方仓库（暂无代码）：https://github.com/Fhujinwu/DCP
- DCP (SOSP'25)：https://arxiv.org/abs/2510.10620（规划工具已移除，研究资料见 _research\dcp-official\）
