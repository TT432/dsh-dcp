# dsh-dcp：DCP 论文的 dsh 插件（SOSP'25 并行 + LLM-DCP 压缩）

在 dsh（DeepSeek Harness）中以工具形式实现两篇缩写为 DCP 的论文：

1. **DCP: Dynamic Context Parallelism**（[arXiv:2510.10620](https://arxiv.org/abs/2510.10620)，SOSP 2025）——
   长上下文训练的**动态上下文并行规划核心**（块生成 → 超图划分放置 → division 调度 → 模拟执行）。
2. **LLM-DCP: Dynamic Compressing Prompts**（[arXiv:2504.11004](https://ar5iv.labs.arxiv.org/html/2504.11004)）——
   **prompt/上下文压缩**：把压缩建模为 MDP，用 LLM 充当 DCP-Agent 多步迭代压缩，保留关键信息。

## 工具

- **dcp_plan**：给定一批序列长度 + mask + 集群/模型参数，生成动态上下文并行规划
  （块统计、设备放置、通信量、division 调度、每设备执行计划）与 vs zigzag/dp 基线的模拟加速比。
- **dcp_benchmark**：按长尾序列长度分布抽样多批，汇总 vs 静态 CP/DP 的加速比与通信降低
  （对应 SOSP DCP 论文 §7 micro-benchmark 流程）。
- **dcp_compress**（LLM-DCP）：用 ctx.llm 的可配置 LLM 充当 DCP-Agent，按论文 MDP 框架
  **多步迭代压缩**长 prompt/上下文：逐步删除冗余、保留关键信息（实体/数字/标识符/指令要点），
  目标压缩率随轨迹收紧（HPC 课程思想）。适合在调用长上下文模型前压缩提示词/文档/会话上下文。

## 安装

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
node --test test\run-tests.mjs       # DCP 并行规划核心
node --test test\compress-tests.mjs  # LLM-DCP 压缩核心（mock LLM）
```

## 实现说明

### LLM-DCP 压缩（dcp_compress）
论文把 prompt 压缩建模为 MDP（状态=压缩中 prompt；动作=逐 token 保留/删除；奖励=压缩率+关键信息保留−输出分布差异−越界惩罚），
DCP-Agent 是逐 token 二分类策略网络（xlm-roberta-large + PPO + HPC 课程训练）。
本插件在 dsh 中的等价实现（**LLM-as-DCP-Agent**）：
- 用 ctx.llm 的可配置 LLM 充当策略网络，按 MDP 框架迭代多步（每步基于上一步的动态上下文）；
- 目标压缩率随轨迹逐步收紧（HPC 课程思想：从宽松到严格）；
- 关键信息保留：压缩指令显式要求保留实体/数字/标识符/指令 + 用户可指定 must-keep 要点；
- 每步校验（空输出/无进展/无减少）与提前终止（达标即停，避免过度压缩）。
- 论文"不依赖黑盒 LLM 训练"的目标在推理端由 LLM 直接承担；dsh 未暴露 tokenizer，
  token 数为字符近似估算（英文≈4 字符/token，CJK 加权），压缩率控制以字符比例为准。
- 官方仓库（Fhujinwu/DCP）在论文接收后才会更新代码，本实现基于论文 §IV 的方法重写。

### DCP 并行规划（dcp_plan / dcp_benchmark）
见 README 早期版本与 `_research\dcp-official\` 官方代码对照；核心算法、成本模型逐行对应官方
`dcp/core/*`，超图划分以纯 JS 求解器近似 KaHyPar/PaToH（详见模块内注释与已知局限）。

## 参考

- LLM-DCP：https://arxiv.org/abs/2504.11004
- DCP (SOSP'25)：https://arxiv.org/abs/2510.10620 · 官方代码 https://github.com/chenyu-jiang/dcp
- LLM-DCP 官方仓库（暂无代码）：https://github.com/Fhujinwu/DCP
