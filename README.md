# dsh-dcp：锚点上下文压缩 + LLM-DCP 提示词压缩

本插件包含两条独立能力线：

1. **锚点压缩**：对齐本机 omp 的 dcp 扩展（`~/.omp/agent/extensions/dcp`，anchor mode），
   由模型通过 `add_anchor` 与 `compress` 主动管理工作上下文。
2. **LLM-DCP**（[arXiv:2504.11004](https://arxiv.org/abs/2504.11004)）：既有的
   `dcp_compress` 工具，用可配置 LLM 充当 DCP-Agent 迭代压缩任意文本。
   它不是 omp 对齐项，行为保持不变。

锚点压缩的完整规格、前置/后置条件、不变量与异常行为见
[`docs/anchor-compression.md`](./docs/anchor-compression.md)。

## 工具

### `add_anchor({ anchor })`

在当前对话位置放置具名锚点；锚点身份是该工具调用的 provider tool-call id。

- 名字非空、不含空白、长度 ≤ 64；`session_start` 为保留名。
- 名字在**整个会话内唯一**（包括已被压缩消耗的锚点），重复会报错。
- `session_start` 是内置虚拟锚点，位于原始用户 prompt 之后；不需要标记消息。
- 与 `compress` 一样，`add_anchor` 也必须是模型直接发起的工具调用；
  `run_code` 子调用会报错，避免“返回成功但没有真正锚点”。

### `compress({ startAnchor, summary, force? })`

把从 `startAnchor` 到**本次 compress 调用之前**的上下文替换为模型自己写的技术摘要。

```text
prompt · work1 · add_anchor(a1) · work2 · work3 · compress(a1, summary)
```

压缩后模型看到：

```text
prompt · work1 · [Compressed: a1 → here] summary · compress(a1, summary) · 确认结果
```

- 起点锚点及其结果被消耗；compress 调用与确认结果保留，后续轮次知道压缩发生过。
- 摘要由**替换节点**携带，落在 compress 调用之前。
- 链式压缩的摘要按序累积；跨越旧压缩时会吸收旧摘要，模型应把其要点并入新摘要。
- `startAnchor: "session_start"` 必须同时传 `force: true`，防止误删工作上下文。
- 未知锚点、空区间、已消耗锚点、缺少 `force`、跨 system-prompt 更新、经
  `run_code` 子调用等情况都会报错，且不产生任何会话事件；错误会列出可用锚点。
- `run_code` 子调用不能压缩：区间终点必须是模型直接发起的 compress 调用。

**持久化语义**：dsh 的 loop 请求是会话日志的纯函数，不能像 omp 一样只改请求副本。
因此本插件使用 dsh 官方的 surface replacement 机制：只追加事件，不删除或改写原始日志；
被压缩区间仅从派生给模型的历史中遮蔽，resume 后仍保持压缩结果。
替换前紧邻追加 `compaction/prune` 影子计价事件，保证上下文占用投影按
「被遮蔽区间 − 摘要节点」正确更新，不发生漂移。

### `dcp_compress`

既有 LLM-DCP 工具：多步迭代压缩 prompt/文档/会话文本，支持 `targetRatio`、`steps`、
`keyPoints` 与 provider/model 覆盖。与上面的锚点压缩不是同一能力。

## 配置（仅环境变量，无配置文件）

| 变量 | 含义 |
| --- | --- |
| `DCP_REMIND_THRESHOLD` | 超阈值提醒。缺省 `0.4`；`(0,1)` 为上下文窗口比例，`>=1` 为绝对 token 数，`<=0` 关闭。 |
| `DCP_DEBUG` | 非空且非 `0/false/off/no` 时，写 `$DSH_HOME/logs/dcp/dcp.log` 与 `context-<session>.jsonl`。 |

提醒在 `agent/pre-step` 注入，列出当前可用锚点并要求立即 `compress`。
与 omp 不同，dsh 的提醒是普通 `user/message` 事件，会持久化，因此**每个 turn 至多一次**。
缺省阈值取 `0.4` 而非 omp 的 `0.8`，是为了让模型驱动压缩先于 dsh 自动 compaction
（后者压力预算约为 `0.8 × (window − 输出预留 − 64k headroom)`）。

压缩需要 `ctx.tokenMeter`（dsh-base 默认挂载）计算影子价格；服务缺席时明确报错，
不会静默造成上下文占用漂移。

## 安装

要求 dsh **0.2.1-rc.2 或更高的 0.2.x**（peer 为 `@deepseek-ai/dsh-tools` /
`@deepseek-ai/dsh-llm` / `@deepseek-ai/dsh-session` / `@deepseek-ai/dsh-token-meter`
`^0.2.1-rc.2`；本机开发/测试对齐 **0.2.1-rc.3**）。

```powershell
# 0) 在插件目录安装依赖（link: 插件需要自己的 node_modules）
npm install

# 1) 安装到目标 profile（web/desktop 均按 workspace 根处理，需 -w）
dsh plugin --profile web add -w <path/to/dcp-plugin>
# desktop 也可：
dsh plugin --profile desktop add -w <path/to/dcp-plugin>

# 2) 在对应 profile 的 cordis.patch.yml 追加（已有则无需重复）：
# - insert:
#     - id: dsh-dcp
#       name: 'dsh-dcp'
```

新装依赖后需重启对应应用；修改 patch 会热加载。已链接的源码发生变更时，
本机 desktop 实测可热更新宿主侧工具。

## 测试

```powershell
node --test test\compress-tests.mjs test\tool-tests.mjs test\anchor-tests.mjs test\anchor-tool-tests.mjs test\remind-tests.mjs
```

覆盖：LLM-DCP 既有回归、锚点扫描与唯一性、`session_start`、区间替换、链式累积、
吸收、空区间/未知/已消耗锚点、system-prompt 更新保护、影子计价、提醒阈值与节流、
工具注册与参数校验。当前共 **54 项测试**。

实机验证（desktop profile，子代理独立会话）：模型依次调用 `add_anchor`、`read`、
`compress`，会话日志确认：

```text
seq 28  compaction/prune   shadowedRange 18–22, shadowedTokenCount 2268
seq 29  user/message       replace 18–22, [Compressed: e2e-anchor → here] ...
seq 30  tool/result        compress 确认：3 message(s) elided，并报告净释放 token 估算
```

返回的 `freedTokens` 按 `shadowedTokenCount − 摘要节点估算 token` 计算（下限为 0），
不是把被遮蔽区间的总量误报为净释放量。

原始 add_anchor/read 事件仍在日志中；派生上下文只保留摘要、compress 调用与确认结果。
同一子会话随后执行第二次压缩（`e2e-final-2`）时，首个 `[Compressed: e2e-final → here]`
摘要仍保留，返回 `4 message(s) elided, ~628 tokens freed`，验证链式累积。

## 实现结构

```text
lib/index.js             插件入口：注册 3 个工具与提醒监听
lib/anchor/anchors.js    表面节点/锚点扫描、名字校验、session_start
lib/anchor/compress.js   区间解析、校验、影子计价 + 替换提交
lib/anchor/prompts.js    工具描述、摘要与提醒文本
lib/anchor/remind.js     阈值、token 估算、agent/pre-step 提醒与 turn-stopping 收尾抑制
lib/anchor/logger.js     DCP_DEBUG 文件日志与 cordis logger
lib/tools/anchor.js      add_anchor / compress 工具接线
lib/dcp/compress.js      既有 LLM-DCP 核心
lib/tools/compress.js    既有 dcp_compress 工具
```

与 omp 的刻意差异（完整清单见规格文档）：压缩是持久的 append-only 表面替换；
不注入 `session_start` 标记消息；提醒会持久化且每 turn 至多一次；缺省提醒阈值为 `0.4`；
in-history system-prompt 更新不可被压缩区间遮蔽。

## 参考

- 对齐基准：`~/.omp/agent/extensions/dcp`（README、docs/design.md、src/*）。
- 规格：[`docs/anchor-compression.md`](./docs/anchor-compression.md)。
- LLM-DCP：https://arxiv.org/abs/2504.11004 · 官方仓库（暂无代码）：https://github.com/Fhujinwu/DCP
