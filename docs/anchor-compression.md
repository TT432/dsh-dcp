# dsh-dcp 锚点压缩（对齐 omp dcp 插件）—— 规格与设计

> 对齐基准：本机 `omp` 扩展 `~/.omp/agent/extensions/dcp`（anchor mode），
> 提交 `48d49ac`。本文件描述 dsh 侧的能力映射、行为契约与刻意差异。

## 1. 目标

把 omp dcp 的能力移植到 dsh-dcp 插件：

| omp dcp 能力 | dsh 实现 |
| --- | --- |
| `add_anchor({ anchor })` 具名锚点 | 同名工具；身份 = assistant 消息里的 `tool-call.id` |
| 虚拟锚点 `session_start`（原始 prompt 之后） | 同名虚拟锚点，解析为第一条 user 消息之后的表面位置；**不注入标记消息** |
| `compress({ startAnchor, summary, force })` | 同名工具；`force=true` 才能从 `session_start` 全量压缩 |
| 压缩区间：起点锚点（被消耗）→ compress 调用自身（保留） | 表面替换：吃掉锚点所在 assistant 消息到 compress 调用前一个表面节点；compress 调用与其结果保留 |
| 摘要注入在 compress 调用之前 | 合成 user 消息作为替换节点落在 compress 调用之前 |
| 会话历史不被改写（仅改请求副本） | **改为 append-only 的表面替换**：原始事件保留在日志中，仅从派生历史里遮蔽（dsh 架构约束，见 §3） |
| 持久化压缩块（custom entries v4） | 无需独立状态：替换节点本身持久在日志里，resume 自动生效 |
| omp 自身压缩后丢弃压缩块 | dsh 压缩（`/compact`）替换范围时自然遮蔽我们的节点；无需状态清理 |
| 超阈值 `compress` 提醒（`DCP_REMIND_THRESHOLD`） | 同名环境变量；`agent/pre-step` 注入提醒 user 消息（持久，非请求副本） |
| `DCP_DEBUG` 文件日志 | 同名环境变量；写 `$DSH_HOME/logs/dcp/`，并写 cordis logger |
| 错误抛给模型、状态不变 | 同：校验先于任何 append；错误文本列出可用锚点 |
| 无配置文件、无 slash 命令 | 同：仅环境变量；不注册命令 |

`dcp_compress`（LLM-DCP 论文的 prompt 压缩工具）是既有独立能力，保留不动，
**不属于**本次 omp 对齐范围。

## 2. 术语

- **表面（surface）**：dsh 从会话日志派生出的、模型可见的有序消息节点序列
  （`session.surface.nodes`，每个节点一个 `SessionSeq`）。
- **压缩节点（compression node）**：`compress` 追加的合成 `user/message` 替换节点，
  其 content 为 `[Compressed: <anchor> → here]\n<summary>`，source 为
  `{ kind: 'dcp-compress', anchor, startToolCallId? }`。
- **compress 调用节点**：包含本次 `compress` 工具调用的 assistant 消息节点。
  它是压缩区间的隐式终点，且**保留**在压缩后的上下文里。
- **影子计价（shadow price）**：dsh token-meter 的公开协议——表面 `replace`
  之前紧邻的 `compaction/prune` 事件记录被遮蔽区间的启发式 token 数，
  使纯消费者（上下文占用投影）无需保留逐节点价格即可扣减。

## 3. dsh 架构约束与方案选择（为什么不是「只改请求副本」）

dsh 的 loop 请求是**会话日志的纯函数**：`agent-loop` 在
`buildRequest()` 里 `deepFreeze(deriveMessages())` 后用
`markAgentLoopRequest(Object.freeze({...}))` 发出；`llm/stream` 事件明确写
「listeners read it, never rewrite it」，`agent/request` 明写「cannot mutate
messages」。因此 omp 的「请求副本变换」在 dsh 没有受支持的 seam。

dsh 的官方等价机制是 **surface replacement**（compaction 用的同一机制）：
日志只追加，被替换区间在派生历史中被遮蔽。所以 dsh-dcp 的压缩是**持久的**，
不是请求级的；它仍然满足「原始日志字节不变」——被遮蔽的原始事件永远留在日志里，
只是不再发给模型。

为保持 UI 上下文占用正确，替换前必须紧邻追加 `compaction/prune`
影子计价事件（这是 token-meter 侧唯一被识别的、面向模型无关替换的计价事件；
`compaction/summary` 受 compaction 括号约束，不可借用）。该事件由
`ctx.tokenMeter.estimateMessage()` 计价，保证与 token-meter 估算器一致。

## 4. 功能需求

### FR-1 `add_anchor({ anchor })`

- 工具参数：`anchor: string`（必填）。
- 校验（失败抛错、无任何 append）：
  - trim 后非空、≤ 64 字符、不含空白；
  - `session_start` 为保留名；
  - 名字在**整个会话**内唯一：扫描日志中所有 `add_anchor` 调用（含已被
    压缩消耗的），排除本次调用自身的 `toolCallId`。
- 历史扫描与工具校验共用同一名字规范：非法/失败的 `add_anchor` 不会成为
  可用锚点，也不能作为压缩起点或被提醒列出。
- 成功：返回确认文本，提示后续以 `startAnchor` 引用；锚点身份是
  provider 分配的 `tool-call.id`（全局唯一、持久）。
- 必须由模型直接调用；`run_code` 子调用没有模型可见的 tool-call 节点，
  直接报错，避免「返回成功但实际没有锚点」。
- 副作用：仅 assistant 消息里已经存在的 tool-call 本身（工具不再写任何状态）。

### FR-2 虚拟锚点 `session_start`

- 解析规则：第一条 `role === 'user'` 且 `source.kind === 'user'` 的表面节点
  （原始 prompt）；若无此类节点则退化为第一条 user 节点；再无则不可用。
- 压缩起点 = 该节点之后第一个表面节点（原始 prompt 保留）。
- **不注入** `[anchor: session_start]` 标记消息（dsh 没有请求级注入 seam；
  改由工具描述与错误文本说明该保留名）。这是与 omp 的刻意差异。
- 从 `session_start` 压缩必须显式 `force: true`（防误删工作上下文/失忆）。

### FR-3 `compress({ startAnchor, summary, force? })`

参数校验：`startAnchor` 非空字符串；`summary` 非空字符串；
`startAnchor === 'session_start'` 时 `force === true`。

定位（全部基于当前表面）：

1. 由 `exec.callId` 找到包含该 tool-call 的 assistant 表面节点
   （`compressIdx`）。找不到 → 报错「必须由模型直接调用，不能经
   `run_code` 子调用」。
2. 起点：
   - `session_start`：`firstUserIdx + 1`；
   - 真实锚点：该锚点 `add_anchor` tool-call 所在 assistant 节点
     （`anchorIdx`），**起点锚点被消耗**。
3. 终点：`compressIdx - 1`。
4. 若 `startIdx > endIdx` → 报错「区间内没有可压缩的消息」。

校验（失败不改状态）：

- 锚点不存在：若是已被先前压缩消耗的锚点，给出专门错误
  「anchor "X" was consumed by an earlier compression」；否则
  「unknown startAnchor ... Available anchors: ...」（列出存活锚点 + `session_start`）。
- 区间为表面上的连续节点序列，天然满足「不切断 tool-call/tool-result 配对」：
  压缩区间从 assistant 节点（或第一条 user 之后）到 compress 调用前的最后一个
  节点，区间内 assistant 消息的 tool-result 都由 loop 按模型顺序追加在同一区间内。
- 若区间内包含非空 `system` 节点（in-history 路由的提示词更新），**拒绝压缩**并
  提示在该更新之后重新打锚点；提示词不是可压缩的对话内容，不能因压缩而退化为
  旧提示词。
- 幂等/重入：两个 compress 调用并行执行时，工具默认是 `exclusive` 调度
  （`defineTool` 未声明 `isConcurrencySafe` 即独占），不会交错。

提交（无 await，同步、相邻）：

1. `session.append('compaction/prune', { shadowedRange: {start, end},
   shadowedSeqs, shadowedTokenCount })`；
2. `session.append('user/message', summaryMessage, { surfaceOp: {op:'replace',
   startSeq: start, endSeq: end}, sourceEventSeqs: shadowedSeqs })`。

结果：

- `shadowedSeqs`：被遮蔽的完整表面节点序列（顺序、非空），**包含不产生派生消息的空
  system/developer/assistant 节点**；`shadowedMessages` 只统计其中有派生消息的条目；
- `shadowedTokenCount`：`Σ ctx.tokenMeter.estimateMessage(node.message)`（空节点自然为 0）；
- 返回 `{ anchor, shadowedMessages, absorbed, shadowedTokens, freedTokens }`，
  `absorbed` = 区间内 source.kind === 'dcp-compress' 的节点数（被吸收的先前压缩），
  `freedTokens = max(0, shadowedTokens − estimateMessage(摘要节点))`（净释放估算）。

### FR-4 压缩语义（与 omp 对齐）

- 起点锚点的 assistant 消息、其后的 tool-result、以及区间内全部消息被遮蔽；
- compress 调用所在 assistant 消息与其 tool-result **保留**（调用是区间终点，
  确认消息是「压缩发生过」的可见记录）；
- 摘要节点落在 compress 调用之前，占据被遮蔽区间的位置；
- 链式：新起点在旧摘要节点之后 → 旧摘要保留，摘要按顺序累积；
- 吸收：新起点在旧摘要节点之前（且区间跨过它）→ 旧摘要被遮蔽，其内容由新的
  `summary` 承接（与 omp 相同，模型负责把要点写进新摘要）；
- 不修改任何已存在的事件；不删除日志。

### FR-5 超阈值提醒

环境变量 `DCP_REMIND_THRESHOLD`：

- 缺省 `0.6`；
- `(0,1)`：上下文窗口的比例；
- `>= 1`：绝对 token 数；
- `<= 0` / 非法：`<=0` 关闭，非法按缺省。

预算 = `floor(contextWindow * ratio)` 或绝对数；`contextWindow` 取
`session.requestContext()?.contextWindow`（最近一次已记录路由）。未知窗口时
提醒不生效（记一次 debug 日志）。

估算：优先 `ctx.tokenMeter.measure(session).totalTokens`；服务缺席时退化为
`Σ 文本字符 / 4 + 每条消息固定开销` 的启发式。再把本 step 尚未写入日志、即将进入
请求的 `decision.messages` 按同一启发式计入，避免阈值附近晚一拍。

触发：`agent/pre-step` 瀑布——先 `await next()`（保留运行时上下文注入），
在 `decision.kind === 'enter'` 且未取消、且估算 ≥ 预算时，把提醒
`UserMessage` 追加到 `decision.messages`。同一会话同一 turn 最多一次。
另注册 `agent/turn-stopping` 只读标记：若该 turn 正在收尾且没有新消息，
不注入提醒、不多驱动一次模型；标记只消费一次，不妨碍 steering 后的继续 step。

提醒文本（列出存活锚点）：

```
[dcp:context] Context is at ~N tokens (P% of the W-token window). Compress NOW:
call compress with startAnchor one of "a1", "a2", "session_start" and a complete
technical summary of everything since that anchor. Finish the compress before
starting any new work.
```

- source：`{ kind: 'dcp-reminder', contextWindow?: number, tokens: number }`，
  其中 `tokens` 是触发时的估算值（含本 step 新消息）。
- 与 omp 差异：omp 的提醒只存在于请求副本；dsh 的提醒是**普通 user/message 事件**，
  会持久化。因此加了「每 turn 最多一次」的节流。

### FR-6 调试日志

`DCP_DEBUG` 为真时：

- `$DSH_HOME/logs/dcp/dcp.log`：`[时间] LEVEL message {...}`；
- `$DSH_HOME/logs/dcp/context-<sessionId>.jsonl`：compress 后的派生消息快照；
- `$DSH_HOME` 取环境变量，缺省 `~/.dsh`。
- 任何日志失败都不得影响会话。

同时通过 `ctx.logger('dsh-dcp')` 发关键事件（加载、压缩、提醒）。

## 5. 非功能需求

- **NFR-1 日志不变量**：只 append；不删除、不改写既有事件；不修改
  `session.deriveMessages()` 返回的共享冻结消息。
- **NFR-2 原子性**：所有校验在任何 append 之前完成。若替换被拒，已追加的
  `compaction/prune` 是一个孤立的影子计价记录——下一次事件会使其过期，
  不影响重建（INV-4 的弱化说明）。
- **NFR-3 上下文占用投影不漂移**：影子计价使 token-meter 的
  `contextPressure.projectedTokens` 扣减被遮蔽区间、加上摘要节点；
  净变化为 `shadowedTokens − summaryTokens`，不要求摘要必然更短。
- **NFR-4 与 compaction 共存**：不注册 compaction 事件括号；`/compact` 或其他
  压缩遮蔽本插件的摘要节点时，不需要任何清理逻辑（无独立状态）。
- **NFR-5 测试**：`node --test test/*.mjs` 全绿；既有 LLM-DCP 测试不回归。

## 6. 异常与副作用

| 场景 | 行为 |
| --- | --- |
| 名字重复（含已消耗锚点） | 抛错，无 append |
| 名字为空/含空白/>64/`session_start` | 抛错，无 append |
| 未知 startAnchor | 抛错，列出存活锚点 |
| startAnchor 已被消耗 | 抛错，说明被哪次压缩消耗（给出名字即可） |
| `session_start` 缺 `force` | 抛错，说明失忆风险与摘要要求 |
| 区间为空 | 抛错，无 append |
| 区间跨过非空 system 提示词更新 | 抛错，提示在该更新后重新打锚点 |
| add_anchor / compress 由 `run_code` 子调用触发 | 抛错，要求直接调用（add_anchor 防止假成功） |
| tokenMeter 服务缺席 | 抛错（fail closed，避免占用投影漂移） |
| 提醒阈值非法/窗口未知 | 关闭提醒（不报错，debug 日志） |

副作用：每次成功 compress 追加 2 个事件（1 个影子计价 + 1 个替换）；
每次提醒追加 1 个 user/message。

## 7. 不变量

- **INV-1** 存活压缩节点的 `shadowedSeqs` 两两不重叠（替换会遮蔽旧节点，
  后续替换只能整体包含或完全在其之后）。
- **INV-2** 任一存活压缩节点的 compress 调用与其 tool-result 永不被其他压缩遮蔽
  （新替换的终点固定在 compress 调用之前）。
- **INV-3** `shadowedSeqs` 是当前表面的连续区间，首尾一致，且包含区间内每个节点。
- **INV-4** 校验失败不产生任何会话事件。
- **INV-5** 压缩后原始日志的全部事件仍在（append-only）。

## 8. 与 omp 的差异清单（刻意）

1. 压缩是持久的表面替换，不是请求副本变换（dsh 无该 seam）。
2. 不注入 `session_start` 标记消息。
3. 锚点名唯一性是「整会话」而非「存活锚点」——与 omp 一致；差异在于 omp
   从持久 branch 扫描，dsh 从事件日志扫描。
4. 提醒会持久化，且每 turn 至多一次（omp 每请求一次、不持久化）。
5. 缺省提醒阈值 `0.6` 而非 `0.8`：dsh 默认启用自动 compaction，其压力阈值约为
   `0.8 × (window − 输出预留 − 64k headroom)`；`0.6` 让模型驱动压缩先于自动
   compaction。可通过环境变量覆盖。
6. 需要 `ctx.tokenMeter`（dsh-base 默认挂载）；缺席时 compress 明确拒绝。
7. 额外的 dsh 安全约束：in-history 的 system-prompt 更新不可被压缩区间遮蔽。

## 9. 验收

- 单元/集成测试覆盖：锚点扫描、add_anchor 校验、非法调用不可解析、
  `source.kind === 'user'` 的 prompt 优先级、session_start、链式累积、
  吸收、空区间、未知/已消耗锚点、force、system-prompt 更新保护、空表面节点、
  影子计价、提醒阈值与节流。
- 实机：插件在 desktop/web profile 加载后 `add_anchor` / `compress` 出现在工具表；
  子代理会话中模型调用 `compress` 后，日志中 `compaction/prune` 与替换事件相邻、
  `deriveMessages()` 只含摘要节点与 compress 调用/结果，确认返回报告释放的 token 数。
