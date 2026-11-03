# dsh-dcp 锚点压缩（对齐 omp dcp 插件）—— 规格与设计

> 对齐基准：本机 `omp` 扩展 `~/.omp/agent/extensions/dcp`（anchor mode），
> 基线 `48d49ac`，跟进 `c83eeaa` → `0583104`（证据正则放宽）→ `99a4f87`
> （摘要链路化、移除机械闸门与工具调用计数器）。
> 本文件描述 dsh 侧的能力映射、行为契约与刻意差异。

## 1. 目标

把 omp dcp 的能力移植到 dsh-dcp 插件：

| omp dcp 能力 | dsh 实现 |
| --- | --- |
| `add_anchor({ anchor })` 具名锚点 | 同名工具；身份 = assistant 消息里的 `tool-call.id` |
| 虚拟锚点 `session_start`（原始 prompt 之后） | 同名虚拟锚点，解析为第一条 user 消息之后的表面位置；**不注入标记消息** |
| `compress({ startAnchor, summary })` | 同名工具；**没有 `force`**（omp `99a4f87` 删除） |
| 压缩区间：起点锚点（被消耗）→ compress 调用自身（保留） | 表面替换：吃掉锚点所在 assistant 消息到 compress 调用前一个表面节点；compress 调用与其结果保留 |
| 摘要注入在 compress 调用之前 | 合成 user 消息作为替换节点落在 compress 调用之前，带 `(block N)` |
| 摘要槽位 + 证据定位 | 推荐结构；解析结果只产出 **advisory**，从不拒绝压缩 |
| 吸收旧块 → 链（`parentIds` / `supersededBy`） | 旧块事件保留；新块 `source.parentIds` 记录直接父块，摘要下渲染一行索引 |
| `expand({ block })` 取回被吸收块全文 | 同名工具；从日志扫描块号，返回摘要全文 + 更深一层索引 |
| 压缩块持久化（omp custom entries v5） | 无需独立状态：压缩节点本身就是日志事件；块号/父子关系记录在消息 `source` |
| omp 自身压缩后丢弃压缩块 | dsh 压缩（`/compact`）替换范围时自然遮蔽本插件的节点；日志保留，`expand` 仍可取回，无需状态清理 |
| 两类提醒：token 压力 / 工作单元边界 | 同名两类；`agent/pre-step` 注入提醒 user 消息（持久，非请求副本）；无工具调用计数器 |
| `[dcp:unit]` 按 todo 结果 toolCallId 去重 | 从日志里最近一条 unit 提醒的 `source.unitCallId` 推导（omp 写 `dcp-state`） |
| `DCP_DEBUG` 文件日志 | 同名环境变量；写 `$DSH_HOME/logs/dcp/`，并写 cordis logger |
| 错误抛给模型、状态不变 | 同：硬错误先于任何 append；错误文本列出可用锚点/块号 |
| 无配置文件、无 slash 命令 | 同：仅环境变量；不注册命令 |

`dcp_compress`（LLM-DCP 论文的 prompt 压缩工具）是既有独立能力，保留不动，
**不属于**本次 omp 对齐范围。

## 2. 术语

- **表面（surface）**：dsh 从会话日志派生出的、模型可见的有序消息节点序列
  （`session.surface.nodes`，每个节点一个 `SessionSeq`）。
- **压缩节点（compression node）**：`compress` 追加的合成 `user/message` 替换节点，
  其 content 为
  `[Compressed: <anchor> → here] (block <id>)\n<summary>`（吸收过旧块时再接索引尾注），
  source 为 `{ kind: 'dcp-compress', anchor, blockId, parentIds, startToolCallId? }`。
- **compress 调用节点**：包含本次 `compress` 工具调用的 assistant 消息节点。
  它是压缩区间的隐式终点，且**保留**在压缩后的上下文里。
- **压缩块（block）**：一条持久的压缩记录。块号 `blockId` 存在 source 里；
  链字段 `parentIds` 记录被它吸收的直接父块（新的在前）。
- **链（chain）**：压缩块之间的父子关系。被吸收的块不再驱动表面替换（区间已被
  子块覆盖），但文本永不删除，按索引行回溯。
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
只是不再发给模型。这也让链成为自然实现：被吸收块的压缩事件本来就在日志里。

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
- 成功：返回确认文本，说明该锚点标记**下一段**的起点（先打锚点、继续工作，
  结束时从更早的锚点压缩，把新锚点留给下一段）。
- 必须由模型直接调用；`run_code` 子调用没有模型可见的 tool-call 节点，
  直接报错，避免「返回成功但实际没有锚点」。
- 副作用：仅 assistant 消息里已经存在的 tool-call 本身（工具不再写任何状态）。

### FR-2 虚拟锚点 `session_start`

- 解析规则：第一条 `role === 'user'` 且 `source.kind === 'user'` 的表面节点
  （原始 prompt）；若无此类节点则退化为第一条 user 节点；再无则不可用。
- 压缩起点 = 该节点之后第一个表面节点（原始 prompt 保留）。
- **不注入** `[anchor: session_start]` 标记消息（dsh 没有请求级注入 seam；
  改由工具描述与错误文本说明该保留名）。
- 从 `session_start` 压缩**不需要 `force`**（omp `99a4f87` 删除了该闸门）：
  吸收的块保留在链中，全量压缩不再不可恢复。工具描述与提醒仍说明它是最有损的
  调用，建议优先用命名锚点。

### FR-3 `compress({ startAnchor, summary })`

参数校验（全部先于任何状态变更）：

- `startAnchor` 非空字符串；`summary` 非空字符串。
- `summary` 只做 **advisory 解析**（FR-3.1），不因形状拒绝。
- 参数里**没有 `force`**；多余参数不会被读取。

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

硬校验（失败不改状态）：

- 锚点不存在：若是已被先前压缩消耗的锚点，给出专门错误
  「anchor "X" was consumed by an earlier compression」；否则
  「unknown startAnchor ... Available anchors: ...」（列出存活锚点 + `session_start`）。
- 区间为表面上的连续节点序列，天然满足「不切断 tool-call/tool-result 配对」。
- 若区间内包含非空 `system` 节点（in-history 路由的提示词更新），**拒绝压缩**并
  提示在该更新之后重新打锚点；提示词不是可压缩的对话内容。
- 若区间内含旧压缩节点，执行**链式吸收**（FR-3.2），不校验新摘要是否继承旧定位。
- 幂等/重入：两个 compress 调用并行执行时，工具默认是 `exclusive` 调度
  （`defineTool` 未声明 `isConcurrencySafe` 即独占），不会交错。

#### FR-3.1 摘要形状（对齐 omp `0583104` / `99a4f87`）

推荐的六个槽位，顺序固定、各出现一次：

```text
## 当前任务        - 正在做的事 证据: <定位>
## 已完成          - 动作/结论 证据: <定位>
## 未完成          - 下一步具体动作 证据: <定位>
## 约束            - 用户要求/不变量 证据: <定位>
## 文件与命令      - 关键路径/命令 证据: <定位>
## 在跑作业与句柄  - 作业/子代理/端口 证据: <定位>
```

解析器**统计偏差并继续**，从不抛错：

- 证据正则 `/^(.*)\s*证据\s*[:：]\s*(\S.*)$/s`：任意位置切分（greedy 取最后一处
  「证据」）、冒号前后可有任意空白、支持全角冒号。`。证据:`（中文标点后无空格）
  是合法写法。
- 偏差类型：没有槽位标题、未知槽位、重复槽位、缺少槽位、`当前任务` 全为 `- 无`、
  槽位下内容不以 `- ` 开头、事实没有 `证据:`。
- 偏差作为 `摘要形状提示（已压缩，未拒绝）` 随 compress 成功结果返回；
  压缩照常提交。
- 完全自由格式的摘要也保留为 `slot: null` 的条目，供 gist/索引使用。
- 存在偏差**不**阻止任何状态变更（INV-2）。

#### FR-3.2 链（对齐 omp `99a4f87`）

- 新压缩区间遮蔽旧压缩节点时，这些节点成为**直接父块**：
  - `parentIds` = 被吸收块的 `blockId`，新的在前；
  - 旧块文本不删除，仍可从日志读取；
  - 新摘要节点文本末尾追加索引尾注：

    ```text
    已并入本块的旧压缩块（索引，全文未丢失）:
    - 块 1「a1」：<当前任务 gist> — 全文: expand({block: 1})
    ```

  - 索引列出**传递祖先**（父块及其父块……），每块一行；gist 取块摘要的
    `当前任务` 第一条事实，其次第一条事实，最长 160 字符。
- 被吸收块不参与表面替换（其区间已由子块覆盖），但 `expand` 永远可返回其全文。
- 不再有「基线继承」校验：旧文本在链里，模型漏写的内容仍可取回。
  因此吸收不会因摘要内容被拒绝。

#### FR-3.3 块号

- 新块号 = 日志中已有块号最大值 + 1。
- 新节点把 `blockId` 写入消息 `source`。
- 升级前产生的块没有 `blockId`：按日志出现顺序补 1..k（日志 append-only，
  补号跨 resume 稳定）；显式块号优先且不与补号冲突。
- 块号只在会话内唯一，不跨会话。

提交（无 await，同步、相邻）：

1. `session.append('compaction/prune', { shadowedRange: {start, end},
   shadowedSeqs, shadowedTokenCount })`；
2. `session.append('user/message', summaryMessage, { surfaceOp: {op:'replace',
   startSeq: start, endSeq: end}, sourceEventSeqs: shadowedSeqs })`。

结果：

- `shadowedSeqs`：被遮蔽的完整表面节点序列（顺序、非空），**包含不产生派生消息的空
  system/developer/assistant 节点**；`shadowedMessages` 只统计其中有派生消息的条目；
- `shadowedTokenCount`：`Σ ctx.tokenMeter.estimateMessage(node.message)`（空节点自然为 0）；
- 返回 `{ anchor, blockId, shadowedMessages, absorbed, absorbedBlockIds, chain,
  shadowedTokens, freedTokens, problems }`；
  `absorbed` / `absorbedBlockIds` 是直接父块数/块号，`chain` 是传递祖先索引，
  `freedTokens = max(0, shadowedTokens − estimateMessage(摘要节点))`。

### FR-4 压缩语义（与 omp 对齐）

- 起点锚点的 assistant 消息、其后的 tool-result、以及区间内全部消息被遮蔽；
- compress 调用所在 assistant 消息与其 tool-result **保留**（调用是区间终点，
  确认消息是「压缩发生过」的可见记录）；
- 摘要节点落在 compress 调用之前，占据被遮蔽区间的位置；
- 链式：新起点在旧摘要节点之后 → 旧摘要保留，摘要按顺序累积；
- 吸收：新起点在旧摘要节点之前（且区间跨过它）→ 旧摘要节点被遮蔽，但块文本
  保留在日志里，由新块的索引行指向；`expand` 可取回；
- 不修改任何已存在的事件；不删除日志。

### FR-5 `expand({ block })`

- 参数：`block: integer`（块号）。
- 校验：块号必须是安全整数；未知块号抛错并列出已知块号。
- 数据来源：扫描整个会话日志里的压缩节点（含已被表面替换遮蔽的），不依赖表面。
- 返回文本：

  ```text
  [block 1「a1」 — 全文]
  <该块模型摘要全文>

  更深一层（索引）:
  - 块 0「a0」：<gist> — 全文: expand({block: 0})
  ```

  没有更深一层时省略索引段。
- `expand` 只返回**压缩块（模型写的摘要）**；被压缩区间内的原始消息/工具返回不会
  因此恢复——它们仍在会话日志里，但不进入派生给模型的上下文。
- `expand` 不改变任何状态；可在同一 turn 内多次调用。

### FR-6 压缩提醒（两类，对齐 omp `5954be4` / `3caea33` / `4d7c8ce` / `c83eeaa` / `99a4f87`）

环境变量：

- `DCP_REMIND_THRESHOLD`：token 压力阈值。缺省 `0.4`；`(0,1)` 为窗口比例；
  `>= 1` 为绝对 token 数；`<= 0` / 非法：`<=0` 关闭，非法按缺省。

两类提醒，按优先级 `[dcp:context]` > `[dcp:unit]`，每个 turn 至多一条：

1. **`[dcp:context]`**：预算 = `floor(contextWindow * ratio)` 或绝对数；
   `contextWindow` 取 `session.requestContext()?.contextWindow`（最近一次已记录路由）。
   未知窗口时 token 触发不生效（记一次 debug 日志）。估算优先
   `ctx.tokenMeter.measure(session).totalTokens`；服务缺席时退化为本地启发式：
   每条消息 8 token 固定开销 + 文本按 **CJK 约 2 字符/token、ASCII 约 4 字符/token**
   分档（单个 4 字符/token 常数会把中文会话低估约 40%）。
   再把本 step 尚未写入日志、即将进入请求的 `decision.messages` 按同一启发式计入，
   避免阈值附近晚一拍。
2. **`[dcp:unit]`**：最新 `todo_write` 的 tool result 报告完成时触发。
   识别措辞：dsh 的 `Updated todo list: 0 pending, 0 in progress, N completed.`
   （N>0），以及 omp 宿主的 `All items complete` / `Remaining items (0)` /
   `Remaining items: none.` / `0 open`。`isError`、无法解析、最新 todo 仍在进行
   都不触发。按结果的 `toolCallId` 去重：同一个完成结果只提醒一次，模型再次调用
   `todo_write` 并完成后才重新提醒（dsh 从日志里最近一条 unit 提醒的
   `source.unitCallId` 读取水位）。

**没有工具调用计数器**：omp `99a4f87` 删除了 `[dcp:budget]` 与 `toolCallMark`。
调用数不是上下文代价；旧规则按 40 次调用提醒，会在上下文远未接近窗口时逼模型压缩，
随后不得不重读刚读过的文件。token 压力是唯一可度量的成本。

触发方式：`agent/pre-step` 瀑布——先 `await next()`（保留运行时上下文注入），
在 `decision.kind === 'enter'` 且未取消时检查；命中后把提醒 `UserMessage`
追加到 `decision.messages`。同一会话同一 turn 最多一条。

**10% 档位（token 提醒）**：`[dcp:context]` 只在上下文每增长 10% 时提醒一次
（`tokenBracket = floor(tokens / (window ?? budget) × 10)`）。档位锚点是表面最近一条
token 提醒的估算值（提醒事件持久在日志里，`source.tokens` 可读回），因此不新增持久状态；
该提醒之后的 `compress` 调用或更低的新估算值视为重置。档位抑制只影响 token 提醒本身，
被抑制的 turn 仍可命中 unit 触发。
另注册 `agent/turn-stopping` 只读标记：若该 turn 正在收尾且没有新消息，
不注入提醒、不多驱动一次模型；标记只消费一次，不妨碍 steering 后的继续 step。

提醒文本（列出存活锚点；说明全量压缩已允许但仍是最高损耗选项）：

```text
[dcp:context] Context is at ~N tokens (P% of the W-token window). Compress NOW:
call compress with startAnchor one of "a1", "a2" and a complete technical
summary of everything since that anchor. Finish the compress before starting
any new work. Full-session compression ("session_start") is allowed and nothing
is destroyed — absorbed blocks stay recallable with expand — but it is the most
lossy call available: ...
```

- source：`{ kind: 'dcp-reminder', trigger?: 'unit', unitCallId?, contextWindow?,
  tokens? }`；token 提醒的 source 不带 `trigger`（与既有行为兼容）。
- 与 omp 差异：omp 的提醒只存在于请求副本；dsh 的提醒是**普通 user/message 事件**，
  会持久化。因此加了「每 turn 最多一次」的节流（比 omp 的每请求一次更严格）。

### FR-7 调试日志

`DCP_DEBUG` 为真时：

- `$DSH_HOME/logs/dcp/dcp.log`：`[时间] LEVEL message {...}`；
- `$DSH_HOME/logs/dcp/context-<sessionId>.jsonl`：compress 后的派生消息快照；
- `$DSH_HOME` 取环境变量，缺省 `~/.dsh`。
- 任何日志失败都不得影响会话。

同时通过 `ctx.logger('dsh-dcp')` 发关键事件（加载、压缩、展开、提醒）。

## 5. 非功能需求

- **NFR-1 日志不变量**：只 append；不删除、不改写既有事件；不修改
  `session.deriveMessages()` 返回的共享冻结消息。
- **NFR-2 原子性**：所有硬校验在任何 append 之前完成。若替换被拒，已追加的
  `compaction/prune` 是一个孤立的影子计价记录——下一次事件会使其过期，
  不影响重建（INV-4 的弱化说明）。
- **NFR-3 上下文占用投影不漂移**：影子计价使 token-meter 的
  `contextPressure.projectedTokens` 扣减被遮蔽区间、加上摘要节点；
  净变化为 `shadowedTokens − summaryTokens`，不要求摘要必然更短。
- **NFR-4 与 compaction 共存**：不注册 compaction 事件括号；`/compact` 或其他
  压缩遮蔽本插件的摘要节点时，不需要任何清理逻辑（无独立状态）。被遮蔽块的
  文本仍在日志里，`expand` 仍可取回。
- **NFR-5 测试**：`node --test test/*.mjs` 全绿（沙箱内需
  `--test-isolation=none` 避免测试运行器 spawn 子进程 EPERM）；
  既有 LLM-DCP 测试不回归。

## 6. 异常与副作用

| 场景 | 行为 |
| --- | --- |
| 名字重复（含已消耗锚点） | 抛错，无 append |
| 名字为空/含空白/>64/`session_start` | 抛错，无 append |
| 未知 startAnchor | 抛错，列出存活锚点 |
| startAnchor 已被消耗 | 抛错，说明被先前压缩消耗 |
| 摘要形状偏差 / 缺少证据 / 自由格式 | **不抛错**；压缩照常，偏差随结果返回 |
| 吸收旧块但新摘要没写旧定位 | **不抛错**；旧块全文由链保留，`expand` 可取回 |
| 区间为空 | 抛错，无 append |
| 区间跨过非空 system 提示词更新 | 抛错，提示在该更新后重新打锚点 |
| add_anchor / compress 由 `run_code` 子调用触发 | 抛错，要求直接调用 |
| 未知压缩块号 | 抛错，列出已知块号 |
| tokenMeter 服务缺席 | 抛错（fail closed，避免占用投影漂移） |
| 提醒阈值非法/窗口未知 | 关闭 token 提醒（不报错，debug 日志）；unit 不受影响 |

副作用：每次成功 compress 追加 2 个事件（1 个影子计价 + 1 个替换）；
每次提醒追加 1 个 user/message；`expand` 无副作用。

## 7. 不变量

- **INV-1** 存活压缩节点的 `shadowedSeqs` 两两不重叠（替换会遮蔽旧节点，
  后续替换只能整体包含或完全在其之后）。
- **INV-2** 压缩内容不构成拒绝理由：摘要形状、证据缺失、旧定位未重写都只产生
  advisory；只有「调用无法有意义」的错误（未知锚点/块号、空区间）才失败。
- **INV-3** `shadowedSeqs` 是当前表面的连续区间，首尾一致，且包含区间内每个节点。
- **INV-4** 硬校验失败不产生任何会话事件。
- **INV-5** 压缩后原始日志的全部事件仍在（append-only）；被吸收块的文本永远可
  通过 `expand` 取回。
- **INV-6** 任一存活压缩节点的 compress 调用与其 tool-result 永不被其他压缩遮蔽
  （新替换的终点固定在 compress 调用之前）。

## 8. 与 omp 的差异清单（刻意）

1. 压缩是持久的表面替换，不是请求副本变换（dsh 无该 seam）。
2. 不注入 `session_start` 标记消息。
3. 锚点名唯一性是「整会话」而非「存活锚点」——与 omp 一致；差异在于 omp
   从持久 branch 扫描，dsh 从事件日志扫描。
4. 提醒会持久化，且每 turn 至多一条（omp 每请求一条、不持久化）。
5. 缺省提醒阈值 `0.4` 而非 `0.8`：dsh 默认启用自动 compaction，其压力阈值约为
   `0.8 × (window − 输出预留 − 64k headroom)`；`0.4` 让模型驱动压缩先于自动
   compaction。可通过环境变量覆盖。
6. 需要 `ctx.tokenMeter`（dsh-base 默认挂载）；缺席时 compress 明确拒绝。
7. 额外的 dsh 安全约束：in-history 的 system-prompt 更新不可被压缩区间遮蔽。
8. 没有 omp 的 `dcp-state` 持久状态与 v4→v5 迁移：压缩块就是日志事件，
   `blockId`/`parentIds` 存在消息 source 里；旧块按日志顺序补号。
9. token 提醒的 10% 档位与 `[dcp:unit]` 去重水位都从日志推导，不写持久状态
   （omp 把 `remindLevel` / `unitRemindId` 写进 `dcp-state`）。
10. dsh 无 stale usage 问题：`estimateContextTokens` 优先使用
    `tokenMeter.measure(session)`，它按当前表面（含影子计价替换）重新计价，
    不会锚定压缩前的旧 usage。
11. dsh 拒绝 `run_code` 子调用压缩/锚点，compress 调用必然在表面上，
    因此不需要 omp 的 eval-kernel 绕过路径与相关水位。

## 9. 验收

- 单元/集成测试覆盖：锚点扫描、add_anchor 校验、非法调用不可解析、
  `source.kind === 'user'` 的 prompt 优先级、session_start（无 force）、链式累积、
  吸收与索引行、`expand` 全文/更深一层/未知块号、旧块补号、空区间、
  未知/已消耗锚点、system-prompt 更新保护、空表面节点、影子计价、
  advisory 摘要解析与放宽的证据正则、两类提醒的触发/优先级/节流、unit 去重、
  token 提醒的 10% 档位与重置、CJK 估算、工具注册与参数校验。
- 实机：插件在 desktop/web profile 加载后 `add_anchor` / `compress` / `expand`
  出现在工具表；子代理会话中模型调用 `compress` 后，日志中 `compaction/prune`
  与替换事件相邻、`deriveMessages()` 只含摘要节点与 compress 调用/结果；
  跨过旧块再次压缩后，新摘要下出现旧块索引行，`expand` 能取回旧摘要全文。
- 2026-11-03 实机 e2e（desktop profile，HMR 重载后，子代理独立会话）：
  `add_anchor e2e-a1 → compress(e2e-a1)` 得 block 1；`compress(session_start)`
  吸收 block 1 得 block 2，摘要下出现 `块 1「e2e-a1」… expand({block: 1})` 索引行；
  `expand({block:1})` 返回块 1 摘要全文。会话日志中 `compaction/prune` 与替换事件
  相邻，`deriveMessages()` 只含摘要节点与 compress 调用/结果。
- 热重载：只 `touch` 补丁（仅改 mtime）不触发；**改变** `cordis.patch.yml` 内容
  会触发 HMR 的 profile recomposition，并把已变化的 `link:` 包目标一起替换
  （2026-11-03 实测，旧版工具表切换到 0.4）。
