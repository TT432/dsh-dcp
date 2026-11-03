# dsh-dcp：锚点上下文压缩（链式摘要 + expand）+ LLM-DCP 提示词压缩

本插件包含两条独立能力线：

1. **锚点压缩**：对齐本机 omp 的 dcp 扩展（`~/.omp/agent/extensions/dcp`，anchor mode），
   由模型通过 `add_anchor` / `compress` / `expand` 主动管理工作上下文；摘要采用推荐的固定槽位，
   但**只做提示、不拒绝压缩**；被吸收的旧压缩块形成链，全文随时可用 `expand` 取回；
   两类提醒（token 压力 / 工作单元结束）驱动模型压缩。
2. **LLM-DCP**（[arXiv:2504.11004](https://arxiv.org/abs/2504.11004)）：既有的
   `dcp_compress` 工具，用可配置 LLM 充当 DCP-Agent 迭代压缩任意文本。
   它不是 omp 对齐项，行为保持不变。

锚点压缩的完整规格、前置/后置条件、不变量与异常行为见
[`docs/anchor-compression.md`](./docs/anchor-compression.md)。
对齐的上游提交：`48d49ac`（锚点压缩基线）→ `c83eeaa`（摘要槽位、继承校验、三类提醒）→
`0583104`（证据正则放宽）→ `99a4f87`（摘要链路化、移除闸门与工具调用计数器）。

## 工具

### `add_anchor({ anchor })`

在当前对话位置放置具名锚点；锚点身份是该工具调用的 provider tool-call id。

- 名字非空、不含空白、长度 ≤ 64；`session_start` 为保留名。
- 名字在**整个会话内唯一**（包括已被压缩消耗的锚点），重复会报错。
- `session_start` 是内置虚拟锚点，位于原始用户 prompt 之后；不需要标记消息。
- 返回文本说明「这个锚点标记**下一段**的起点」：先打锚点、继续工作，结束时从更早的
  锚点（或 `session_start`）压缩，把刚打的锚点留给下一段。
- 与 `compress` 一样，`add_anchor` 也必须是模型直接发起的工具调用；
  `run_code` 子调用会报错，避免“返回成功但没有真正锚点”。

### `compress({ startAnchor, summary })`

把从 `startAnchor` 到**本次 compress 调用之前**的上下文替换为模型自己写的技术摘要。

```text
prompt · work1 · add_anchor(a1) · work2 · work3 · compress(a1, summary)
```

压缩后模型看到：

```text
prompt · work1 · [Compressed: a1 → here] (block 1) summary · compress(a1, summary) · 确认结果
```

- 起点锚点及其结果被消耗；compress 调用与确认结果保留，后续轮次知道压缩发生过。
- 摘要由**替换节点**携带，落在 compress 调用之前；节点带 `(block N)` 标识。
- 链式压缩的摘要按序累积；新起点在旧摘要之前时**吸收**旧块：旧块文本不删除，
  在新摘要下渲染为一行索引（含 gist 与 `expand({block: N})`），全文随时可取回。
- 未知锚点、空区间、已消耗锚点、跨 system-prompt 更新、经 `run_code` 子调用等情况
  都会报错，且不产生任何会话事件；错误会列出可用锚点。
- **不再有 `force` 参数**：`session_start` 可直接压缩。链保证旧文本仍可展开，所以
  全量压缩不再是“不可恢复的失忆”；提醒文本仍建议优先用命名锚点。
- **摘要形状是请求，不是闸门**：槽位缺失、证据缺失、自由格式都不会拒绝压缩；
  偏差随成功结果返回为 `摘要形状提示（已压缩，未拒绝）`。

**推荐的摘要结构（固定槽位 + 证据定位）**：

```text
## 当前任务        - 正在做的事 证据: <定位>
## 已完成          - 动作/结论 证据: <定位>
## 未完成          - 下一步具体动作 证据: <定位>
## 约束            - 用户要求/不变量 证据: <定位>
## 文件与命令      - 关键路径/命令 证据: <定位>
## 在跑作业与句柄  - 作业/子代理/端口 证据: <定位>
```

- 六个槽位各一次、建议按此顺序；没有内容的槽位写单条 `- 无`；`当前任务` 建议写实际任务。
- 每条事实是 `- ` 开头的条目，并以 `证据: <定位>` 结尾（路径、`file:line`、会话 id、
  commit、URL、命令）。定位没有白名单：`。证据:`、全角冒号、任意定位文本都接受。
- 摘要是**指针**，不是原文副本；吸收旧块时不再要求逐字继承旧定位——旧块全文就在链里。

**持久化语义**：dsh 的 loop 请求是会话日志的纯函数，不能像 omp 一样只改请求副本。
因此本插件使用 dsh 官方的 surface replacement 机制：只追加事件，不删除或改写原始日志；
被压缩区间仅从派生给模型的历史中遮蔽，resume 后仍保持压缩结果。
替换前紧邻追加 `compaction/prune` 影子计价事件，保证上下文占用投影按
「被遮蔽区间 − 摘要节点」正确更新，不发生漂移。

压缩块没有独立状态：它们就是带 `source.kind === 'dcp-compress'` 的 `user/message` 事件。
`blockId` / `parentIds` 记录在消息的 `source` 里；旧版本（无 `blockId`）的块按日志顺序
补号，跨 resume 稳定。

### `expand({ block })`

取回一条压缩块的全文——通常来自更新摘要下的索引行。返回该块模型摘要的全文，
并附上它自己的索引行（若有更深一层），因此可以沿链继续往回走。
块已经不在当前表面上（被后来压缩吸收）也仍然可取回，因为原始事件保留在日志里。
未知块号报错并列出已知块号。

### `dcp_compress`

既有 LLM-DCP 工具：多步迭代压缩 prompt/文档/会话文本，支持 `targetRatio`、`steps`、
`keyPoints` 与 provider/model 覆盖。与上面的锚点压缩不是同一能力。

## 配置（仅环境变量，无配置文件）

| 变量 | 含义 |
| --- | --- |
| `DCP_REMIND_THRESHOLD` | token 压力提醒阈值。缺省 `0.4`；`(0,1)` 为上下文窗口比例，`>=1` 为绝对 token 数，`<=0` 关闭。 |
| `DCP_DEBUG` | 非空且非 `0/false/off/no` 时，写 `$DSH_HOME/logs/dcp/dcp.log` 与 `context-<session>.jsonl`。 |

提醒在 `agent/pre-step` 注入，两类按优先级 `[dcp:context]` > `[dcp:unit]`，
**每个 turn 至多一条**；`[dcp:context]` 还按上下文每增长 10% 才提醒一次——档位从最近一条
token 提醒的估算值推导，其后的压缩或估算回落会重置档位（不新增持久状态）。
与 omp 不同，dsh 的提醒是普通 `user/message` 事件，会持久化，因此用 turn 级节流 +
档位共同限制重复（omp 的提醒不落盘，靠每请求 + 档位）。

- `[dcp:context]`：估算上下文达到阈值。
- `[dcp:unit]`：最新的 `todo_write` 结果报告列表完成
  （dsh 的 `Updated todo list: 0 pending, 0 in progress, N completed.`，N>0；
  也识别 omp 宿主的 `All items complete` / `Remaining items (0)` /
  `Remaining items: none.` / `0 open`）。按结果 toolCallId 去重——同一个完成结果只提醒一次，
  模型再次调用 `todo_write` 并完成后才会重新提醒。

**没有工具调用计数器**（omp `99a4f87` 删除 `[dcp:budget]`）：调用数不是上下文代价。
旧实现按 40 次调用提醒，会在上下文远未接近窗口时逼模型压缩，随后不得不重读刚读过的文件。
token 压力是唯一可度量的成本，所以只保留 token / unit 两类触发。

token 压力提醒缺省阈值取 `0.4` 而非 omp 的 `0.8`，是为了让模型驱动压缩先于 dsh 自动
compaction（后者压力预算约为 `0.8 × (window − 输出预留 − 64k headroom)`）。
token 缺失时的启发式估算按 CJK（约 2 字符/token）与 ASCII（约 4 字符/token）分档，
避免中文会话被低估约 40%。

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

新装依赖或修改插件源码后**需要重启对应应用**。修改 `cordis.patch.yml` 会热加载，但
`link:` 源码本身的变更**不保证热更新**：2026-11-03 本机 web 进程在源码改为 0.4.0 后，
新建子代理会话仍执行旧版工具（旧版确认文本、压缩节点无 `blockId`、`compress(session_start)`
报旧版 force/严格档错误）。

## 测试

```powershell
node --test test/compress-tests.mjs test/tool-tests.mjs test/blocks-tests.mjs \
  test/anchor-tests.mjs test/anchor-tool-tests.mjs test/remind-tests.mjs test/summary-tests.mjs
```

在受限沙箱（如 DSH 文件沙箱）内 Node 测试运行器 spawn 子进程会 EPERM，
改用 `node --test --test-isolation=none ...` 在同一进程内运行。

覆盖：LLM-DCP 既有回归、锚点扫描与唯一性、`session_start`、区间替换、链式累积、
吸收旧块与索引行、`expand` 全文取回/未知块号、旧块补号、摘要 advisory 解析与证据正则、
两类提醒的触发/优先级/节流/unit 去重、token 提醒的 10% 档位、CJK 估算、工具注册与参数校验。

本版本的单元/集成测试在真实 dsh Session（0.2.1-rc.3）上运行，覆盖上述全部逻辑，共 **83 项全绿**。

**实机 e2e（链式吸收 / expand）尚未在本机运行进程重跑**：2026-11-03 在 web 进程里派生子代理会话
执行 `add_anchor → compress(a1) → compress(session_start) → expand(block 1)` 时，运行中的插件
仍是旧版（`add_anchor` 返回旧文本、压缩节点无 `(block N)` / `blockId`、`compress(session_start)`
报旧版 force/严格档错误），说明已链接源码没有热更新（见「安装」）。该次 e2e 只证明旧版行为，
**不能作为本版本的实机证据**；重启应用后应按同一序列复测。

历史实机记录（链式摘要之前的版本，保持原文）：模型依次调用 `add_anchor`、`read`、
`compress`，会话日志确认：

```text
seq 28  compaction/prune   shadowedRange 18–22, shadowedTokenCount 2268
seq 29  user/message       replace 18–22, [Compressed: e2e-anchor → here] ...
seq 30  tool/result        compress 确认：3 message(s) elided，并报告净释放 token 估算
```

返回的 `freedTokens` 按 `shadowedTokenCount − 摘要节点估算 token` 计算（下限为 0），
不是把被遮蔽区间的总量误报为净释放量。原始 add_anchor/read 事件仍在日志中；派生上下文只保留
摘要、compress 调用与确认结果。同一子会话随后执行第二次压缩（`e2e-final-2`）时，首个
`[Compressed: e2e-final → here]` 摘要仍保留，返回 `4 message(s) elided, ~628 tokens freed`。
该记录来自旧版本，不包含链式索引与 `expand`。

## 实现结构

```text
lib/index.js             插件入口：注册 3 个锚点工具 + dcp_compress 与提醒监听
lib/anchor/anchors.js    表面节点/锚点扫描、名字校验、session_start
lib/anchor/blocks.js     压缩块链：日志扫描/补号、祖先索引、expand 渲染
lib/anchor/compress.js   区间解析、校验、影子计价 + 替换提交（链元数据）
lib/anchor/summary.js    advisory 摘要解析、证据定位、gist、问题提示
lib/anchor/prompts.js    工具描述、摘要与两类提醒文本、索引格式
lib/anchor/remind.js     阈值解析、token 估算（CJK 分档）、两类提醒与去重/节流
lib/anchor/logger.js     DCP_DEBUG 文件日志与 cordis logger
lib/tools/anchor.js      add_anchor / compress / expand 工具接线
lib/dcp/compress.js      既有 LLM-DCP 核心
lib/tools/compress.js    既有 dcp_compress 工具
```

与 omp 的刻意差异（完整清单见规格文档）：压缩是持久的 append-only 表面替换；
不注入 `session_start` 标记消息；提醒会持久化且每 turn 至多一次；缺省提醒阈值为 `0.4`；
in-history system-prompt 更新不可被压缩区间遮蔽；token 提醒的 10% 档位与 unit 去重水位
都从日志推导而不写独立状态；压缩块本身也是日志事件，没有 omp 的 `dcp-state` 与版本迁移；
dsh 禁止 `run_code` 子调用压缩，因此无需 omp 的 eval-kernel 绕过路径。

## 参考

- 对齐基准：`~/.omp/agent/extensions/dcp`（README、docs/design.md、src/*），当前跟进 `99a4f87`。
- 规格：[`docs/anchor-compression.md`](./docs/anchor-compression.md)。
- LLM-DCP：https://arxiv.org/abs/2504.11004 · 官方仓库（暂无代码）：https://github.com/Fhujinwu/DCP
