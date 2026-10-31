/**
 * Tool descriptions and message formats for anchor-based context compression.
 *
 * These are plain text constants: dsh adds no system-prompt injection, so the
 * tool catalog itself carries the usage contract (same design as the omp dcp
 * extension this plugin mirrors).
 */

/** Built-in tool name (model-facing) that places a named anchor. */
export const ADD_ANCHOR_TOOL = 'add_anchor'

/** Built-in tool name (model-facing) that replaces a span with a summary. */
export const COMPRESS_TOOL = 'compress'

/** Reserved virtual anchor that resolves right after the original user prompt. */
export const SESSION_START_ANCHOR = 'session_start'

/** Message source kind of a landed compression node. */
export const COMPRESSION_SOURCE_KIND = 'dcp-compress'

/** Message source kind of the reminder injected at a step. */
export const REMINDER_SOURCE_KIND = 'dcp-reminder'

/** Prefix identifying the synthetic token-pressure reminder. */
export const REMIND_MARKER = '[dcp:context]'

/** Prefix identifying the work-unit-boundary reminder. */
export const UNIT_MARKER = '[dcp:unit]'

/** Prefix identifying the tool-call-budget reminder. */
export const BUDGET_MARKER = '[dcp:budget]'

/** Maximum accepted anchor-name length. */
export const ANCHOR_NAME_MAX = 64

/**
 * `session_start` is not a peer of the named anchors: it wipes everything and
 * therefore demands force + the strict tier. Offering it inside "startAnchor
 * one of ..." made the reminder recommend a call that fails validation, so it
 * is called out separately (omp dcp does the same).
 */
export const FULL_SESSION_HINT =
  'Do not reach for "session_start" as a shortcut: a full-session wipe needs force=true, the strict summary tier, and must merge every locator of the block it absorbs.'

/** The fixed summary slot list, as named in every prompt. */
export const SUMMARY_SLOT_NAMES = '当前任务 / 已完成 / 未完成 / 约束 / 文件与命令 / 在跑作业与句柄'

/** add_anchor tool description. */
export const ADD_ANCHOR_DESCRIPTION = `Place a named anchor at this point in the conversation. Anchors are stable bookmarks used by the compress tool: compress(startAnchor, summary) replaces everything from the anchor up to the compress call with your summary. Drop an anchor whenever a stretch of work concludes so you can compress it later. A virtual anchor "session_start" always exists right after the original user prompt. Anchor names must be unique within the session; "session_start" is reserved. Anchors are consumed by the compression that starts from them.`

/** compress tool description (summary contract + inheritance rules). */
export const COMPRESS_DESCRIPTION = `Replace everything from a start anchor up to this compress call with a structured summary you author, reclaiming context. startAnchor is a name placed earlier with add_anchor, or "session_start" for the point right after the original user prompt. The end of the compressed span is implicit: this compress call itself.

Semantics: everything after the start anchor up to this compress call is replaced by your summary, which appears right before the compress call; the compress call and its confirmation stay in your context, so later turns can see the compression happened. The start anchor is consumed. Earlier compressions whose compress call lies before the start anchor stay in place — their summaries and yours accumulate in order. A span that reaches back across a prior compression absorbs it entirely: the absorbed summary is DELETED, so your new summary must MERGE it, not rewrite it. Concretely, every locator (path, file:line, \`identifier\`, CONSTANT) that the absorbed blocks recorded under 已完成 / 约束 / 文件与命令 must still appear somewhere in your summary; if a fact is obsolete, keep its locator and say it is superseded (「已废弃: <原因>」). A summary that drops one is rejected with the missing locators listed — nothing is compressed. Errors (unknown anchor, empty range, start anchor inside an active compression, dropped inherited facts) change nothing. Session history is never modified — compression only reshapes what is sent to you.

SUMMARY SHAPE — fixed slots, in this exact order and with these exact Chinese headers; a summary that does not match is rejected and nothing is compressed:

## 当前任务
- 正在做的事（必填，不能写「无」） 证据: <定位>
## 已完成
- 动作/结论 证据: <定位>
## 未完成
- 下一步的具体动作 证据: <定位>
## 约束
- 用户要求/不变量/禁忌 证据: <定位>
## 文件与命令
- 关键路径、入口、可复现命令 证据: <定位>
## 在跑作业与句柄
- 后台作业、子代理 id、服务端口 证据: <定位>

Rules: every slot appears exactly once, in that order; inside a slot each line starts with "- " and each fact ends with 证据: <locator>; a slot with nothing to report is a single "- 无". Facts are externalized — cite where the fact lives (path, file:line, session id, tool name, command) and do NOT paste原文, because the summary is the only record future turns will have and a paraphrase without a pointer cannot be re-checked later. Facts stay under 600 characters.

Full-session compression (startAnchor = "session_start") requires force=true: it discards all working context and is the known amnesia footgun. Only confirm when the current work unit is complete, and expect the strict tier — 已完成 and 未完成 must carry real content, and every 证据 must be a locatable pointer (path, file:line, session id, URI or command). That cost is deliberate: the more a summary throws away, the more it must prove.`

/** `summary` parameter description shared by the tool schema. */
export const COMPRESS_SUMMARY_PARAM_DESCRIPTION = 'Slot-structured summary replacing everything from the anchor up to this compress call; the only record future turns will have. Required slot headers, in this order: ## 当前任务 / ## 已完成 / ## 未完成 / ## 约束 / ## 文件与命令 / ## 在跑作业与句柄; every fact is a `- ` bullet ending in `证据: <locator>`, an empty slot is `- 无`. Absorbing a prior compression deletes it: every locator it recorded under 已完成 / 约束 / 文件与命令 must still appear in this summary (say 「已废弃: <原因>」 with the locator if obsolete) or the compress is rejected.'

/** Tool result text for a placed anchor. */
export function formatAnchorPlaced(name) {
  return `Anchor "${name}" recorded at this point in the conversation. Cite it as startAnchor in a later compress call; that compression consumes the anchor.`
}

/** Model-facing header of the replacement node, mirroring omp's format. */
export function formatCompressedSummary(startName, summary) {
  return `[Compressed: ${startName} → here]\n${summary}`
}

/** Quoted anchor list for reminder text. */
function anchorList(anchors) {
  return anchors.length > 0 ? anchors.map((name) => `"${name}"`).join(', ') : '"session_start"'
}

/** Over-threshold (token pressure) reminder content. */
export function formatReminder({ tokens, window, anchors }) {
  const pct = window ? ` (${Math.round((tokens / window) * 100)}% of the ${window}-token window)` : ''
  return (
    `${REMIND_MARKER} Context is at ~${tokens} tokens${pct}. Compress NOW: ` +
    `call compress with startAnchor one of ${anchorList(anchors)} and a complete technical ` +
    `summary of everything since that anchor. Finish the compress before starting any new work. ` +
    FULL_SESSION_HINT
  )
}

/** Work-unit-boundary reminder content. */
export function formatUnitReminder({ anchors }) {
  return (
    `${UNIT_MARKER} The todo set just completed — that is a work-unit boundary. ` +
    `Place an anchor at the start of the next unit, then compress the finished one: ` +
    `compress with startAnchor one of ${anchorList(anchors)}, and make the summary follow the fixed slot ` +
    `schema (${SUMMARY_SLOT_NAMES}); every fact ends in 证据: <定位>. ${FULL_SESSION_HINT}`
  )
}

/** Tool-call-budget reminder content. */
export function formatBudgetReminder({ used, budget, anchors }) {
  return (
    `${BUDGET_MARKER} ${used} tool calls have run since the last compress (budget ${budget}). ` +
    `If this stretch has converged, place an anchor and compress it now (startAnchor one of ` +
    `${anchorList(anchors)}); if it has not, keep working — this reminder repeats until you compress. ` +
    FULL_SESSION_HINT
  )
}
