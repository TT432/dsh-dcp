/**
 * Tool descriptions and message formats for anchor-based context compression.
 *
 * These are plain text constants: dsh adds no system-prompt injection, so the
 * tool catalog itself carries the usage contract (same design as the omp dcp
 * extension this plugin mirrors).
 *
 * Design stance (aligned with omp `99a4f87`): the *prompt* is where the
 * discipline lives. dcp does not reject a summary for its shape, so the
 * description has to state what a good summary is worth and why — the model is
 * the only component that can judge whether a fact deserves to survive. The
 * chain (absorbed blocks stay recallable through `expand`) is what makes a thin
 * summary recoverable instead of fatal.
 */

/** Built-in tool name (model-facing) that places a named anchor. */
export const ADD_ANCHOR_TOOL = 'add_anchor'

/** Built-in tool name (model-facing) that replaces a span with a summary. */
export const COMPRESS_TOOL = 'compress'

/** Built-in tool name (model-facing) that returns a compressed block's full text. */
export const EXPAND_TOOL = 'expand'

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

/** Maximum accepted anchor-name length. */
export const ANCHOR_NAME_MAX = 64

/**
 * `session_start` is not a peer of the named anchors: it discards everything at
 * once. The chain keeps absorbed blocks recallable, so it is allowed without a
 * separate confirmation, but the reminder still steers to a named anchor
 * (omp `99a4f87` removed the `force` gate; the hint is now advice, not a gate).
 */
export const FULL_SESSION_HINT =
  'Full-session compression ("session_start") is allowed and nothing is destroyed — absorbed blocks stay recallable with expand — but it is the most lossy call available: prefer compressing a span that starts at an anchor placed after the finished unit.'

/** The fixed summary slot list, as named in every prompt. */
export const SUMMARY_SLOT_NAMES = '当前任务 / 已完成 / 未完成 / 约束 / 文件与命令 / 在跑作业与句柄'

/** add_anchor tool description. */
export const ADD_ANCHOR_DESCRIPTION = `Place a named anchor at this point in the conversation. Anchors are stable bookmarks used by the compress tool: compress(startAnchor, summary) replaces everything from the anchor up to the compress call with your summary. Drop an anchor whenever a stretch of work concludes so you can compress it later, then keep working from the new position — the anchor you just placed marks where the *next* span starts. A virtual anchor "session_start" always exists right after the original user prompt. Anchor names must be unique within the session; "session_start" is reserved.`

/** compress tool description (summary shape is advisory, chain is recallable). */
export const COMPRESS_DESCRIPTION = `Replace everything from a start anchor up to this compress call with a summary you author, reclaiming context. startAnchor is a name placed earlier with add_anchor, or "session_start" for the point right after the original user prompt. The end of the compressed span is implicit: this compress call itself.

Semantics: everything after the start anchor up to this compress call is replaced by your summary, which appears right before the compress call; the compress call and its confirmation stay in your context, so later turns can see the compression happened. The start anchor is consumed. Earlier compressions whose compress call lies before the start anchor stay in place — their summaries and yours accumulate in order. A span that reaches back across a prior compression absorbs it: the absorbed block is kept in dcp's own chain and listed under your summary as an index line, so its full text is one expand call away and nothing you leave out is destroyed.

No summary is ever rejected. Shape is a request, not a gate — but the summary is the only record of the span, so write it for the reader who has to continue the work:

Recommended shape (fixed slots, this order, these Chinese headers):
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

Rules worth following: each slot once, in order; each fact a "- " bullet ending in 证据: <定位> (path, file:line, session id, commit, URL, command) so the next turn can re-read the source instead of trusting a paraphrase; a slot with nothing to report is a single "- 无". Facts are pointers, not pastes — keep each line short enough to re-read. Deviations come back in this tool's result as 摘要形状提示（已压缩，未拒绝）.

Full-session compression (startAnchor = "session_start") discards all working context at once. It is fine now that absorbed blocks remain recallable, but it is still the most lossy call you can make: prefer compressing a span that ends at an anchor you placed after the work unit, and state in 当前任务 what the next turn should do first.`

/** expand tool description. */
export const EXPAND_DESCRIPTION = `Retrieve the full text of a compression block that is currently shown only as an index line under a newer summary, or re-read your own block. Compression blocks form a chain: a newer block absorbs older ones and lists them as one-line indexes (with a gist). Call expand with the block id from such an index line to get that block's whole summary — including its own index lines, so you can walk further back if the fact you need is older still. Use it when a detail you need was compressed away or looks wrong; the block text is the record dcp kept, so re-read it before re-deriving anything from scratch.`

/** `startAnchor` parameter description shared by the tool schema. */
export const COMPRESS_START_PARAM_DESCRIPTION =
  'Anchor where compression starts (consumed); "session_start" compresses from the original prompt. Prefer a named anchor placed after a finished work unit.'

/** `summary` parameter description shared by the tool schema. */
export const COMPRESS_SUMMARY_PARAM_DESCRIPTION =
  'Your summary of everything from the anchor up to this call — the only record of that span in later turns (absorbed older blocks stay recallable through expand). Recommended slots, in order: ## 当前任务 / ## 已完成 / ## 未完成 / ## 约束 / ## 文件与命令 / ## 在跑作业与句柄; each fact a `- ` bullet ending in `证据: <定位>`. Shape is advisory: a summary that does not follow it is still applied.'

/** `block` parameter description shared by the expand schema. */
export const EXPAND_BLOCK_PARAM_DESCRIPTION =
  'Compression block id, as listed under a newer summary\'s index lines'

/** Tool result text for a placed anchor. */
export function formatAnchorPlaced(name) {
  return `Anchor "${name}" recorded here. It marks where the NEXT span starts: compress an earlier anchor (or "session_start") when this stretch ends, and keep this one as the start of what comes next.`
}

/** One index line per absorbed block. */
export function formatAncestorIndex(ancestors) {
  return ancestors
    .map((ancestor) => `- 块 ${ancestor.blockId}「${ancestor.anchor}」：${ancestor.gist || '(无摘要)'} — 全文: expand({block: ${ancestor.blockId}})`)
    .join('\n')
}

/**
 * Human-facing header of the replacement node: the summary in full, followed by
 * one index line per absorbed ancestor. Ancestors cost O(history) in lines, not
 * in bytes.
 *
 * @param {{ startName: string, blockId: number, summary: string, ancestors: { blockId: number, anchor: string, gist: string }[] }} block
 *        - rendered block.
 * @returns {string} replacement node text.
 */
export function formatCompressedSummary(block) {
  const head = `[Compressed: ${block.startName} → here] (block ${block.blockId})`
  if (!Array.isArray(block.ancestors) || block.ancestors.length === 0) return `${head}\n${block.summary}`
  return `${head}\n${block.summary}\n\n已并入本块的旧压缩块（索引，全文未丢失）:\n${formatAncestorIndex(block.ancestors)}`
}

/**
 * Expanded form of one block (used by `expand`): its summary body plus a
 * deeper-level index, so the model can continue walking the chain.
 *
 * @param {{ blockId: number, anchor: string, summary: string }} block - block to render.
 * @param {{ blockId: number, anchor: string, gist: string }[]} ancestors - its ancestry.
 * @returns {string} expanded block text.
 */
export function formatExpandedBlock(block, ancestors) {
  const head = `[block ${block.blockId}「${block.anchor}」 — 全文]`
  if (!Array.isArray(ancestors) || ancestors.length === 0) return `${head}\n${block.summary}`
  return `${head}\n${block.summary}\n\n更深一层（索引）:\n${formatAncestorIndex(ancestors)}`
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
    `${FULL_SESSION_HINT}`
  )
}

/** Work-unit-boundary reminder content. */
export function formatUnitReminder({ anchors }) {
  return (
    `${UNIT_MARKER} The todo set just completed — that is a work-unit boundary. ` +
    `Place an anchor for the next unit, then compress the finished one: ` +
    `compress with startAnchor one of ${anchorList(anchors)} and a summary whose 当前任务 ` +
    `says what the next turn should do first. ${FULL_SESSION_HINT}`
  )
}
