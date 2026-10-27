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

/** Message source kind of the over-threshold reminder injected at a step. */
export const REMINDER_SOURCE_KIND = 'dcp-reminder'

/** Prefix identifying the synthetic reminder text. */
export const REMIND_MARKER = '[dcp:context]'

/** Maximum accepted anchor-name length. */
export const ANCHOR_NAME_MAX = 64

/** add_anchor tool description. */
export const ADD_ANCHOR_DESCRIPTION = `Place a named anchor at this point in the conversation. Anchors are stable bookmarks used by the compress tool: compress(startAnchor, summary) replaces everything from the anchor up to the compress call with your summary. Drop an anchor whenever a stretch of work concludes so you can compress it later. A virtual anchor "session_start" always exists right after the original user prompt. Anchor names must be unique within the session; "session_start" is reserved. Anchors are consumed by the compression that starts from them.`

/** compress tool description. */
export const COMPRESS_DESCRIPTION = `Replace everything from a start anchor up to this compress call with a technical summary you author, reclaiming context. startAnchor is a name placed earlier with add_anchor, or "session_start" for the point right after the original user prompt. The end of the compressed span is implicit: this compress call itself.

Semantics: everything after the start anchor up to this compress call is replaced by your summary, which appears right before the compress call; the compress call and its confirmation stay in your context, so later turns can see the compression happened. The start anchor is consumed. Earlier compressions whose compress call lies before the start anchor stay in place — their summaries and yours accumulate in order. A span that reaches back across a prior compression absorbs it entirely, so carry its key points into your new summary. The session log keeps every original event; compression only removes the span from what you are sent.

Write summaries that fully preserve decisions, code changes, file paths, and outcomes — they are the only record future turns will have of the compressed span. Never compress content you still need verbatim. Errors (unknown anchor, empty range, an anchor consumed by an earlier compression) change nothing.

Full-session compression (startAnchor = "session_start") requires force=true: it discards all working context and is the known amnesia footgun. Confirm only when the current work unit is complete, and make that summary state 当前任务 / 刚完成的动作 / 下一步具体动作 (current task, last action, next step).`

/** Tool result text for a placed anchor. */
export function formatAnchorPlaced(name) {
  return `Anchor "${name}" recorded at this point in the conversation. Cite it as startAnchor in a later compress call; that compression consumes the anchor.`
}

/** Model-facing header of the replacement node, mirroring omp's format. */
export function formatCompressedSummary(startName, summary) {
  return `[Compressed: ${startName} → here]\n${summary}`
}

/** Over-threshold reminder content. */
export function formatReminder({ tokens, window, anchors }) {
  const pct = window ? ` (${Math.round((tokens / window) * 100)}% of the ${window}-token window)` : ''
  const list = anchors.length > 0 ? anchors.map((name) => `"${name}"`).join(', ') : '"session_start"'
  return (
    `${REMIND_MARKER} Context is at ~${tokens} tokens${pct}. Compress NOW: ` +
    `call compress with startAnchor one of ${list} and a complete technical ` +
    `summary of everything since that anchor. Finish the compress before starting any new work.`
  )
}
