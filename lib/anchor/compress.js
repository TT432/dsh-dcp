/**
 * Anchor compression: turn a validated span of the model-visible surface into
 * one model-authored summary node.
 *
 * The commit is two synchronous appends, in this order:
 *
 *   1. `compaction/prune` — the shadow price of the exact span. dsh's token
 *      meter prices a surface replacement through the log event immediately
 *      before it (`compaction/summary` for a summarizing compaction,
 *      `compaction/prune` for a model-free replacement). A compression is
 *      model-free from the plugin's side: the model writes the summary in its
 *      own turn, so no extra summarization call is made. Without this event
 *      the token-meter's context-pressure projection would keep the old total
 *      and the occupancy display would not shrink after a compression.
 *   2. `user/message` — the summary node, with `surfaceOp: {op:'replace'}`
 *      over the whole span and `sourceEventSeqs` citing every shadowed node.
 *
 * Every hard validation runs before either append, so a rejected call mutates
 * nothing. Summary *shape* is not a hard validation (omp `99a4f87`): deviations
 * ride the result as advisories, because a rejection costs a turn and the
 * model re-sends the same draft anyway.
 *
 * Absorbing an older block is loss-free: the block stays in the log and
 * `expand({block})` returns its text. The new block records the absorbed ids as
 * `parentIds` and lists them as one-line indexes under its summary.
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  assertAnchorNameAvailable,
  findToolCallEntryIndex,
  firstUserEntryIndex,
  readEvents,
  scanAllAnchorCalls,
  scanLiveAnchors,
  surfaceEntries,
  validateAnchorName,
} from './anchors.js'
import {
  ancestorIndex,
  nextBlockId,
  renderCompressedSummary,
  scanCompressionBlocks,
} from './blocks.js'
import { COMPRESSION_SOURCE_KIND, SESSION_START_ANCHOR } from './prompts.js'
import { parseSummary } from './summary.js'

/**
 * Human-readable list of the anchors a model may still start from.
 * @param {{ name: string }[]} liveAnchors - anchors on the current surface.
 * @returns {string} quoted list, or `(none)`.
 */
function anchorList(liveAnchors) {
  return liveAnchors.map((anchor) => `"${anchor.name}"`).join(', ') || '(none)'
}

/**
 * Validate the model-facing arguments of one compress call. Only arguments
 * whose call cannot mean anything are rejected; summary shape is advisory.
 *
 * @param {{ startAnchor?: unknown, summary?: unknown }} args - tool args.
 * @returns {{ startName: string, summary: string, parsed: object }} validated values.
 */
export function validateCompressArguments(args) {
  const rawStart = args?.startAnchor
  if (typeof rawStart !== 'string' || rawStart.trim().length === 0) {
    throw new Error('compress requires a non-empty string "startAnchor"')
  }
  const rawSummary = args?.summary
  if (typeof rawSummary !== 'string' || rawSummary.trim().length === 0) {
    throw new Error('compress requires a non-empty string "summary" — it is the only record future turns will have of the compressed span')
  }
  return {
    startName: rawStart.trim(),
    summary: rawSummary,
    parsed: parseSummary(rawSummary),
  }
}

/**
 * Resolve the surface span one compress call replaces.
 *
 * @param {object} input
 * @param {object} input.session - live dsh Session.
 * @param {string} input.startName - validated start anchor name.
 * @param {string} input.callId - the in-flight compress tool-call id.
 * @returns {{ startName: string, startToolCallId: string | null, shadowedEntries: object[], shadowedSeqs: number[] }}
 * @throws a model-facing error when the anchor or span cannot be resolved.
 */
export function resolveCompressionSpan({ session, startName, callId }) {
  const events = readEvents(session)
  const entries = surfaceEntries(session, events)
  if (entries.length === 0) {
    throw new Error('compress: no session messages are available to compress; retry on a later turn')
  }
  const compressIdx = findToolCallEntryIndex(entries, callId)
  if (compressIdx < 0) {
    throw new Error(
      'compress must be a direct model tool call: this call is not on the model-visible surface ' +
        '(run_code sub-dispatches cannot compress, because the span end is the call itself)',
    )
  }
  const liveAnchors = scanLiveAnchors(entries)
  const recordedAnchors = scanAllAnchorCalls(events)
  const nodes = [...session.surface.nodes]
  const compressNodePos = nodes.indexOf(entries[compressIdx].seq)
  if (compressNodePos < 0) {
    throw new Error('compress: the compress call is no longer on the current surface; retry on a later turn')
  }

  let startNodePos
  let startToolCallId = null
  if (startName === SESSION_START_ANCHOR) {
    const firstUser = firstUserEntryIndex(entries)
    if (firstUser < 0) throw new Error('compress: "session_start" is unavailable because the session has no user message yet')
    const firstUserNodePos = nodes.indexOf(entries[firstUser].seq)
    if (firstUserNodePos < 0) {
      throw new Error('compress: the original prompt is no longer on the current surface; retry on a later turn')
    }
    startNodePos = firstUserNodePos + 1 // the prompt itself is preserved
  } else {
    const anchor = liveAnchors.find((candidate) => candidate.name === startName)
    if (anchor === undefined) {
      const consumed = recordedAnchors.find((candidate) => candidate.name === startName)
      if (consumed !== undefined) {
        throw new Error(
          `startAnchor "${startName}" was consumed by an earlier compression. Available anchors: ${anchorList(liveAnchors)}`,
        )
      }
      throw new Error(`unknown startAnchor "${startName}". Available anchors: ${anchorList(liveAnchors)}`)
    }
    startNodePos = nodes.indexOf(anchor.seq) // the anchor's assistant message is consumed with the span
    if (startNodePos < 0) {
      throw new Error('compress: the start anchor is no longer on the current surface; retry on a later turn')
    }
    startToolCallId = anchor.toolCallId
  }

  const endNodePos = compressNodePos - 1
  if (startNodePos > endNodePos) {
    throw new Error(`range from "${startName}" to this compress call contains no messages to compress`)
  }
  if (startNodePos < 0 || endNodePos >= nodes.length) {
    throw new Error('compress: the resolved span is outside the current surface; retry on a later turn')
  }

  // Replacement metadata must cover EVERY surface node, including empty
  // system/developer/assistant nodes that derive no message. Message entries
  // are only for pricing and anchor scans; shadowedSeqs is the full node span.
  const shadowedSeqs = nodes.slice(startNodePos, endNodePos + 1)
  const shadowedSet = new Set(shadowedSeqs)
  const shadowedEntries = entries.filter((entry) => shadowedSet.has(entry.seq))
  if (shadowedEntries.length === 0) {
    throw new Error(`range from "${startName}" to this compress call contains no messages to compress`)
  }

  // A non-empty system node after node 0 is a prompt update on an in-history
  // route; shadowing it would make the model fall back to an older prompt.
  // The prompt is not conversation content, so refuse rather than silently
  // compressing less than requested.
  for (const entry of shadowedEntries) {
    if (entry.message?.role === 'system') {
      throw new Error(
        `compress: the span from "${startName}" crosses a system-prompt update (seq ${entry.seq}); ` +
          'the prompt is not compressible content. Add a new anchor after that update and compress from it.',
      )
    }
  }
  return { startName, startToolCallId, shadowedEntries, shadowedSeqs }
}

/**
 * Append one compression: shadow price, then the summary replacement.
 *
 * @param {object} input
 * @param {object} input.session - live dsh Session.
 * @param {object} input.tokenMeter - `ctx.tokenMeter` service (needed for the shadow price).
 * @param {string} input.startAnchor - raw model argument (validated here).
 * @param {string} input.summary - raw model argument (validated here).
 * @param {string} input.callId - the in-flight compress tool-call id.
 * @param {object} [input.logger] - debug logger.
 * @returns {{ anchor: string, blockId: number, shadowedMessages: number, absorbed: number, absorbedBlockIds: number[], chain: { blockId: number, anchor: string, gist: string }[], shadowedTokens: number, freedTokens: number, problems: string[], shadowedSeqs: number[] }}
 */
export function compressFromAnchor({ session, tokenMeter, startAnchor, summary, callId, logger }) {
  const validated = validateCompressArguments({ startAnchor, summary })
  if (typeof tokenMeter?.estimateMessage !== 'function') {
    throw new Error(
      'compress requires the dsh token meter service (ctx.tokenMeter); it prices the replaced span so the context occupancy ' +
        'projection shrinks correctly',
    )
  }

  const span = resolveCompressionSpan({ session, startName: validated.startName, callId })
  const shadowed = span.shadowedEntries
  const shadowedSeqs = span.shadowedSeqs
  const firstSeq = shadowedSeqs[0]
  const lastSeq = shadowedSeqs[shadowedSeqs.length - 1]

  // Blocks are re-derived from the log; the absorbed ones are exactly the
  // compression nodes inside the new span. Their text stays in the log, so
  // absorption needs no merge/inheritance validation.
  const blocks = scanCompressionBlocks(readEvents(session))
  const blockBySeq = new Map(blocks.map((block) => [block.seq, block]))
  const absorbedBlocks = []
  for (const entry of shadowed) {
    if (entry.message?.source?.kind !== COMPRESSION_SOURCE_KIND) continue
    const block = blockBySeq.get(entry.seq)
    if (block !== undefined) absorbedBlocks.push(block)
  }
  const parentIds = absorbedBlocks.map((block) => block.blockId).sort((a, b) => b - a)
  const blockId = nextBlockId(blocks)
  const ancestors = ancestorIndex(blocks, { blockId, parentIds })

  let shadowedTokenCount = 0
  for (const entry of shadowed) {
    const price = tokenMeter.estimateMessage(entry.message)
    if (!Number.isFinite(price) || price < 0) {
      throw new Error('compress: the token meter returned an invalid price for the compressed span')
    }
    shadowedTokenCount += price
  }
  shadowedTokenCount = Math.round(shadowedTokenCount)
  if (!Number.isSafeInteger(shadowedTokenCount)) {
    throw new Error('compress: the compressed span is too large to price safely')
  }

  const source = {
    kind: COMPRESSION_SOURCE_KIND,
    anchor: validated.startName,
    blockId,
    parentIds,
    ...(span.startToolCallId === null ? {} : { startToolCallId: span.startToolCallId }),
  }
  const message = createUserMessage({
    content: [{
      type: 'text',
      text: renderCompressedSummary({
        startName: validated.startName,
        blockId,
        summary: validated.summary,
        ancestors,
      }),
    }],
    source,
  })
  const summaryTokens = Math.round(tokenMeter.estimateMessage(message))
  const freedTokens = Math.max(0, shadowedTokenCount - summaryTokens)

  // Shadow price (arms the token-meter projection claim) and replacement stay
  // adjacent and synchronous: no await may separate them (the protocol requires
  // the metering event to be the immediately preceding event).
  session.append('compaction/prune', {
    shadowedRange: { start: firstSeq, end: lastSeq },
    shadowedSeqs,
    shadowedTokenCount,
  })
  const replacement = session.append('user/message', message, {
    surfaceOp: { op: 'replace', startSeq: firstSeq, endSeq: lastSeq },
    sourceEventSeqs: shadowedSeqs,
  })

  logger?.info('Compressed span', {
    anchor: validated.startName,
    blockId,
    shadowedMessages: shadowed.length,
    shadowedNodes: shadowedSeqs.length,
    shadowedTokenCount,
    freedTokens,
    absorbedBlocks: parentIds,
    problems: validated.parsed.problems.length,
    replacementSeq: replacement.seq,
  })
  return {
    anchor: validated.startName,
    blockId,
    shadowedMessages: shadowed.length,
    absorbed: absorbedBlocks.length,
    absorbedBlockIds: parentIds,
    chain: ancestors,
    shadowedTokens: shadowedTokenCount,
    freedTokens,
    problems: validated.parsed.problems,
    shadowedSeqs,
  }
}

/**
 * Record one anchor after validating its name against the whole session.
 * @param {object} input
 * @param {object} input.session - live dsh Session.
 * @param {unknown} input.anchor - raw model argument.
 * @param {string} input.callId - the in-flight add_anchor tool-call id.
 * @returns {{ anchor: string }} validated name.
 */
export function recordAnchor({ session, anchor, callId }) {
  const name = validateAnchorName(anchor)
  if (typeof callId !== 'string' || callId.length === 0) {
    throw new Error('add_anchor: missing tool-call identity')
  }
  const events = readEvents(session)
  const recorded = scanAllAnchorCalls(events)
  assertAnchorNameAvailable(recorded, name, callId)
  // An anchor is its model-visible tool call. A run_code sub-dispatch has no
  // such surface node, so accepting it would report success without creating
  // any anchor that compress could later resolve.
  const entries = surfaceEntries(session, events)
  if (findToolCallEntryIndex(entries, callId) < 0) {
    throw new Error(
      'add_anchor must be a direct model tool call: this call is not on the model-visible surface ' +
        '(run_code sub-dispatches cannot create anchors)',
    )
  }
  return { anchor: name }
}
