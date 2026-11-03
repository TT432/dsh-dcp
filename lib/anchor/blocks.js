/**
 * Compression blocks over the append-only session log.
 *
 * dsh has no custom-entry store like omp's `dcp-state`: a landed compression is
 * itself a session event (`user/message` with `source.kind === 'dcp-compress'`),
 * and the log keeps that event even after the surface replacement hides it.
 * The block list is therefore re-derived from the log on every use — there is
 * no durable plugin state to migrate or reconcile.
 *
 * Blocks form a **chain** (same model as omp v5): a compression that reaches
 * back across an older block records it in `parentIds` and lists it as a
 * one-line index under the new summary. The older block's text is never
 * deleted — `expand({block})` returns it verbatim, including its own index
 * lines, so the model can walk further back.
 *
 * Block identity: new blocks persist `blockId` in the compression message's
 * `source`. Blocks written before the chain change carry no id; they are
 * assigned 1..k in log order (deterministic, since the log is append-only).
 * New ids continue after the highest id ever seen.
 */

import { readEvents } from './anchors.js'
import { gistOf } from './summary.js'
import { COMPRESSION_SOURCE_KIND, formatCompressedSummary, formatExpandedBlock } from './prompts.js'

/** Marker that starts the ancestor-index footer inside a compression message. */
const INDEX_MARKER = '已并入本块的旧压缩块（索引，全文未丢失）:'

/**
 * Text of a landed compression node.
 * @param {object} message - derived message of the compression node.
 * @returns {string} the node's text (empty when it carries none).
 */
function compressionNodeText(message) {
  const content = message?.content
  if (!Array.isArray(content)) return ''
  for (const block of content) {
    if (block?.type === 'text' && typeof block.text === 'string') return block.text
  }
  return ''
}

/**
 * The model-authored summary of a compression node: everything after the
 * `[Compressed: ... → here]` header, with the ancestor-index footer removed.
 *
 * @param {string} text - full compression-node text.
 * @returns {string} summary body.
 */
export function extractSummaryBody(text) {
  const firstNewline = text.indexOf('\n')
  if (firstNewline < 0) return ''
  const rest = text.slice(firstNewline + 1)
  const indexAt = rest.lastIndexOf(`\n\n${INDEX_MARKER}`)
  return indexAt < 0 ? rest : rest.slice(0, indexAt)
}

/**
 * Resolve a compression event to a block record.
 * @param {number} seq - event seq of the compression node.
 * @param {object} message - the node's message.
 * @param {number} blockId - resolved block id.
 * @returns {object} block record.
 */
function toBlock(seq, message, blockId) {
  const source = message?.source ?? {}
  const text = compressionNodeText(message)
  return {
    blockId,
    seq,
    anchor: typeof source.anchor === 'string' && source.anchor.length > 0 ? source.anchor : 'unknown',
    startToolCallId: typeof source.startToolCallId === 'string' ? source.startToolCallId : null,
    parentIds: Array.isArray(source.parentIds)
      ? source.parentIds.filter((id) => Number.isSafeInteger(id) && id > 0)
      : [],
    text,
    summary: extractSummaryBody(text),
  }
}

/**
 * Every compression block recorded in the session log, in log order.
 *
 * @param {readonly object[]} events - committed session events.
 * @returns {object[]} blocks (oldest first).
 */
export function scanCompressionBlocks(events) {
  const blocks = []
  const used = new Set()
  const legacy = []

  for (const event of events) {
    if (event?.type !== 'user/message') continue
    const message = event.data
    if (message?.source?.kind !== COMPRESSION_SOURCE_KIND) continue
    const declared = message.source.blockId
    if (Number.isSafeInteger(declared) && declared > 0 && !used.has(declared)) {
      used.add(declared)
      blocks.push(toBlock(event.seq, message, declared))
    } else {
      legacy.push({ seq: event.seq, message })
    }
  }

  let next = 1
  for (const entry of legacy) {
    while (used.has(next)) next++
    used.add(next)
    blocks.push(toBlock(entry.seq, entry.message, next))
  }

  blocks.sort((left, right) => left.seq - right.seq)
  return blocks
}

/**
 * Block id for a new compression.
 * @param {readonly object[]} blocks - existing blocks.
 * @returns {number} next id.
 */
export function nextBlockId(blocks) {
  let max = 0
  for (const block of blocks) if (block.blockId > max) max = block.blockId
  return max + 1
}

/**
 * Block by id.
 * @param {readonly object[]} blocks - existing blocks.
 * @param {number} blockId - wanted id.
 * @returns {object | undefined} the block.
 */
export function findBlock(blocks, blockId) {
  return blocks.find((block) => block.blockId === blockId)
}

/**
 * Full ancestry of a block, nearest parent first. A parent always has a
 * smaller id, so cycles are impossible by construction; the traversal still
 * guards against corrupt state.
 *
 * @param {readonly object[]} blocks - existing blocks.
 * @param {{ blockId: number, parentIds: number[] }} block - block (or prospective block).
 * @returns {object[]} ancestor blocks.
 */
export function ancestorsOf(blocks, block) {
  const byId = new Map(blocks.map((candidate) => [candidate.blockId, candidate]))
  const out = []
  const seen = new Set([block.blockId])
  const walk = (ids) => {
    const ordered = [...ids].filter((id) => Number.isSafeInteger(id) && id > 0).sort((a, b) => b - a)
    for (const id of ordered) {
      if (seen.has(id)) continue
      seen.add(id)
      const parent = byId.get(id)
      if (parent === undefined) continue
      out.push(parent)
      walk(parent.parentIds)
    }
  }
  walk(block.parentIds)
  return out
}

/**
 * Ancestors rendered for an index line (id, anchor, gist).
 * @param {readonly object[]} blocks - existing blocks.
 * @param {{ blockId: number, parentIds: number[] }} block - block or prospective block.
 * @returns {{ blockId: number, anchor: string, gist: string }[]} index entries.
 */
export function ancestorIndex(blocks, block) {
  return ancestorsOf(blocks, block).map((ancestor) => ({
    blockId: ancestor.blockId,
    anchor: ancestor.anchor,
    gist: gistOf(ancestor.summary),
  }))
}

/**
 * Text of the replacement node for a new compression.
 *
 * @param {object} input
 * @param {string} input.startName - start anchor name.
 * @param {number} input.blockId - id assigned to the new block.
 * @param {string} input.summary - model-authored summary.
 * @param {{ blockId: number, anchor: string, gist: string }[]} input.ancestors - index entries.
 * @returns {string} replacement message text.
 */
export function renderCompressedSummary({ startName, blockId, summary, ancestors }) {
  return formatCompressedSummary({ startName, blockId, summary, ancestors })
}

/**
 * Render one block in full, plus its own index lines so the model can walk back.
 *
 * @param {object} block - block to render.
 * @param {{ blockId: number, anchor: string, gist: string }[]} ancestors - its ancestry.
 * @returns {string} expanded block text.
 */
export function renderExpandedBlock(block, ancestors) {
  return formatExpandedBlock(block, ancestors)
}

/**
 * Resolve one `expand({ block })` call against the log.
 *
 * @param {object} input
 * @param {object} input.session - live dsh Session.
 * @param {unknown} input.block - raw model argument (numeric id).
 * @returns {{ block: number, content: string }} expanded text.
 * @throws a model-facing error when the id is invalid or unknown.
 */
export function expandBlock({ session, block }) {
  const id = typeof block === 'number' ? block : Number(block)
  if (!Number.isSafeInteger(id)) throw new Error('expand requires an integer "block" id')
  const blocks = scanCompressionBlocks(readEvents(session))
  const found = findBlock(blocks, id)
  if (found === undefined) {
    const known = blocks.map((candidate) => candidate.blockId).join(', ') || '(none)'
    throw new Error(`unknown compression block ${id}. Known blocks: ${known}`)
  }
  return {
    block: found.blockId,
    content: renderExpandedBlock(found, ancestorIndex(blocks, found)),
  }
}
