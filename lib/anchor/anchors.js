/**
 * Anchor model over the dsh session surface.
 *
 * An anchor is a named position in the conversation:
 *   - a real anchor is an `add_anchor` tool call; its identity is the
 *     provider-assigned tool-call id (globally unique, persisted, never
 *     mutated). It lives inside an assistant message on the model-visible
 *     surface;
 *   - `session_start` is a virtual anchor at the first user message (the
 *     original prompt). It is never consumed and never needs a marker message.
 *
 * The surface is the ordered list of message-producing session events
 * (`session.surface.nodes`). A compression replaces a contiguous span of those
 * nodes, so every position this module reports is a surface node seq.
 *
 * Reading history: dsh marks `snapshotEvents()`/`eventAt()` deprecated for its
 * own repository, but they remain the only public readers that expose a
 * surface node's event (needed for tool-call ids and shadow prices). This
 * plugin is an external consumer, so it replays the log through them once per
 * tool call.
 */

import { ADD_ANCHOR_TOOL, ANCHOR_NAME_MAX, SESSION_START_ANCHOR } from './prompts.js'

/** True for a plain non-array object. */
function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Parse a recorded tool-call `arguments` string (raw JSON as the model wrote it).
 * @param {unknown} raw - raw argument string.
 * @returns {Record<string, unknown> | null} parsed object, or null.
 */
export function parseToolArguments(raw) {
  if (typeof raw !== 'string' || raw.trim() === '') return null
  try {
    const parsed = JSON.parse(raw)
    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

/**
 * Narrow one content block to a model tool call.
 * @param {unknown} block - content block.
 * @returns {boolean} whether the block is a `tool-call`.
 */
export function isToolCallBlock(block) {
  return (
    isRecord(block) &&
    block.type === 'tool-call' &&
    typeof block.id === 'string' &&
    block.id.length > 0 &&
    typeof block.name === 'string' &&
    block.name.length > 0
  )
}

/**
 * Normalize an anchor name, returning null for exactly the names the tool
 * rejects. This predicate is shared by tool validation and history scans, so
 * a failed add_anchor (whitespace, overlong, reserved) cannot later resolve
 * as a usable anchor.
 * @param {unknown} raw - candidate name.
 * @returns {string | null} trimmed valid name, or null.
 */
function normalizeAnchorName(raw) {
  if (typeof raw !== 'string') return null
  const name = raw.trim()
  if (name.length === 0 || name.length > ANCHOR_NAME_MAX || /\s/u.test(name) || name === SESSION_START_ANCHOR) return null
  return name
}

/**
 * Anchor name carried by one `add_anchor` tool call, or null.
 * @param {unknown} block - content block.
 * @returns {string | null} trimmed anchor name.
 */
export function anchorNameOfBlock(block) {
  if (!isToolCallBlock(block) || block.name !== ADD_ANCHOR_TOOL) return null
  const args = parseToolArguments(block.arguments)
  return normalizeAnchorName(args?.anchor)
}

/**
 * Replay the committed session log once.
 * @param {object} session - live dsh Session.
 * @returns {readonly object[]} committed events in seq order (empty if unavailable).
 */
export function readEvents(session) {
  if (typeof session.snapshotEvents !== 'function') return []
  try {
    return session.snapshotEvents() ?? []
  } catch {
    return []
  }
}

/**
 * Map the current surface to its derived messages in model-visible order.
 *
 * Each entry carries the node's `seq` (the identity a replacement needs), its
 * committed event, and the message `deriveMessages()` would emit for it.
 * Empty system/developer nodes produce no message and are skipped, matching
 * the derivation itself.
 *
 * @param {object} session - live dsh Session.
 * @param {readonly object[]} [events] - pre-read log snapshot to reuse.
 * @returns {{ seq: number, event: object, message: object }[]} surface entries.
 */
export function surfaceEntries(session, events = readEvents(session)) {
  if (typeof session.deriveEventMessage !== 'function') {
    throw new Error('dsh-dcp: this dsh runtime lacks session.deriveEventMessage(); anchor compression needs dsh 0.2.x')
  }
  const bySeq = new Map()
  for (const event of events) bySeq.set(event.seq, event)
  const entries = []
  for (const seq of session.surface.nodes) {
    const event = bySeq.get(seq)
    if (event === undefined) continue
    const message = session.deriveEventMessage(event)
    if (message === null || message === undefined) continue
    entries.push({ seq, event, message })
  }
  return entries
}

/**
 * Index of the original user prompt on the surface. Prefer the first
 * `source.kind === 'user'` message (the direct human prompt); fall back to the
 * first user-role node for sessions whose prompts use another producer kind.
 * Injected runtime context can precede the actual prompt, so role alone is not
 * sufficient. Tool results are a separate `tool` role in dsh.
 * @param {{ message: object }[]} entries - surface entries.
 * @returns {number} index, or -1 when the session has no user message yet.
 */
export function firstUserEntryIndex(entries) {
  for (let i = 0; i < entries.length; i++) {
    const message = entries[i].message
    if (message?.role === 'user' && message.source?.kind === 'user') return i
  }
  for (let i = 0; i < entries.length; i++) {
    if (entries[i].message?.role === 'user') return i
  }
  return -1
}

/**
 * Every `add_anchor` call recorded anywhere in the session log (including
 * anchors already consumed by an earlier compression), in log order.
 * @param {readonly object[]} events - committed session events.
 * @returns {{ name: string, toolCallId: string, seq: number }[]} recorded anchors.
 */
export function scanAllAnchorCalls(events) {
  const anchors = []
  for (const event of events) {
    if (event?.type !== 'assistant/message') continue
    const content = event.data?.message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) {
      const name = anchorNameOfBlock(block)
      if (name === null) continue
      anchors.push({ name, toolCallId: block.id, seq: event.seq })
    }
  }
  return anchors
}

/**
 * Live anchors on the current surface, in message order. `session_start`
 * comes first when a user prompt exists (its `entryIndex` is that prompt; a
 * compression from it starts at `entryIndex + 1`). Duplicate names should not
 * occur (add_anchor rejects them); the first occurrence wins.
 * @param {{ seq: number, message: object }[]} entries - surface entries.
 * @returns {{ name: string, toolCallId: string | null, entryIndex: number, seq: number | null }[]} anchors.
 */
export function scanLiveAnchors(entries) {
  const anchors = []
  const firstUser = firstUserEntryIndex(entries)
  if (firstUser >= 0) {
    anchors.push({
      name: SESSION_START_ANCHOR,
      toolCallId: null,
      entryIndex: firstUser,
      seq: entries[firstUser].seq,
    })
  }
  const seen = new Set([SESSION_START_ANCHOR])
  for (let i = 0; i < entries.length; i++) {
    const content = entries[i].message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) {
      const name = anchorNameOfBlock(block)
      if (name === null || seen.has(name)) continue
      seen.add(name)
      anchors.push({ name, toolCallId: block.id, entryIndex: i, seq: entries[i].seq })
    }
  }
  return anchors
}

/**
 * Index of the surface entry whose assistant message carries one tool call.
 * @param {{ message: object }[]} entries - surface entries.
 * @param {string} callId - tool-call id to locate.
 * @returns {number} entry index, or -1.
 */
export function findToolCallEntryIndex(entries, callId) {
  if (typeof callId !== 'string' || callId.length === 0) return -1
  for (let i = 0; i < entries.length; i++) {
    const content = entries[i].message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) {
      if (isToolCallBlock(block) && block.id === callId) return i
    }
  }
  return -1
}

/**
 * Validate one proposed anchor name. Throws a model-facing error.
 * @param {unknown} raw - model-supplied anchor value.
 * @returns {string} trimmed, accepted name.
 */
export function validateAnchorName(raw) {
  if (typeof raw !== 'string') throw new Error('add_anchor requires a string "anchor" parameter')
  const normalized = normalizeAnchorName(raw)
  if (normalized !== null) return normalized
  const name = raw.trim()
  if (name.length === 0) throw new Error('anchor name must not be empty')
  if (name.length > ANCHOR_NAME_MAX) throw new Error(`anchor name too long (${name.length} > ${ANCHOR_NAME_MAX} chars)`)
  if (/\s/u.test(name)) throw new Error('anchor name must not contain whitespace')
  if (name === SESSION_START_ANCHOR) {
    throw new Error(`"${SESSION_START_ANCHOR}" is reserved for the built-in anchor after the original user prompt`)
  }
  throw new Error('anchor name is invalid')
}

/**
 * Reject a name already used anywhere in this session (consumed names included).
 * @param {{ name: string, toolCallId: string }[]} recordedAnchors - all recorded anchors.
 * @param {string} name - proposed name.
 * @param {string} ownCallId - the in-flight add_anchor call id (excluded from the check).
 */
export function assertAnchorNameAvailable(recordedAnchors, name, ownCallId) {
  const duplicate = recordedAnchors.find((anchor) => anchor.name === name && anchor.toolCallId !== ownCallId)
  if (duplicate) {
    throw new Error(`anchor "${name}" already exists in this session; anchor names must be unique`)
  }
}
