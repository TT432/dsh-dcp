/**
 * Compress reminders.
 *
 * Compression is model-driven: nothing forces a `compress`, and dsh's
 * automatic compaction is a separate, less targeted mechanism. A model that
 * simply forgets walks the context past the window (or lets auto-compaction
 * discard working context). Three triggers append one synthetic user message
 * at a step boundary:
 *
 *   1. `[dcp:context]` — estimated context crosses the token budget
 *      (`DCP_REMIND_THRESHOLD`).
 *   2. `[dcp:unit]` — the newest `todo_write` result reports a finished item
 *      set (0 pending, 0 in progress, >0 completed). That is the work-unit
 *      boundary; the trigger makes it observable instead of judged.
 *   3. `[dcp:budget]` — N tool calls have run since the last compress
 *      (`DCP_REMIND_TOOLCALLS`, default 40; observed unit size in omp was
 *      ~36 calls/todo item over 20 sessions).
 *
 * Only one reminder is appended per session turn (dsh logs the reminder
 * message, unlike omp whose reminder lives on the request copy only); token
 * pressure wins over the unit boundary, which wins over the call budget.
 * Token and unit triggers recompute from the live surface, so a compression
 * silences them; the budget trigger resets because the compress call itself
 * becomes the new baseline.
 *
 * Threshold (`DCP_REMIND_THRESHOLD`):
 *   0 < v < 1  → fraction of the routed model's context window (default 0.4);
 *   v >= 1     → absolute token count;
 *   v <= 0     → token trigger disabled.
 *
 * The 0.4 default fires before dsh's automatic compaction, whose pressure
 * budget is about `0.8 × (window − reserved output − 64k headroom)`.
 *
 * This module is the dsh port of `~/.omp/agent/extensions/dcp/src/remind.ts`
 * (commits 5954be4 + 3caea33 + 4d7c8ce + c83eeaa). The durable-watermark
 * mechanism maps to "scan the surface for the newest compress call": in dsh a
 * compression must be a direct model tool call (run_code sub-dispatches are
 * rejected), so the call is always present in the log — the failure mode that
 * forced omp's watermark (eval-kernel compressions leaving no stream marker)
 * cannot occur here. The 10%-bracket cursor also falls away: a durable
 * reminder is already throttled to one per turn, which is stricter than a
 * per-request bracket.
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { isToolCallBlock, readEvents, scanLiveAnchors, surfaceEntries } from './anchors.js'
import {
  BUDGET_MARKER,
  COMPRESS_TOOL,
  REMIND_MARKER,
  REMINDER_SOURCE_KIND,
  UNIT_MARKER,
  formatBudgetReminder,
  formatReminder,
  formatUnitReminder,
} from './prompts.js'

/** Default fraction of the context window. */
export const DEFAULT_REMIND_THRESHOLD = 0.4

/** Default tool calls between compressions. */
export const DEFAULT_TOOLCALL_BUDGET = 40

const CHARS_PER_TOKEN = 4

/**
 * Parse `DCP_REMIND_THRESHOLD`.
 * @param {unknown} raw - raw environment value.
 * @returns {number} effective threshold (default when unset or non-numeric).
 */
export function parseRemindThreshold(raw) {
  if (raw === undefined || raw === null) return DEFAULT_REMIND_THRESHOLD
  const text = String(raw).trim()
  if (text === '') return DEFAULT_REMIND_THRESHOLD
  const value = Number(text)
  return Number.isFinite(value) ? value : DEFAULT_REMIND_THRESHOLD
}

/**
 * Parse `DCP_REMIND_TOOLCALLS`.
 * @param {unknown} raw - raw environment value.
 * @returns {number} effective budget (default when unset or non-numeric).
 */
export function parseToolcallBudget(raw) {
  if (raw === undefined || raw === null) return DEFAULT_TOOLCALL_BUDGET
  const text = String(raw).trim()
  if (text === '') return DEFAULT_TOOLCALL_BUDGET
  const value = Number(text)
  return Number.isFinite(value) ? value : DEFAULT_TOOLCALL_BUDGET
}

/**
 * Parse the reminder policy from the environment.
 * @param {Record<string, string | undefined>} env - process environment.
 * @returns {{ threshold: number, toolCallBudget: number }} policy.
 */
export function parseRemindPolicy(env) {
  return {
    threshold: parseRemindThreshold(env?.DCP_REMIND_THRESHOLD),
    toolCallBudget: parseToolcallBudget(env?.DCP_REMIND_TOOLCALLS),
  }
}

/**
 * Context window of the most recently routed request, when known.
 * @param {object} session - live dsh Session.
 * @returns {number | undefined} positive token capacity.
 */
export function resolveContextWindow(session) {
  let context
  try {
    context = session.requestContext?.()
  } catch {
    context = undefined
  }
  const window = context?.contextWindow
  return typeof window === 'number' && window > 0 ? window : undefined
}

/**
 * Token budget that triggers the reminder, or undefined when unavailable.
 * @param {number} threshold - parsed threshold.
 * @param {object} session - live dsh Session.
 * @returns {number | undefined} budget in tokens.
 */
export function remindBudget(threshold, session) {
  if (!(threshold > 0)) return undefined
  if (threshold >= 1) return Math.floor(threshold)
  const window = resolveContextWindow(session)
  return window === undefined ? undefined : Math.floor(window * threshold)
}

/** Count one content block's characters (images/files price as zero). */
function blockChars(block) {
  if (typeof block?.text === 'string') return block.text.length
  if (block?.type === 'tool-call') {
    const args = typeof block.arguments === 'string' ? block.arguments.length : 0
    return (typeof block.name === 'string' ? block.name.length : 0) + args
  }
  return 0
}

/**
 * Character-based fallback estimate used only when the token meter is absent.
 * @param {readonly object[]} messages - derived model-visible messages.
 * @returns {number} estimated tokens.
 */
export function heuristicContextTokens(messages) {
  let chars = 0
  for (const message of messages) {
    chars += 8 // role and framing overhead
    const content = message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) chars += blockChars(block)
  }
  return Math.ceil(chars / CHARS_PER_TOKEN)
}

/**
 * Estimated token size of the current (already compressed) context.
 * @param {object} session - live dsh Session.
 * @param {object | undefined} tokenMeter - `ctx.tokenMeter`, when mounted.
 * @returns {number} estimated tokens.
 */
export function estimateContextTokens(session, tokenMeter) {
  if (typeof tokenMeter?.measure === 'function') {
    try {
      const measurement = tokenMeter.measure(session)
      if (measurement && Number.isFinite(measurement.totalTokens)) return measurement.totalTokens
    } catch {
      // fall through to the local heuristic
    }
  }
  try {
    return heuristicContextTokens(session.deriveMessages())
  } catch {
    return 0
  }
}

/** Number of model tool calls carried by one surface entry. */
function entryToolCalls(entry) {
  const content = entry?.message?.content
  if (!Array.isArray(content)) return 0
  let count = 0
  for (const block of content) if (isToolCallBlock(block)) count++
  return count
}

/**
 * Total tool calls on the current surface.
 * @param {readonly object[]} entries - surface entries.
 * @returns {number} call count.
 */
export function countToolCalls(entries) {
  let count = 0
  for (const entry of entries) count += entryToolCalls(entry)
  return count
}

/**
 * Tool calls issued since the last compression, measured from the newest
 * `compress` call on the surface.
 *
 * In dsh a compression must be a direct model tool call (`run_code`
 * sub-dispatches are rejected because the span end must be the call itself),
 * so its call block is always on the surface. The scan is therefore exact —
 * unlike omp, where eval-kernel compressions left no `write` block and forced
 * a durable watermark.
 *
 * @param {readonly object[]} entries - surface entries.
 * @returns {number} calls since the newest compress call.
 */
export function toolCallsSinceLastCompress(entries) {
  let lastCompress = -1
  for (let i = entries.length - 1; i >= 0; i--) {
    const content = entries[i]?.message?.content
    if (!Array.isArray(content)) continue
    if (content.some((block) => isToolCallBlock(block) && block.name === COMPRESS_TOOL)) {
      lastCompress = i
      break
    }
  }
  let count = 0
  for (let i = lastCompress + 1; i < entries.length; i++) count += entryToolCalls(entries[i])
  return count
}

/** Text of a tool-result message. */
function toolMessageText(message) {
  const content = message?.content
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  let text = ''
  for (const block of content) if (typeof block?.text === 'string') text += block.text
  return text
}

/**
 * Tool name for every recorded tool call in the session log.
 * @param {readonly object[]} events - committed session events.
 * @returns {Map<string, string>} call id → tool name.
 */
function toolNamesByCallId(events) {
  const names = new Map()
  for (const event of events) {
    if (event?.type !== 'assistant/message') continue
    const content = event.data?.message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) {
      if (isToolCallBlock(block)) names.set(block.id, block.name)
    }
  }
  return names
}

/**
 * Text of the newest `todo_write` result when it reports a finished item set,
 * else null. The newest todo result is the decision point: a mid-progress list
 * is not a boundary, and a todo list elided by a compression is not a
 * completion either.
 *
 * dsh's todo tool renders `Updated todo list: N pending, N in progress,
 * N completed.` (omp's wording differs).
 *
 * @param {readonly object[]} entries - surface entries.
 * @param {readonly object[]} events - committed session events (tool-call names).
 * @returns {string | null} the completion text, or null.
 */
export function finishedTodoSet(entries, events = []) {
  const names = toolNamesByCallId(events)
  for (let i = entries.length - 1; i >= 0; i--) {
    const message = entries[i]?.message
    if (message?.role !== 'tool') continue
    const callId = typeof message.toolCallId === 'string' ? message.toolCallId : message.source?.callId
    if (names.get(callId) !== 'todo_write') continue
    if (message.isError === true) return null
    const text = toolMessageText(message)
    const match = /Updated todo list:\s*(\d+)\s*pending,\s*(\d+)\s*in progress,\s*(\d+)\s*completed/i.exec(text)
    if (!match) return null
    const pending = Number(match[1])
    const inProgress = Number(match[2])
    const completed = Number(match[3])
    return pending === 0 && inProgress === 0 && completed > 0 ? text : null
  }
  return null
}

/**
 * Build a reminder message.
 * @param {{ tokens?: number, window?: number | undefined, anchors: string[], trigger?: 'context' | 'unit' | 'budget', used?: number, budget?: number }} input
 *        - reminder facts; `trigger` defaults to the token-pressure reminder.
 * @returns {object} identified user message.
 */
export function buildReminderMessage({ tokens, window, anchors, trigger = 'context', used, budget }) {
  if (trigger === 'unit') {
    return createUserMessage({
      content: [{ type: 'text', text: formatUnitReminder({ anchors }) }],
      source: { kind: REMINDER_SOURCE_KIND, trigger },
    })
  }
  if (trigger === 'budget') {
    return createUserMessage({
      content: [{ type: 'text', text: formatBudgetReminder({ used, budget, anchors }) }],
      source: { kind: REMINDER_SOURCE_KIND, trigger, used, budget },
    })
  }
  return createUserMessage({
    content: [{ type: 'text', text: formatReminder({ tokens, window, anchors }) }],
    source: {
      kind: REMINDER_SOURCE_KIND,
      ...(window === undefined ? {} : { contextWindow: window }),
      tokens,
    },
  })
}

/**
 * Register the step-boundary reminder waterfall.
 *
 * Calls `next()` first so the default decision (runtime-context injection)
 * survives, then appends at most one reminder to the messages entering the
 * step.
 *
 * @param {object} ctx - plugin context.
 * @param {{ threshold: number, toolCallBudget: number, logger: object }} options - parsed policy and logger.
 */
export function registerReminder(ctx, { threshold, toolCallBudget, logger }) {
  const tokenEnabled = threshold > 0
  if (!tokenEnabled && !(toolCallBudget > 0)) {
    logger.info('DCP reminder: token and tool-call triggers disabled; only the unit trigger remains', { threshold, toolCallBudget })
  }
  const lastRemindedTurn = new WeakMap()
  const stoppingTurns = new WeakMap()
  const windowWarned = new WeakSet()

  ctx.on('agent/pre-step', async (payload, next) => {
    const decision = await next()
    if (decision?.kind !== 'enter' || payload?.signal?.aborted === true) return decision
    // A step-1 wake with no messages would close the turn immediately; do not
    // resurrect it with a reminder.
    if (payload.step === 1 && decision.messages.length === 0) return decision
    const session = payload?.agent?.session
    if (session === undefined || session === null) return decision
    // turn-stopping has fired for this turn. With no fresh input the loop is
    // about to close: let it close instead of manufacturing another LLM step.
    // Consume the marker once so a continuation after steering is not blocked.
    if (stoppingTurns.get(session) === payload.turn) {
      stoppingTurns.delete(session)
      if (decision.messages.length === 0) return decision
    }
    if (lastRemindedTurn.get(session) === payload.turn) return decision

    let entries
    try {
      entries = surfaceEntries(session)
    } catch {
      return decision
    }
    if (entries.length === 0) return decision
    const anchors = scanLiveAnchors(entries).map((anchor) => anchor.name)
    const window = resolveContextWindow(session)

    // 1) Token pressure — recomputed on the current surface plus this step's
    //    incoming batch (its messages have not entered the durable surface yet).
    const budget = remindBudget(threshold, session)
    if (budget !== undefined) {
      const tokenMeter = typeof ctx.get === 'function' ? ctx.get('tokenMeter') : undefined
      const tokens = estimateContextTokens(session, tokenMeter) + heuristicContextTokens(decision.messages)
      if (tokens >= budget) {
        return inject(
          `${REMIND_MARKER} token pressure`,
          buildReminderMessage({ tokens, window, anchors }),
          { tokens, budget, window },
        )
      }
    } else if (tokenEnabled && !windowWarned.has(session)) {
      windowWarned.add(session)
      logger.debug('DCP reminder skipped: routed context window is unknown', { sessionId: session.id })
    }

    // 2) Work-unit boundary — the newest todo result reports a finished set.
    const events = readEvents(session)
    let completedTodo = null
    try {
      completedTodo = finishedTodoSet(entries, events)
    } catch {
      completedTodo = null
    }
    if (completedTodo !== null) {
      return inject(`${UNIT_MARKER} work-unit boundary`, buildReminderMessage({ anchors, trigger: 'unit' }), {})
    }

    // 3) Tool-call budget since the newest compress call.
    if (toolCallBudget > 0) {
      const used = toolCallsSinceLastCompress(entries)
      if (used >= toolCallBudget) {
        return inject(
          `${BUDGET_MARKER} tool-call budget`,
          buildReminderMessage({ anchors, trigger: 'budget', used, budget: toolCallBudget }),
          { used, budget: toolCallBudget },
        )
      }
    }

    // Mark the turn (one reminder per turn) and append the message.
    /**
     * @param {string} label - log label.
     * @param {object} message - reminder user message.
     * @param {object} details - extra log fields.
     * @returns {object} the augmented decision.
     */
    function inject(label, message, details) {
      lastRemindedTurn.set(session, payload.turn)
      logger.info(`DCP reminder injected: ${label}`, {
        sessionId: session.id,
        turn: payload.turn,
        step: payload.step,
        ...details,
      })
      return { ...decision, messages: [...decision.messages, message] }
    }

    return decision
  })

  // Only marks the boundary; never steers or changes the turn outcome.
  ctx.on('agent/turn-stopping', (payload) => {
    const session = payload?.agent?.session
    if (session !== undefined && session !== null) stoppingTurns.set(session, payload.turn)
  })
}
