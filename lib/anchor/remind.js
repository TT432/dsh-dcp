/**
 * Compress reminders.
 *
 * Compression is model-driven: nothing forces a `compress`, and dsh's
 * automatic compaction is a separate, less targeted mechanism. Two triggers
 * append one synthetic user message at a step boundary:
 *
 *   1. `[dcp:context]` — estimated context crosses the token budget
 *      (`DCP_REMIND_THRESHOLD`).
 *   2. `[dcp:unit]` — the newest `todo_write` result reports a finished item
 *      set. That is the work-unit boundary; the trigger makes it observable
 *      instead of judged. It fires once per finished result (deduped on the
 *      result's tool-call id, persisted in the reminder's `source`).
 *
 * There is deliberately no "N tool calls since the last compress" trigger
 * (omp `99a4f87` removed it): a call count is not a context cost. The removed
 * `[dcp:budget]` fired at 40 calls regardless of context size; in omp session
 * 01a19fdd that meant a compression three minutes in, on a context nowhere near
 * the window, after which the model re-read three files it had just read.
 *
 * Only one reminder is appended per session turn (dsh logs the reminder
 * message, unlike omp whose reminder lives on the request copy only); token
 * pressure wins over the unit boundary. Token and unit triggers recompute from
 * the live surface, so a compression silences them.
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
 * (commits 5954be4 + 3caea33 + 4d7c8ce + c83eeaa + 99a4f87). omp persists
 * `remindLevel` / `unitRemindId` in `dcp-state` because its reminders are
 * recomputed per request and never persisted. dsh's reminders *are* durable
 * `user/message` events, so both watermarks are read back from the log instead
 * of adding a state store: the 10% bracket from the latest token reminder's
 * estimate (a later compression resets it), the unit watermark from the latest
 * unit reminder's `unitCallId`.
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { isToolCallBlock, readEvents, scanLiveAnchors, surfaceEntries } from './anchors.js'
import {
  COMPRESS_TOOL,
  REMINDER_SOURCE_KIND,
  REMIND_MARKER,
  UNIT_MARKER,
  formatReminder,
  formatUnitReminder,
} from './prompts.js'

/** Default fraction of the context window. */
export const DEFAULT_REMIND_THRESHOLD = 0.4

const CHARS_PER_TOKEN = 4
/** CJK text costs roughly twice as many tokens per character as ASCII. */
const CJK_CHARS_PER_TOKEN = 2

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
 * Parse the reminder policy from the environment.
 * @param {Record<string, string | undefined>} env - process environment.
 * @returns {{ threshold: number }} policy.
 */
export function parseRemindPolicy(env) {
  return { threshold: parseRemindThreshold(env?.DCP_REMIND_THRESHOLD) }
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

/** Whether one UTF-16 code unit belongs to a dense CJK block. */
function isCjk(code) {
  return code >= 0x2e80 && code <= 0x9fff
}

/**
 * Token estimate for one string, weighting CJK characters differently:
 * a single 4-chars/token constant under-counts Chinese-heavy sessions by ~40%
 * (omp measured 2.5 chars/token on the first uncompressed sample of 01a19fdd).
 *
 * @param {string} text - text to estimate.
 * @returns {number} estimated tokens.
 */
export function estimateTextTokens(text) {
  let cjk = 0
  for (let i = 0; i < text.length; i++) if (isCjk(text.charCodeAt(i))) cjk++
  const rest = text.length - cjk
  return Math.ceil(cjk / CJK_CHARS_PER_TOKEN + rest / CHARS_PER_TOKEN)
}

/** Estimate one content block's tokens (images/files price as zero). */
function blockTokens(block) {
  if (typeof block?.text === 'string') return estimateTextTokens(block.text)
  if (block?.type === 'tool-call') {
    const args = typeof block.arguments === 'string' ? block.arguments : ''
    return estimateTextTokens(args) + (typeof block.name === 'string' ? estimateTextTokens(block.name) : 0)
  }
  return 0
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
 * Character-based fallback estimate used only when the token meter is absent.
 * @param {readonly object[]} messages - derived model-visible messages.
 * @returns {number} estimated tokens.
 */
export function heuristicContextTokens(messages) {
  let tokens = 0
  for (const message of messages) {
    tokens += 8 // role and framing overhead
    const content = message?.content
    if (!Array.isArray(content)) continue
    for (const block of content) tokens += blockTokens(block)
  }
  return tokens
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

/** Host wordings for a finished todo set (omp matches across hosts). */
const FINISHED_TODO_PATTERNS = [
  /All items complete/i,
  /Remaining items \(0\)/i,
  /Remaining items:\s*none/i,
  /\b0 open\b/i,
]

/**
 * Text of the newest `todo_write` result when it reports a finished item set,
 * else null. The newest todo result is the decision point: a mid-progress list
 * is not a boundary, and a todo list elided by a compression is not a
 * completion either.
 *
 * dsh's todo tool renders `Updated todo list: N pending, N in progress,
 * N completed.`; omp's hosts use `All items complete` / `Remaining items (0)` /
 * `Remaining items: none.` / `0 open`. All are matched (omp `99a4f87` fixed the
 * pattern because the old one only knew a wording the host no longer used).
 *
 * @param {readonly object[]} entries - surface entries.
 * @param {readonly object[]} events - committed session events (tool-call names).
 * @returns {{ id: string, text: string } | null} the finished result, or null.
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
    const id = typeof callId === 'string' && callId.length > 0 ? callId : text.slice(0, 32)

    const match = /Updated todo list:\s*(\d+)\s*pending,\s*(\d+)\s*in progress,\s*(\d+)\s*completed/i.exec(text)
    if (match) {
      const pending = Number(match[1])
      const inProgress = Number(match[2])
      const completed = Number(match[3])
      return pending === 0 && inProgress === 0 && completed > 0 ? { id, text } : null
    }
    if (FINISHED_TODO_PATTERNS.some((pattern) => pattern.test(text))) return { id, text }
    return null // newest todo result is mid-progress → no boundary
  }
  return null
}

/** Above the threshold, one token reminder per further 10% of the context. */
const REMIND_STEP_PERCENT = 10

/**
 * Bracket index for the current estimate: 10% of the context window per step.
 * Absolute thresholds step on 10% of the budget when no window is known
 * (mirrors omp's `tokenBracket`).
 * @param {number} tokens - estimated context tokens.
 * @param {number} budget - token budget in effect.
 * @param {number | undefined} window - routed context window, when known.
 * @returns {number} bracket index.
 */
export function tokenBracket(tokens, budget, window) {
  const base = window !== undefined && window > 0 ? window : budget
  return Math.floor((tokens / base) * (100 / REMIND_STEP_PERCENT))
}

/**
 * Most recent token-pressure reminder on the surface, or undefined when the
 * bracket may start over.
 *
 * omp persists `remindLevel` in `dcp-state` because its reminder is recomputed
 * on every request. dsh's reminder is a durable `user/message` event, so the
 * bracket is read back from the log instead of adding a state store; a
 * compression issued after the last reminder resets it (the context the
 * reminder complained about was pruned).
 *
 * @param {readonly object[]} entries - surface entries.
 * @returns {{ tokens: number, seq: number } | undefined} reminder estimate.
 */
export function lastTokenReminder(entries) {
  for (let i = entries.length - 1; i >= 0; i--) {
    const message = entries[i]?.message
    if (message?.role !== 'user') continue
    const source = message.source
    if (source?.kind !== REMINDER_SOURCE_KIND || source.trigger !== undefined) continue
    if (typeof source.tokens !== 'number' || !Number.isFinite(source.tokens)) continue
    for (let j = i + 1; j < entries.length; j++) {
      const content = entries[j]?.message?.content
      if (!Array.isArray(content)) continue
      if (content.some((block) => isToolCallBlock(block) && block.name === COMPRESS_TOOL)) return undefined
    }
    return { tokens: source.tokens, seq: entries[i].seq }
  }
  return undefined
}

/**
 * `unitCallId` of the most recent unit reminder recorded in the log, or null.
 *
 * The newest finished todo result stays "finished" until the model calls
 * `todo_write` again, so without this watermark `[dcp:unit]` would repeat on
 * every turn for the rest of the session (omp keeps `unitRemindId` in state for
 * the same reason).
 *
 * @param {readonly object[]} events - committed session events.
 * @returns {string | null} the tool-call id already reported, or null.
 */
export function lastUnitReminderId(events) {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i]
    if (event?.type !== 'user/message') continue
    const source = event.data?.source
    if (source?.kind !== REMINDER_SOURCE_KIND || source.trigger !== 'unit') continue
    return typeof source.unitCallId === 'string' && source.unitCallId.length > 0 ? source.unitCallId : null
  }
  return null
}

/**
 * Build a reminder message.
 * @param {{ tokens?: number, window?: number | undefined, anchors: string[], trigger?: 'context' | 'unit', unitCallId?: string }} input
 *        - reminder facts; `trigger` defaults to the token-pressure reminder.
 * @returns {object} identified user message.
 */
export function buildReminderMessage({ tokens, window, anchors, trigger = 'context', unitCallId }) {
  if (trigger === 'unit') {
    return createUserMessage({
      content: [{ type: 'text', text: formatUnitReminder({ anchors }) }],
      source: {
        kind: REMINDER_SOURCE_KIND,
        trigger,
        ...(typeof unitCallId === 'string' && unitCallId.length > 0 ? { unitCallId } : {}),
      },
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
 * @param {{ threshold: number, logger: object }} options - parsed policy and logger.
 */
export function registerReminder(ctx, { threshold, logger }) {
  const tokenEnabled = threshold > 0
  if (!tokenEnabled) {
    logger.info('DCP reminder: token trigger disabled; only the unit trigger remains', { threshold })
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
        const level = tokenBracket(tokens, budget, window)
        const previous = lastTokenReminder(entries)
        const previousLevel = previous === undefined ? 0 : tokenBracket(previous.tokens, budget, window)
        // A lower estimate than the last reminder means the context fell back
        // (a compression the scan could not attribute, a route/window change):
        // start the bracket over instead of suppressing the new growth.
        const reset = previous === undefined || tokens < previous.tokens
        if (reset || level > previousLevel) {
          return inject(
            `${REMIND_MARKER} token pressure`,
            buildReminderMessage({ tokens, window, anchors }),
            { tokens, budget, window, level },
          )
        }
        // Same 10% bracket as the last reminder: a durable reminder would only
        // pile up in the log, so let the unit trigger have this turn.
        logger.debug('DCP token reminder suppressed: same 10% bracket', {
          tokens,
          budget,
          level,
          previousTokens: previous.tokens,
        })
      }
    } else if (tokenEnabled && !windowWarned.has(session)) {
      windowWarned.add(session)
      logger.debug('DCP reminder skipped: routed context window is unknown', { sessionId: session.id })
    }

    // 2) Work-unit boundary — the newest todo result reports a finished set,
    //    and that result has not already been reported.
    const events = readEvents(session)
    let completedTodo = null
    try {
      completedTodo = finishedTodoSet(entries, events)
    } catch {
      completedTodo = null
    }
    if (completedTodo !== null && lastUnitReminderId(events) !== completedTodo.id) {
      return inject(
        `${UNIT_MARKER} work-unit boundary`,
        buildReminderMessage({ anchors, trigger: 'unit', unitCallId: completedTodo.id }),
        { unitCallId: completedTodo.id },
      )
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
