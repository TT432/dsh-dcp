/**
 * Over-threshold compress reminder.
 *
 * Compression is model-driven: nothing forces a `compress`, and dsh's
 * automatic compaction is a separate, less targeted mechanism. A model that
 * simply forgets walks the context past the window (or lets auto-compaction
 * discard working context). This module appends one synthetic user message at
 * a step boundary whenever the estimated context crosses the budget, naming
 * the live anchors and demanding an immediate compress.
 *
 * Unlike omp (whose reminder only exists on the outgoing request copy), a dsh
 * step's messages are logged, so this reminder is durable. It is therefore
 * throttled to at most one per session turn.
 *
 * Threshold (`DCP_REMIND_THRESHOLD`):
 *   0 < v < 1  → fraction of the routed model's context window (default 0.6);
 *   v >= 1     → absolute token count;
 *   v <= 0     → disabled.
 *
 * The 0.6 default fires before dsh's automatic compaction, whose pressure
 * budget is about `0.8 × (window − reserved output − 64k headroom)`.
 */

import { createUserMessage } from '@deepseek-ai/dsh-llm'
import { scanLiveAnchors, surfaceEntries } from './anchors.js'
import { REMINDER_SOURCE_KIND, formatReminder } from './prompts.js'

/** Default fraction of the context window. */
export const DEFAULT_REMIND_THRESHOLD = 0.6

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

/**
 * Build the reminder message.
 * @param {{ tokens: number, window: number | undefined, anchors: string[] }} input - reminder facts.
 * @returns {object} identified user message.
 */
export function buildReminderMessage({ tokens, window, anchors }) {
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
 * survives, then appends the reminder to the messages entering the step.
 *
 * @param {object} ctx - plugin context.
 * @param {{ threshold: number, logger: object }} options - parsed threshold and logger.
 */
export function registerReminder(ctx, { threshold, logger }) {
  if (!(threshold > 0)) {
    logger.info('DCP reminder disabled', { threshold })
    return
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

    const budget = remindBudget(threshold, session)
    if (budget === undefined) {
      if (!windowWarned.has(session)) {
        windowWarned.add(session)
        logger.debug('DCP reminder skipped: routed context window is unknown', { sessionId: session.id })
      }
      return decision
    }

    const tokenMeter = typeof ctx.get === 'function' ? ctx.get('tokenMeter') : undefined
    // The incoming batch has not entered the durable surface yet; include its
    // estimated size so the reminder fires for the request about to be sent,
    // not one step later.
    const tokens = estimateContextTokens(session, tokenMeter) + heuristicContextTokens(decision.messages)
    if (tokens < budget) return decision

    let entries
    try {
      entries = surfaceEntries(session)
    } catch {
      return decision
    }
    if (entries.length === 0) return decision
    const anchors = scanLiveAnchors(entries).map((anchor) => anchor.name)
    const window = resolveContextWindow(session)
    const message = buildReminderMessage({ tokens, window, anchors })
    lastRemindedTurn.set(session, payload.turn)
    logger.info('DCP over-threshold reminder injected', {
      sessionId: session.id,
      turn: payload.turn,
      step: payload.step,
      tokens,
      budget,
      window,
    })
    return { ...decision, messages: [...decision.messages, message] }
  })

  // Only marks the boundary; never steers or changes the turn outcome.
  ctx.on('agent/turn-stopping', (payload) => {
    const session = payload?.agent?.session
    if (session !== undefined && session !== null) stoppingTurns.set(session, payload.turn)
  })
}
