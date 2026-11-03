// Shared fixture builders for the anchor-compression tests.
//
// Messages/events are constructed with the real dsh packages (0.2.1-rc.3) and
// appended to a detached Session, so surface derivation, replacement
// validation, and message shaping are the production code paths.
import { Session } from '@deepseek-ai/dsh-session'
import {
  createAssistantMessage,
  createSystemMessage,
  createToolResultMessage,
  createUserMessage,
} from '@deepseek-ai/dsh-llm'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'

/** Fake service face matching ctx.tokenMeter for pricing. */
export const tokenMeter = { estimateMessage }

/** Slot headers of the recommended summary schema, in order. */
export const SUMMARY_SLOTS = ['当前任务', '已完成', '未完成', '约束', '文件与命令', '在跑作业与句柄']

/**
 * Build a well-shaped slot summary. Every slot defaults to `- 无` except
 * 当前任务 (the recommended minimum); override individual slot bodies to
 * exercise the advisory parser or the ancestor index.
 * @param {Record<string, string>} [overrides] - slot name → bullet block.
 * @returns {string} summary text.
 */
export function slotSummary(overrides = {}) {
  const defaults = {
    '当前任务': '- 测试上下文压缩 证据: test/helpers.mjs:1',
    '已完成': '- 无',
    '未完成': '- 无',
    '约束': '- 无',
    '文件与命令': '- 无',
    '在跑作业与句柄': '- 无',
  }
  const merged = { ...defaults, ...overrides }
  return SUMMARY_SLOTS.map((slot) => `## ${slot}\n${merged[slot]}`).join('\n')
}

/** Create a detached session for tests. */
export function createSession(id = 'dcp-test-session') {
  return Session.create(id)
}

/** Append the rendered system prompt (surface node 0). */
export function appendSystem(session, text = 'SYSTEM PROMPT') {
  return session.append('system/message', { turn: 0, step: 0, message: createSystemMessage(text) }, { surfaceOp: 'append' })
}

/** Append the original user prompt. */
export function appendPrompt(session, text = 'Original user prompt', turn = 1) {
  return session.append('user/message', createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
}

/**
 * Append one assistant message (optionally with tool calls) plus the log-only
 * `tool/call` events the loop records before dispatch.
 * @returns {{ seq: number, callSeqs: Map<string, number> }}
 */
export function appendAssistant(session, { turn, step, text, calls = [] }) {
  const content = []
  if (text !== undefined) content.push({ type: 'text', text })
  for (const call of calls) {
    content.push({ type: 'tool-call', id: call.callId, name: call.name, arguments: JSON.stringify(call.args ?? {}) })
  }
  const event = session.append('assistant/message', {
    turn,
    step,
    message: createAssistantMessage({ content, source: { provider: 'test-provider', model: 'test-model' } }),
    stream: [],
    usage: { inputTokens: 100, outputTokens: 10 },
  }, { surfaceOp: 'append' })
  const callSeqs = new Map()
  for (const call of calls) {
    const callSeq = session.append('tool/call', {
      turn,
      step,
      callId: call.callId,
      name: call.name,
      arguments: JSON.stringify(call.args ?? {}),
    }).seq
    callSeqs.set(call.callId, callSeq)
  }
  return { seq: event.seq, callSeqs }
}

/** Append one tool result answering a previously recorded call. */
export function appendToolResult(session, { turn, step, callId, callSeq, text = 'ok' }) {
  return session.append('tool/result', {
    turn,
    step,
    message: createToolResultMessage({ callId, content: [{ type: 'text', text }], isError: false }),
  }, { surfaceOp: 'append', sourceEventSeqs: [callSeq] })
}

/** Append one request/context record (route metadata, including the window). */
export function appendRequestContext(session, { provider = 'test-provider', model = 'test-model', contextWindow } = {}) {
  return session.append('request/context', {
    provider,
    model,
    ...(contextWindow === undefined ? {} : { contextWindow }),
  })
}

/**
 * Build the canonical omp-style scenario:
 *   system · prompt · [step1: add_anchor(callId) + result] · [step2: WORK] ·
 *   [step3: compress(callId)]
 * @returns {{ session, anchorCallId, compressCallId, compressEntrySeq }}
 */
export function buildAnchorScenario({
  session = createSession(),
  anchorName = 'a1',
  anchorCallId = 'call-anchor-1',
  compressCallId = 'call-compress-1',
  work = 'WORK DETAILS',
} = {}) {
  appendSystem(session)
  appendPrompt(session)

  const anchorStep = appendAssistant(session, {
    turn: 1,
    step: 1,
    text: 'Placing an anchor.',
    calls: [{ callId: anchorCallId, name: 'add_anchor', args: { anchor: anchorName } }],
  })
  appendToolResult(session, {
    turn: 1,
    step: 1,
    callId: anchorCallId,
    callSeq: anchorStep.callSeqs.get(anchorCallId),
    text: `anchor ${anchorName}`,
  })

  appendAssistant(session, { turn: 2, step: 1, text: work })

  const compressStep = appendAssistant(session, {
    turn: 3,
    step: 1,
    calls: [{ callId: compressCallId, name: 'compress', args: { startAnchor: anchorName, summary: slotSummary() } }],
  })

  return { session, anchorCallId, compressCallId, compressEntrySeq: compressStep.seq }
}

/** Surface entry (seq + derived message) list, mirroring lib/anchor/anchors.js. */
export function entriesOf(session) {
  const events = session.snapshotEvents()
  const bySeq = new Map(events.map((event) => [event.seq, event]))
  const out = []
  for (const seq of session.surface.nodes) {
    const event = bySeq.get(seq)
    if (event === undefined) continue
    const message = session.deriveEventMessage(event)
    if (message === null || message === undefined) continue
    out.push({ seq, event, message })
  }
  return out
}

/** Derived message texts keyed by role for quick assertions. */
export function derivedShape(session) {
  return session.deriveMessages().map((message) => ({
    role: message.role,
    source: message.source.kind,
    text: message.content.map((block) => block.text ?? block.arguments ?? '').join(''),
  }))
}
