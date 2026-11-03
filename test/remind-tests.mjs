// Reminder tests: threshold parsing, budget resolution, estimation (CJK-aware
// fallback), and the agent/pre-step waterfall for the two triggers
// (token pressure, work-unit boundary) — the tool-call budget trigger was
// removed with omp 99a4f87.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  DEFAULT_REMIND_THRESHOLD,
  buildReminderMessage,
  estimateContextTokens,
  estimateTextTokens,
  finishedTodoSet,
  heuristicContextTokens,
  lastTokenReminder,
  lastUnitReminderId,
  parseRemindPolicy,
  parseRemindThreshold,
  registerReminder,
  remindBudget,
  resolveContextWindow,
  tokenBracket,
} from '../lib/anchor/remind.js'
import {
  appendAssistant,
  appendPrompt,
  appendRequestContext,
  appendSystem,
  appendToolResult,
  createSession,
} from './helpers.mjs'
import { surfaceEntries } from '../lib/anchor/anchors.js'

const silentLogger = { info() {}, debug() {}, warn() {}, error() {}, saveContext() {} }

/** Session with a prompt and a routed context window. */
function buildRemindSession({ contextWindow = 1000, work = 'some work text' } = {}) {
  const session = createSession('remind-session')
  appendSystem(session)
  appendPrompt(session)
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: work }],
    source: { kind: 'runtime-context' },
  }), { surfaceOp: 'append' })
  appendRequestContext(session, { contextWindow })
  return session
}

/** Append a `todo_write` call plus its rendered result. */
function appendTodoWrite(session, { callId, text, turn = 1, step = 1 }) {
  const assistant = appendAssistant(session, {
    turn,
    step,
    calls: [{ callId, name: 'todo_write', args: { todos: [] } }],
  })
  appendToolResult(session, { turn, step, callId, callSeq: assistant.callSeqs.get(callId), text })
}

/** Append a generic tool call plus its result. */
function appendToolCall(session, { callId, name = 'read', text = 'ok', turn = 1, step = 1 }) {
  const assistant = appendAssistant(session, {
    turn,
    step,
    calls: [{ callId, name, args: {} }],
  })
  appendToolResult(session, { turn, step, callId, callSeq: assistant.callSeqs.get(callId), text })
}

/** Fake plugin context capturing `on` registrations and serving tokenMeter. */
function fakeCtx(tokenMeter) {
  const listeners = []
  return {
    listeners,
    ctx: {
      on(name, handler) { listeners.push({ name, handler }) },
      get(name) { return name === 'tokenMeter' ? tokenMeter : undefined },
    },
  }
}

/** Invoke a registered pre-step handler with the given defaults. */
async function runPreStep(handler, { session, turn = 1, step = 2, messages = [] }) {
  const controller = new AbortController()
  return handler(
    { agent: { session }, turn, step, signal: controller.signal },
    async () => ({ kind: 'enter', messages }),
  )
}

/** One ordinary user message standing in for the step's fresh input. */
function userMessage() {
  return createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })
}

/** Persist an injected reminder the way the loop does (durable user message). */
function persistReminder(session, message) {
  session.append('user/message', message, { surfaceOp: 'append' })
}

test('parseRemindThreshold：缺省/数值/非法/关闭', () => {
  assert.equal(DEFAULT_REMIND_THRESHOLD, 0.4)
  assert.equal(parseRemindThreshold(undefined), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold(''), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold('abc'), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold('0.75'), 0.75)
  assert.equal(parseRemindThreshold('200000'), 200000)
  assert.equal(parseRemindThreshold('0'), 0)
  assert.equal(parseRemindThreshold('-1'), -1)
})

test('parseRemindPolicy：只剩 token 阈值（budget 已移除）', () => {
  assert.deepEqual(parseRemindPolicy({}), { threshold: 0.4 })
  assert.deepEqual(parseRemindPolicy({ DCP_REMIND_THRESHOLD: '0.5', DCP_REMIND_TOOLCALLS: '10' }), { threshold: 0.5 })
})

test('remindBudget：比例/绝对值/关闭/未知窗口', () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  assert.equal(remindBudget(0.6, session), 600)
  assert.equal(remindBudget(1234, session), 1234)
  assert.equal(remindBudget(0, session), undefined)
  assert.equal(remindBudget(-1, session), undefined)
  const windowless = createSession('windowless')
  appendSystem(windowless)
  appendPrompt(windowless)
  assert.equal(resolveContextWindow(windowless), undefined)
  assert.equal(remindBudget(0.6, windowless), undefined)
})

test('estimateContextTokens：tokenMeter 优先，缺失时用启发式', () => {
  const session = buildRemindSession()
  assert.equal(estimateContextTokens(session, { measure: () => ({ totalTokens: 4321 }) }), 4321)
  const fallback = estimateContextTokens(session, undefined)
  assert.ok(fallback > 0)
  assert.equal(fallback, heuristicContextTokens(session.deriveMessages()))
  // 抛错的 meter 也退化为启发式
  assert.equal(estimateContextTokens(session, { measure: () => { throw new Error('boom') } }), fallback)
})

test('estimateTextTokens：CJK 按 2 字符/token、ASCII 按 4 字符/token 分档', () => {
  assert.equal(estimateTextTokens('x'.repeat(400)), 100)
  assert.equal(estimateTextTokens('中'.repeat(400)), 200)
  assert.equal(estimateTextTokens('中x'.repeat(200)), 150) // 200 CJK / 2 + 200 ASCII / 4
  assert.equal(estimateTextTokens(''), 0)
  // 纯中文样本不会被 4 字符/token 低估 40%
  const chinese = createSession('cjk-heavy')
  sessionAppendText(chinese, '中'.repeat(400))
  assert.equal(heuristicContextTokens(chinese.deriveMessages()), 8 + 200)
})

/** Append one user text message (helper for the CJK estimate test). */
function sessionAppendText(session, text) {
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
}

test('buildReminderMessage：token 提醒列出锚点、注入 source kind', () => {
  const message = buildReminderMessage({ tokens: 900, window: 1000, anchors: ['a1'] })
  assert.equal(message.role, 'user')
  assert.equal(message.source.kind, 'dcp-reminder')
  assert.equal(message.source.tokens, 900)
  assert.equal(message.source.contextWindow, 1000)
  assert.equal(message.source.trigger, undefined)
  const text = message.content[0].text
  assert.ok(text.startsWith('[dcp:context] Context is at ~900 tokens'))
  assert.ok(text.includes('"a1"'))
  assert.ok(text.includes('call compress with startAnchor'))
  assert.ok(text.includes('session_start'))
  assert.ok(text.includes('recallable with expand'))
})

test('buildReminderMessage：unit 提醒带 unitCallId 与自己的 source', () => {
  const unit = buildReminderMessage({ anchors: ['a1'], trigger: 'unit', unitCallId: 'call-todo-1' })
  assert.ok(unit.content[0].text.startsWith('[dcp:unit]'))
  assert.ok(unit.content[0].text.includes('work-unit boundary'))
  assert.equal(unit.source.trigger, 'unit')
  assert.equal(unit.source.unitCallId, 'call-todo-1')
})

test('finishedTodoSet：dsh 与 omp 的完成措辞都识别，最新未完成则不触发', () => {
  const finishedTexts = [
    'Updated todo list: 0 pending, 0 in progress, 2 completed.',
    'All items complete',
    'Remaining items (0)',
    'Remaining items: none.',
    'Overall: 4/4 done, 0 open.',
  ]
  for (const text of finishedTexts) {
    const session = buildRemindSession()
    appendTodoWrite(session, { callId: 'call-todo-1', text })
    const unit = finishedTodoSet(surfaceEntries(session), session.snapshotEvents())
    assert.deepEqual(unit, { id: 'call-todo-1', text }, text)
  }

  const mid = buildRemindSession()
  appendTodoWrite(mid, { callId: 'call-todo-1', text: 'Updated todo list: 1 pending, 0 in progress, 1 completed.' })
  assert.equal(finishedTodoSet(surfaceEntries(mid), mid.snapshotEvents()), null)

  const none = buildRemindSession()
  appendTodoWrite(none, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 0 completed.' })
  assert.equal(finishedTodoSet(surfaceEntries(none), none.snapshotEvents()), null)
})

test('registerReminder：token 关闭时仍注册（unit 触发保留）', () => {
  const { ctx, listeners } = fakeCtx(undefined)
  registerReminder(ctx, { threshold: 0, logger: silentLogger })
  assert.deepEqual(listeners.map((entry) => entry.name), ['agent/pre-step', 'agent/turn-stopping'])
})

test('registerReminder：超预算注入提醒，同 turn 只一次，下个 turn 再提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })

  const first = await runPreStep(listeners[0].handler, { session, turn: 1, messages: [userMessage()] })
  assert.equal(first.messages.length, 2)
  assert.equal(first.messages[1].source.kind, 'dcp-reminder')

  const second = await runPreStep(listeners[0].handler, { session, turn: 1, messages: [userMessage()] })
  assert.equal(second.messages.length, 1)

  const nextTurn = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  assert.equal(nextTurn.messages.length, 2)
})

test('registerReminder：未超预算/未知窗口/reject 决策都不注入', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const under = await runPreStep(listeners[0].handler, { session, messages: [userMessage()] })
  assert.equal(under.messages.length, 1)

  const windowless = createSession('windowless-2')
  appendSystem(windowless)
  appendPrompt(windowless)
  const noWindow = await runPreStep(listeners[0].handler, { session: windowless, messages: [userMessage()] })
  assert.equal(noWindow.messages.length, 1)

  const rejected = await listeners[0].handler(
    { agent: { session }, turn: 1, step: 1, signal: new AbortController().signal },
    async () => ({ kind: 'reject' }),
  )
  assert.equal(rejected.kind, 'reject')
})

test('registerReminder：step 1 无消息不复活已完成的 turn', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 1, step: 1, messages: [] })
  assert.equal(decision.messages.length, 0)
})

test('registerReminder：本 step 新消息计入阈值，避免晚一拍', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 90 }) })
  registerReminder(ctx, { threshold: 0.1, logger: silentLogger })
  const bigMessage = createUserMessage({
    content: [{ type: 'text', text: 'x'.repeat(400) }],
    source: { kind: 'user' },
  })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 1, messages: [bigMessage] })
  assert.equal(decision.messages.length, 2)
  assert.equal(decision.messages[1].source.kind, 'dcp-reminder')
})

test('registerReminder：turn-stopping 后空消息不复活 turn，标记只消费一次', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const stopping = listeners.find((entry) => entry.name === 'agent/turn-stopping').handler
  stopping({ agent: { session }, turn: 7, signal: new AbortController().signal })

  const closing = await runPreStep(listeners[0].handler, { session, turn: 7, messages: [] })
  assert.equal(closing.messages.length, 0)

  // 标记已消费：若之后确有继续执行的 step，不应对其永久屏蔽提醒。
  const continued = await runPreStep(listeners[0].handler, { session, turn: 7, messages: [] })
  assert.equal(continued.messages.length, 1)
})

test('unit 提醒：todo 全部完成时注入（token 未超）', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 2 completed.' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  const reminder = decision.messages.at(-1)
  assert.equal(reminder.source.trigger, 'unit')
  assert.equal(reminder.source.unitCallId, 'call-todo-1')
  assert.ok(reminder.content[0].text.startsWith('[dcp:unit]'))
})

test('unit 提醒：同一完成结果只提醒一次，新 todo 调用后重新提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 2 completed.' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })

  const first = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  assert.equal(first.messages.at(-1).source.trigger, 'unit')
  persistReminder(session, first.messages.at(-1))

  // 同一个完成的 todo 结果仍是「最新」，但没有新结果 → 不再重复。
  const same = await runPreStep(listeners[0].handler, { session, turn: 3, messages: [userMessage()] })
  assert.equal(same.messages.length, 1)

  // 模型再次调用 todo_write 并完成 → 新的 toolCallId，重新提醒。
  appendTodoWrite(session, {
    callId: 'call-todo-2',
    text: 'All items complete',
    turn: 3,
    step: 1,
  })
  const next = await runPreStep(listeners[0].handler, { session, turn: 4, messages: [userMessage()] })
  const reminder = next.messages.at(-1)
  assert.equal(reminder.source.trigger, 'unit')
  assert.equal(reminder.source.unitCallId, 'call-todo-2')
})

test('lastUnitReminderId：读取最近一次 unit 提醒的 toolCallId', () => {
  const session = buildRemindSession()
  assert.equal(lastUnitReminderId(session.snapshotEvents()), null)
  persistReminder(session, buildReminderMessage({ anchors: ['a1'], trigger: 'unit', unitCallId: 'call-todo-1' }))
  assert.equal(lastUnitReminderId(session.snapshotEvents()), 'call-todo-1')
  persistReminder(session, buildReminderMessage({ tokens: 900, window: 1000, anchors: ['a1'] }))
  assert.equal(lastUnitReminderId(session.snapshotEvents()), 'call-todo-1')
})

test('优先级：token 压力优先于 unit 边界', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 1 completed.' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  const reminder = decision.messages.at(-1)
  assert.ok(reminder.content[0].text.startsWith('[dcp:context]'))
  assert.equal(reminder.source.trigger, undefined)
})

test('tokenBracket：按窗口（或绝对预算）的 10% 分档', () => {
  assert.equal(tokenBracket(500, 500, 1000), 5)
  assert.equal(tokenBracket(999, 500, 1000), 9)
  assert.equal(tokenBracket(1000, 500, 1000), 10)
  // 无窗口：以绝对预算为基准
  assert.equal(tokenBracket(1000, 1000, undefined), 10)
})

test('lastTokenReminder：读取最近 token 提醒；unit 提醒不算，后续压缩会重置', () => {
  const session = buildRemindSession()
  persistReminder(session, buildReminderMessage({ tokens: 900, window: 1000, anchors: ['a1'] }))
  assert.equal(lastTokenReminder(surfaceEntries(session)).tokens, 900)
  // unit 提醒不参与档位
  persistReminder(session, buildReminderMessage({ anchors: ['a1'], trigger: 'unit', unitCallId: 'call-todo-1' }))
  assert.equal(lastTokenReminder(surfaceEntries(session)).tokens, 900)
  // 其后发生压缩 → 重置
  appendToolCall(session, { callId: 'call-compress-1', name: 'compress' })
  assert.equal(lastTokenReminder(surfaceEntries(session)), undefined)
})

test('token 档位：同一 10% 档跨 turn 不重复提醒，跨档后再次提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const meter = { tokens: 900, measure() { return { totalTokens: meter.tokens } } }
  const { ctx, listeners } = fakeCtx(meter)
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const first = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  assert.equal(first.messages.length, 2)
  persistReminder(session, first.messages[1])

  const same = await runPreStep(listeners[0].handler, { session, turn: 3, messages: [userMessage()] })
  assert.equal(same.messages.length, 1)

  meter.tokens = 1000
  const higher = await runPreStep(listeners[0].handler, { session, turn: 4, messages: [userMessage()] })
  assert.equal(higher.messages.length, 2)
  assert.ok(higher.messages[1].content[0].text.startsWith('[dcp:context]'))
})

test('token 档位：压缩后或估算回落时重置', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const meter = { tokens: 900, measure() { return { totalTokens: meter.tokens } } }
  const { ctx, listeners } = fakeCtx(meter)
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const first = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  persistReminder(session, first.messages[1])

  // 提醒之后的 compress 调用 → 档位重置
  appendToolCall(session, { callId: 'call-compress-1', name: 'compress' })
  const afterCompress = await runPreStep(listeners[0].handler, { session, turn: 3, messages: [userMessage()] })
  assert.equal(afterCompress.messages.length, 2)
  persistReminder(session, afterCompress.messages[1])

  // 估算低于上次提醒 → 视为回落，重置
  meter.tokens = 800
  const afterDrop = await runPreStep(listeners[0].handler, { session, turn: 4, messages: [userMessage()] })
  assert.equal(afterDrop.messages.length, 2)
})
