// Reminder tests: threshold parsing, budget resolution, estimation fallback,
// and the agent/pre-step waterfall behavior for the three triggers
// (token pressure, work-unit boundary, tool-call budget).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  DEFAULT_REMIND_THRESHOLD,
  DEFAULT_TOOLCALL_BUDGET,
  buildReminderMessage,
  estimateContextTokens,
  finishedTodoSet,
  heuristicContextTokens,
  lastTokenReminder,
  parseRemindPolicy,
  parseRemindThreshold,
  parseToolcallBudget,
  registerReminder,
  remindBudget,
  resolveContextWindow,
  tokenBracket,
  toolCallsSinceLastCompress,
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

test('parseToolcallBudget / parseRemindPolicy：缺省 40，非法回落，关闭', () => {
  assert.equal(DEFAULT_TOOLCALL_BUDGET, 40)
  assert.equal(parseToolcallBudget(undefined), 40)
  assert.equal(parseToolcallBudget('abc'), 40)
  assert.equal(parseToolcallBudget('12'), 12)
  assert.equal(parseToolcallBudget('0'), 0)
  assert.deepEqual(parseRemindPolicy({}), { threshold: 0.4, toolCallBudget: 40 })
  assert.deepEqual(
    parseRemindPolicy({ DCP_REMIND_THRESHOLD: '0.5', DCP_REMIND_TOOLCALLS: '10' }),
    { threshold: 0.5, toolCallBudget: 10 },
  )
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

test('buildReminderMessage：token 提醒列出锚点、注入 source kind', () => {
  const message = buildReminderMessage({ tokens: 900, window: 1000, anchors: ['a1'] })
  assert.equal(message.role, 'user')
  assert.equal(message.source.kind, 'dcp-reminder')
  assert.equal(message.source.tokens, 900)
  assert.equal(message.source.contextWindow, 1000)
  const text = message.content[0].text
  assert.ok(text.startsWith('[dcp:context] Context is at ~900 tokens'))
  assert.ok(text.includes('"a1"'))
  assert.ok(text.includes('call compress with startAnchor'))
  assert.ok(text.includes('session_start'))
})

test('buildReminderMessage：unit / budget 提醒各带自己的标记与 source', () => {
  const unit = buildReminderMessage({ anchors: ['a1'], trigger: 'unit' })
  assert.ok(unit.content[0].text.startsWith('[dcp:unit]'))
  assert.equal(unit.source.trigger, 'unit')
  const budget = buildReminderMessage({ anchors: ['a1'], trigger: 'budget', used: 42, budget: 40 })
  assert.ok(budget.content[0].text.startsWith('[dcp:budget] 42 tool calls'))
  assert.equal(budget.source.trigger, 'budget')
  assert.equal(budget.source.used, 42)
  assert.equal(budget.source.budget, 40)
})

test('finishedTodoSet / toolCallsSinceLastCompress：基于表面扫描', async () => {
  const session = buildRemindSession()
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 2 completed.' })
  appendToolCall(session, { callId: 'call-read-1' })
  const entries = surfaceEntries(session)
  assert.ok(finishedTodoSet(entries, session.snapshotEvents()) !== null)
  assert.equal(toolCallsSinceLastCompress(entries), 2) // todo_write + read
})

test('registerReminder：token 与 toolcall 都关闭时仍注册（unit 触发保留）', () => {
  const { ctx, listeners } = fakeCtx(undefined)
  registerReminder(ctx, { threshold: 0, toolCallBudget: 0, logger: silentLogger })
  assert.deepEqual(listeners.map((entry) => entry.name), ['agent/pre-step', 'agent/turn-stopping'])
})

test('registerReminder：超预算注入提醒，同 turn 只一次，下个 turn 再提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })

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
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
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
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 1, step: 1, messages: [] })
  assert.equal(decision.messages.length, 0)
})

test('registerReminder：本 step 新消息计入阈值，避免晚一拍', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 90 }) })
  registerReminder(ctx, { threshold: 0.1, toolCallBudget: 0, logger: silentLogger })
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
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
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
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  const reminder = decision.messages.at(-1)
  assert.equal(reminder.source.trigger, 'unit')
  assert.ok(reminder.content[0].text.startsWith('[dcp:unit]'))
  assert.ok(reminder.content[0].text.includes('work-unit boundary'))
})

test('unit 提醒：最新 todo 仍未完成则不注入', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 1 pending, 0 in progress, 1 completed.' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  assert.equal(decision.messages.length, 1)
})

test('budget 提醒：自上次压缩以来的工具调用达到预算时注入', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendToolCall(session, { callId: 'call-read-1' })
  appendToolCall(session, { callId: 'call-read-2' })
  appendToolCall(session, { callId: 'call-read-3' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 2, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  const reminder = decision.messages.at(-1)
  assert.equal(reminder.source.trigger, 'budget')
  assert.equal(reminder.source.used, 3)
  assert.equal(reminder.source.budget, 2)
  assert.ok(reminder.content[0].text.startsWith('[dcp:budget] 3 tool calls'))
})

test('budget 提醒：compress 之后重新计数', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendToolCall(session, { callId: 'call-compress-1', name: 'compress' })
  appendToolCall(session, { callId: 'call-read-1' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 2, logger: silentLogger })
  const decision = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage()] })
  assert.equal(decision.messages.length, 1)
})

test('优先级：token 压力优先于 unit 边界', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  appendTodoWrite(session, { callId: 'call-todo-1', text: 'Updated todo list: 0 pending, 0 in progress, 1 completed.' })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
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

test('lastTokenReminder：读取最近 token 提醒；unit/budget 提醒不算，后续压缩会重置', () => {
  const session = buildRemindSession()
  persistReminder(session, buildReminderMessage({ tokens: 900, window: 1000, anchors: ['a1'] }))
  assert.equal(lastTokenReminder(surfaceEntries(session)).tokens, 900)
  // unit 提醒不参与档位
  persistReminder(session, buildReminderMessage({ anchors: ['a1'], trigger: 'unit' }))
  assert.equal(lastTokenReminder(surfaceEntries(session)).tokens, 900)
  // 其后发生压缩 → 重置
  appendToolCall(session, { callId: 'call-compress-1', name: 'compress' })
  assert.equal(lastTokenReminder(surfaceEntries(session)), undefined)
})

test('token 档位：同一 10% 档跨 turn 不重复提醒，跨档后再次提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const meter = { tokens: 900, measure() { return { totalTokens: meter.tokens } } }
  const { ctx, listeners } = fakeCtx(meter)
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
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
  registerReminder(ctx, { threshold: 0.5, toolCallBudget: 0, logger: silentLogger })
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
