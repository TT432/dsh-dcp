// Over-threshold reminder tests: threshold parsing, budget resolution,
// estimation fallback, and the agent/pre-step waterfall behavior.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  DEFAULT_REMIND_THRESHOLD,
  buildReminderMessage,
  estimateContextTokens,
  heuristicContextTokens,
  parseRemindThreshold,
  registerReminder,
  remindBudget,
  resolveContextWindow,
} from '../lib/anchor/remind.js'
import { appendRequestContext, appendSystem, appendPrompt, createSession } from './helpers.mjs'

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
async function runPreStep(handler, { session, turn = 1, step = 1, messages = [] }) {
  const controller = new AbortController()
  return handler(
    { agent: { session }, turn, step, signal: controller.signal },
    async () => ({ kind: 'enter', messages }),
  )
}

test('parseRemindThreshold：缺省/数值/非法/关闭', () => {
  assert.equal(parseRemindThreshold(undefined), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold(''), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold('abc'), DEFAULT_REMIND_THRESHOLD)
  assert.equal(parseRemindThreshold('0.75'), 0.75)
  assert.equal(parseRemindThreshold('200000'), 200000)
  assert.equal(parseRemindThreshold('0'), 0)
  assert.equal(parseRemindThreshold('-1'), -1)
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

test('buildReminderMessage：列出锚点、注入 source kind', () => {
  const session = buildRemindSession()
  const message = buildReminderMessage({ tokens: 900, window: 1000, anchors: ['session_start', 'a1'] })
  assert.equal(message.role, 'user')
  assert.equal(message.source.kind, 'dcp-reminder')
  assert.equal(message.source.tokens, 900)
  assert.equal(message.source.contextWindow, 1000)
  const text = message.content[0].text
  assert.ok(text.startsWith('[dcp:context] Context is at ~900 tokens'))
  assert.ok(text.includes('"session_start", "a1"'))
  assert.ok(text.includes('call compress with startAnchor'))
})

test('registerReminder：阈值 <= 0 时不注册监听', () => {
  const { ctx, listeners } = fakeCtx(undefined)
  registerReminder(ctx, { threshold: 0, logger: silentLogger })
  assert.equal(listeners.length, 0)
})

test('registerReminder：超预算注入提醒，同 turn 只一次，下个 turn 再提醒', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 900 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  assert.equal(listeners.length, 2)
  assert.equal(listeners[0].name, 'agent/pre-step')
  assert.equal(listeners[1].name, 'agent/turn-stopping')

  const userMessage = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })
  const first = await runPreStep(listeners[0].handler, { session, turn: 1, messages: [userMessage] })
  assert.equal(first.messages.length, 2)
  assert.equal(first.messages[1].source.kind, 'dcp-reminder')

  const second = await runPreStep(listeners[0].handler, { session, turn: 1, messages: [userMessage] })
  assert.equal(second.messages.length, 1)

  const nextTurn = await runPreStep(listeners[0].handler, { session, turn: 2, messages: [userMessage] })
  assert.equal(nextTurn.messages.length, 2)
})

test('registerReminder：未超预算/未知窗口/reject 决策都不注入', async () => {
  const session = buildRemindSession({ contextWindow: 1000 })
  const { ctx, listeners } = fakeCtx({ measure: () => ({ totalTokens: 100 }) })
  registerReminder(ctx, { threshold: 0.5, logger: silentLogger })
  const userMessage = createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })
  const under = await runPreStep(listeners[0].handler, { session, messages: [userMessage] })
  assert.equal(under.messages.length, 1)

  const windowless = createSession('windowless-2')
  appendSystem(windowless)
  appendPrompt(windowless)
  const noWindow = await runPreStep(listeners[0].handler, { session: windowless, messages: [userMessage] })
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

  const closing = await runPreStep(listeners[0].handler, { session, turn: 7, step: 2, messages: [] })
  assert.equal(closing.messages.length, 0)

  // 标记已消费：若之后确有继续执行的 step，不应对其永久屏蔽提醒。
  const continued = await runPreStep(listeners[0].handler, { session, turn: 7, step: 3, messages: [] })
  assert.equal(continued.messages.length, 1)
  assert.equal(continued.messages[0].source.kind, 'dcp-reminder')
})
