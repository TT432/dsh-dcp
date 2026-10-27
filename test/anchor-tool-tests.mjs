// Tool wiring tests for add_anchor / compress, plus plugin entry registration.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { registerAnchorTools } from '../lib/tools/anchor.js'
import { apply } from '../lib/index.js'
import {
  appendAssistant,
  appendPrompt,
  appendSystem,
  buildAnchorScenario,
  createSession,
  derivedShape,
  tokenMeter,
} from './helpers.mjs'

const silentLogger = { info() {}, debug() {}, warn() {}, error() {}, saveContext() {} }

/** Fake plugin context capturing registered tools. */
function fakeCtx() {
  const tools = []
  return {
    tools,
    ctx: {
      tools: { register: (tool) => tools.push(tool) },
      get(name) { return name === 'tokenMeter' ? tokenMeter : undefined },
    },
  }
}

/** Minimal session whose one assistant message carries the in-flight add_anchor call. */
function sessionWithPendingAnchor(callId = 'call-new', name = 'b1') {
  const session = createSession('tool-anchor-session')
  appendSystem(session)
  appendPrompt(session)
  appendAssistant(session, {
    turn: 1,
    step: 1,
    calls: [{ callId, name: 'add_anchor', args: { anchor: name } }],
  })
  return session
}

test('registerAnchorTools：注册 add_anchor 与 compress 的完整 schema', () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  assert.deepEqual(tools.map((tool) => tool.name), ['add_anchor', 'compress'])

  const [addAnchor, compress] = tools
  for (const tool of tools) {
    assert.equal(tool.parameters.type, 'object')
    assert.ok(tool.output.schema)
    assert.equal(typeof tool.output.render, 'function')
    assert.equal(typeof tool.execute, 'function')
  }
  assert.deepEqual(addAnchor.parameters.required, ['anchor'])
  assert.deepEqual(compress.parameters.required, ['startAnchor', 'summary'])
  assert.ok(compress.parameters.properties.force)
})

test('add_anchor：成功记录并渲染确认文本', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const session = sessionWithPendingAnchor('call-new', 'b1')
  const controller = new AbortController()
  const value = await tools[0].execute(
    { anchor: 'b1' },
    { callId: 'call-new', agent: { session }, signal: controller.signal },
  )
  assert.deepEqual(value, { anchor: 'b1' })
  const [block] = tools[0].output.render({ anchor: 'b1' }, value)
  assert.equal(block.type, 'text')
  assert.ok(block.text.includes('Anchor "b1" recorded'))
  assert.ok(block.text.includes('consumes the anchor'))
})

test('add_anchor：重名（含本次调用外的同名）与非法名报错', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const session = sessionWithPendingAnchor('call-new', 'b1')
  const exec = { callId: 'call-new', agent: { session }, signal: new AbortController().signal }
  await assert.rejects(tools[0].execute({ anchor: 'b 1' }, exec), /whitespace/)
  await assert.rejects(tools[0].execute({ anchor: 'session_start' }, exec), /reserved/)
  // 由另一个调用占用的名字
  await assert.rejects(tools[0].execute({ anchor: 'b1' }, { ...exec, callId: 'call-other' }), /already exists/)
})

test('compress：执行压缩并渲染结果', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const value = await tools[1].execute(
    { startAnchor: 'a1', summary: 'WORK DONE' },
    { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
  )
  assert.equal(value.anchor, 'a1')
  assert.equal(value.shadowedMessages, 3)
  assert.equal(value.absorbed, 0)
  assert.ok(value.shadowedTokens > 0)
  assert.ok(value.freedTokens >= 0 && value.freedTokens <= value.shadowedTokens)
  const [block] = tools[1].output.render({}, value)
  assert.ok(block.text.includes('Compressed from "a1"'))
  assert.ok(block.text.includes('3 message(s) elided'))
  assert.ok(block.text.includes(`~${value.freedTokens} tokens freed`))
  assert.ok(block.text.includes('stay visible'))
  assert.equal(derivedShape(session).filter((m) => m.source === 'dcp-compress').length, 1)
})

test('compress：session_start 缺 force 的报错透传给模型', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  await assert.rejects(
    tools[1].execute(
      { startAnchor: 'session_start', summary: 'ALL' },
      { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
    ),
    /force=true/,
  )
})

test('compress：参数校验由 defineTool 执行', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const exec = { callId: compressCallId, agent: { session }, signal: new AbortController().signal }
  await assert.rejects(tools[1].execute({ startAnchor: 'a1' }, exec), /summary|required/i)
})

test('apply：注册三个工具与 pre-step / turn-stopping 监听', () => {
  const tools = []
  const listeners = []
  const ctx = {
    tools: { register: (tool) => tools.push(tool) },
    on(name, handler) { listeners.push({ name, handler }) },
    get(name) { return name === 'tokenMeter' ? tokenMeter : undefined },
    logger: () => ({ info() {}, debug() {}, warn() {}, error() {} }),
  }
  apply(ctx)
  assert.deepEqual(tools.map((tool) => tool.name).sort(), ['add_anchor', 'compress', 'dcp_compress'])
  assert.deepEqual(listeners.map((entry) => entry.name).sort(), ['agent/pre-step', 'agent/turn-stopping'])
})
