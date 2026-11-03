// Tool wiring tests for add_anchor / compress / expand, plus plugin entry
// registration.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { validateJsonSchemaValue } from '@deepseek-ai/dsh-tools'
import { registerAnchorTools } from '../lib/tools/anchor.js'
import { apply } from '../lib/index.js'
import {
  appendAssistant,
  appendPrompt,
  appendSystem,
  buildAnchorScenario,
  createSession,
  derivedShape,
  slotSummary,
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

/** Mirror the runtime's post-execute output check. */
function assertOutput(tool, value) {
  assert.deepEqual(validateJsonSchemaValue(tool.output.schema, value, 'value'), [], tool.name)
}

/** Minimal session whose one assistant message carries the in-flight add_anchor call. */
function sessionWithPendingAnchor(callId = 'call-new', name = 'b1') {  const session = createSession('tool-anchor-session')
  appendSystem(session)
  appendPrompt(session)
  appendAssistant(session, {
    turn: 1,
    step: 1,
    calls: [{ callId, name: 'add_anchor', args: { anchor: name } }],
  })
  return session
}

test('registerAnchorTools：注册 add_anchor / compress / expand 的完整 schema', () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  assert.deepEqual(tools.map((tool) => tool.name), ['add_anchor', 'compress', 'expand'])

  const [addAnchor, compress, expand] = tools
  for (const tool of tools) {
    assert.equal(tool.parameters.type, 'object')
    assert.ok(tool.output.schema)
    assert.equal(typeof tool.output.render, 'function')
    assert.equal(typeof tool.execute, 'function')
  }
  assert.deepEqual(addAnchor.parameters.required, ['anchor'])
  assert.deepEqual(compress.parameters.required, ['startAnchor', 'summary'])
  // omp 99a4f87 删除 force：压缩参数里不再有闸门。
  assert.equal(compress.parameters.properties.force, undefined)
  assert.deepEqual(expand.parameters.required, ['block'])
  assert.equal(expand.parameters.properties.block.type, 'integer')
})

test('add_anchor：成功记录并渲染确认文本（NEXT span 语义）', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const session = sessionWithPendingAnchor('call-new', 'b1')
  const controller = new AbortController()
  const value = await tools[0].execute(
    { anchor: 'b1' },
    { callId: 'call-new', agent: { session }, signal: controller.signal },
  )
  assert.deepEqual(value, { anchor: 'b1' })
  assertOutput(tools[0], value)
  const [block] = tools[0].output.render({ anchor: 'b1' }, value)
  assert.equal(block.type, 'text')
  assert.ok(block.text.includes('Anchor "b1" recorded'))
  assert.ok(block.text.includes('NEXT span'))
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

test('compress：执行压缩并渲染结果（blockId + advisory）', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const summary = slotSummary({ '已完成': '- 完成工具接线 证据: lib/tools/anchor.js:1' })
  const value = await tools[1].execute(
    { startAnchor: 'a1', summary },
    { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
  )
  assert.equal(value.anchor, 'a1')
  assert.equal(value.blockId, 1)
  assert.equal(value.shadowedMessages, 3)
  assert.equal(value.absorbed, 0)
  assert.deepEqual(value.chain, [])
  assert.deepEqual(value.problems, [])
  assert.ok(value.shadowedTokens > 0)
  assert.ok(value.freedTokens >= 0 && value.freedTokens <= value.shadowedTokens)
  assertOutput(tools[1], value)
  const [block] = tools[1].output.render({}, value)
  assert.ok(block.text.includes('Compressed from "a1"'))
  assert.ok(block.text.includes('3 message(s) elided'))
  assert.ok(block.text.includes('as block 1'))
  assert.ok(block.text.includes(`~${value.freedTokens} tokens freed`))
  assert.ok(block.text.includes('stay visible'))
  assert.ok(!block.text.includes('摘要形状提示'))
  assert.equal(derivedShape(session).filter((m) => m.source === 'dcp-compress').length, 1)
})

test('compress：形状偏差随成功结果返回（已压缩，未拒绝）', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const value = await tools[1].execute(
    { startAnchor: 'a1', summary: '只有一句自由摘要' },
    { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
  )
  assert.ok(value.problems.length > 0)
  assertOutput(tools[1], value)
  const [block] = tools[1].output.render({}, value)
  assert.ok(block.text.includes('摘要形状提示（已压缩，未拒绝）'))
})

test('compress：session_start 不再需要 force', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const value = await tools[1].execute(
    { startAnchor: 'session_start', summary: slotSummary() },
    { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
  )
  assert.equal(value.anchor, 'session_start')
  assert.equal(value.blockId, 1)
  assertOutput(tools[1], value)
})

test('expand：取回压缩块全文并由 render 输出', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const summary = slotSummary()
  await tools[1].execute(
    { startAnchor: 'a1', summary },
    { callId: compressCallId, agent: { session }, signal: new AbortController().signal },
  )
  const value = await tools[2].execute(
    { block: 1 },
    { callId: 'call-expand-1', agent: { session }, signal: new AbortController().signal },
  )
  assert.equal(value.block, 1)
  assert.ok(value.content.includes('[block 1「a1」 — 全文]'))
  assert.ok(value.content.includes(summary))
  assertOutput(tools[2], value)
  const [block] = tools[2].output.render({}, value)
  assert.equal(block.text, value.content)
  await assert.rejects(
    tools[2].execute({ block: 9 }, { callId: 'call-expand-2', agent: { session }, signal: new AbortController().signal }),
    /unknown compression block 9/,
  )
})

test('compress：参数校验由 defineTool 执行', async () => {
  const { ctx, tools } = fakeCtx()
  registerAnchorTools(ctx, { logger: silentLogger })
  const { session, compressCallId } = buildAnchorScenario()
  const exec = { callId: compressCallId, agent: { session }, signal: new AbortController().signal }
  await assert.rejects(tools[1].execute({ startAnchor: 'a1' }, exec), /summary|required/i)
})

test('apply：注册四个工具与 pre-step / turn-stopping 监听', () => {
  const tools = []
  const listeners = []
  const ctx = {
    tools: { register: (tool) => tools.push(tool) },
    on(name, handler) { listeners.push({ name, handler }) },
    get(name) { return name === 'tokenMeter' ? tokenMeter : undefined },
    logger: () => ({ info() {}, debug() {}, warn() {}, error() {} }),
  }
  apply(ctx)
  assert.deepEqual(tools.map((tool) => tool.name).sort(), ['add_anchor', 'compress', 'dcp_compress', 'expand'])
  assert.deepEqual(listeners.map((entry) => entry.name).sort(), ['agent/pre-step', 'agent/turn-stopping'])
})
