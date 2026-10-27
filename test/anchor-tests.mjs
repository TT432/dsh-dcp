// Anchor model + compression core tests against a real detached dsh Session.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  firstUserEntryIndex,
  scanAllAnchorCalls,
  scanLiveAnchors,
  surfaceEntries,
  validateAnchorName,
} from '../lib/anchor/anchors.js'
import { compressFromAnchor, recordAnchor } from '../lib/anchor/compress.js'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { createSystemMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  appendAssistant,
  appendToolResult,
  buildAnchorScenario,
  createSession,
  derivedShape,
  entriesOf,
  tokenMeter,
  appendSystem,
  appendPrompt,
} from './helpers.mjs'

/** A logger that records nothing. */
const silentLogger = { info() {}, debug() {}, warn() {}, error() {}, saveContext() {} }

/** Scenario with one work step between the prompt and the anchor. */
function buildSplitScenario() {
  const session = createSession('split-session')
  appendSystem(session)
  appendPrompt(session)
  appendAssistant(session, { turn: 1, step: 1, text: 'EARLY WORK' })
  const anchorStep = appendAssistant(session, {
    turn: 2,
    step: 1,
    calls: [{ callId: 'call-anchor-1', name: 'add_anchor', args: { anchor: 'a1' } }],
  })
  appendToolResult(session, { turn: 2, step: 1, callId: 'call-anchor-1', callSeq: anchorStep.callSeqs.get('call-anchor-1') })
  appendAssistant(session, { turn: 3, step: 1, text: 'LATE WORK' })
  appendAssistant(session, {
    turn: 4,
    step: 1,
    calls: [{ callId: 'call-compress-1', name: 'compress', args: { startAnchor: 'a1', summary: 'S' } }],
  })
  return { session, compressCallId: 'call-compress-1' }
}

test('scanLiveAnchors：session_start + 真实锚点，消耗后消失', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.deepEqual(scanLiveAnchors(entriesOf(session)).map((a) => a.name), ['session_start', 'a1'])
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: compressCallId, logger: silentLogger })
  assert.deepEqual(scanLiveAnchors(entriesOf(session)).map((a) => a.name), ['session_start'])
})

test('firstUserEntryIndex：第一条 user 消息即原始 prompt', () => {
  const { session } = buildAnchorScenario()
  const entries = entriesOf(session)
  assert.equal(entries[firstUserEntryIndex(entries)].message.role, 'user')
  assert.equal(entries[firstUserEntryIndex(entries)].message.content[0].text, 'Original user prompt')
})

test('firstUserEntryIndex：优先 source.kind=user，session_start 保留真正 prompt', () => {
  const session = createSession('prompt-source-priority')
  appendSystem(session)
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: 'INJECTED CONTEXT' }],
    source: { kind: 'runtime-context' },
  }), { surfaceOp: 'append' })
  appendPrompt(session, 'REAL PROMPT')
  const entriesBefore = entriesOf(session)
  assert.equal(firstUserEntryIndex(entriesBefore), 2)

  const anchorStep = appendAssistant(session, {
    turn: 2,
    step: 1,
    calls: [{ callId: 'call-anchor-1', name: 'add_anchor', args: { anchor: 'a1' } }],
  })
  appendToolResult(session, { turn: 2, step: 1, callId: 'call-anchor-1', callSeq: anchorStep.callSeqs.get('call-anchor-1') })
  appendAssistant(session, { turn: 3, step: 1, text: 'WORK' })
  appendAssistant(session, {
    turn: 4,
    step: 1,
    calls: [{ callId: 'call-compress-1', name: 'compress', args: {} }],
  })
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'session_start',
    summary: 'S',
    force: true,
    callId: 'call-compress-1',
    logger: silentLogger,
  })
  assert.equal(result.shadowedMessages, 3)
  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'user', 'user', 'assistant'])
  assert.equal(shape[1].text, 'INJECTED CONTEXT')
  assert.equal(shape[2].text, 'REAL PROMPT')
  assert.equal(shape[3].text, '[Compressed: session_start → here]\nS')
})

test('validateAnchorName：空/空白/超长/保留名', () => {
  assert.equal(validateAnchorName(' a1 '), 'a1')
  assert.throws(() => validateAnchorName('   '), /must not be empty/)
  assert.throws(() => validateAnchorName('a 1'), /whitespace/)
  assert.throws(() => validateAnchorName('x'.repeat(65)), /too long/)
  assert.throws(() => validateAnchorName('session_start'), /reserved/)
  assert.throws(() => validateAnchorName(42), /string "anchor"/)
})

test('失败的非法 add_anchor 不会成为可用锚点或被压缩起点', () => {
  for (const badName of ['a 1', 'x'.repeat(65)]) {
    const { session, anchorCallId, compressCallId } = buildAnchorScenario({ anchorName: badName })
    assert.throws(() => recordAnchor({ session, anchor: badName, callId: anchorCallId }))
    assert.deepEqual(scanLiveAnchors(entriesOf(session)).map((a) => a.name), ['session_start'])
    assert.throws(
      () => compressFromAnchor({ session, tokenMeter, startAnchor: badName, summary: 'S', callId: compressCallId, logger: silentLogger }),
      /unknown startAnchor/,
    )
  }
})

test('recordAnchor：本调用自身不算重复，其他同名（含已消耗）拒绝', () => {
  const { session, anchorCallId, compressCallId } = buildAnchorScenario()
  assert.deepEqual(recordAnchor({ session, anchor: 'a1', callId: anchorCallId }), { anchor: 'a1' })
  assert.throws(() => recordAnchor({ session, anchor: 'a1', callId: 'call-other' }), /already exists/)
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: compressCallId, logger: silentLogger })
  // 消耗后的名字仍然占用（整会话唯一）
  assert.throws(() => recordAnchor({ session, anchor: 'a1', callId: 'call-later' }), /already exists/)
})

test('recordAnchor：run_code 子调用不能假成功创建锚点', () => {
  const { session } = buildAnchorScenario()
  assert.throws(
    () => recordAnchor({ session, anchor: 'brand-new', callId: 'ptc:call:1' }),
    /add_anchor must be a direct model tool call/,
  )
})

test('compress：区间遮蔽 + 摘要节点 + compress 调用保留', () => {
  const { session, compressCallId } = buildAnchorScenario()
  const before = session.seq
  const entriesBefore = entriesOf(session)
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a1',
    summary: 'WORK DONE: files changed, tests pass',
    callId: compressCallId,
    logger: silentLogger,
  })
  assert.equal(result.anchor, 'a1')
  assert.equal(result.shadowedMessages, 3) // anchor assistant + result + WORK
  assert.equal(result.absorbed, 0)
  assert.equal(session.seq, before + 2)

  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'user', 'assistant'])
  assert.equal(shape[1].text, 'Original user prompt')
  assert.equal(shape[2].source, 'dcp-compress')
  assert.equal(shape[2].text, '[Compressed: a1 → here]\nWORK DONE: files changed, tests pass')
  const compressMessage = session.deriveMessages().at(-1)
  assert.equal(compressMessage.content[0].type, 'tool-call')
  assert.equal(compressMessage.content[0].id, compressCallId)

  // 影子计价事件紧邻替换事件
  const events = session.snapshotEvents()
  const prune = events[before]
  const replacement = events[before + 1]
  assert.equal(prune.type, 'compaction/prune')
  assert.equal(replacement.type, 'user/message')
  assert.deepEqual(replacement.sourceEventSeqs, result.shadowedSeqs)
  assert.deepEqual(replacement.surfaceOp, {
    op: 'replace',
    startSeq: result.shadowedSeqs[0],
    endSeq: result.shadowedSeqs[result.shadowedSeqs.length - 1],
  })

  // 影子价格 = 被遮蔽消息的真实启发式价格
  const shadowedPrice = entriesBefore
    .filter((entry) => result.shadowedSeqs.includes(entry.seq))
    .reduce((sum, entry) => sum + estimateMessage(entry.message), 0)
  assert.equal(prune.data.shadowedTokenCount, shadowedPrice)
  assert.deepEqual(prune.data.shadowedSeqs, result.shadowedSeqs)
  assert.equal(result.freedTokens, Math.max(0, shadowedPrice - estimateMessage(replacement.data)))
})

test('compress：session_start 必须 force，且保留原始 prompt', () => {
  const { session, compressCallId } = buildSplitScenario()
  const before = session.seq
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: 'S', callId: compressCallId, logger: silentLogger }),
    /force=true/,
  )
  assert.equal(session.seq, before)

  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'session_start',
    summary: 'WHOLE SESSION',
    force: true,
    callId: compressCallId,
    logger: silentLogger,
  })
  assert.equal(result.shadowedMessages, 4) // EARLY WORK + anchor + result + LATE WORK
  assert.deepEqual(derivedShape(session).map((m) => m.role), ['system', 'user', 'user', 'assistant'])
  assert.equal(derivedShape(session)[1].text, 'Original user prompt')
  assert.equal(derivedShape(session)[2].text, '[Compressed: session_start → here]\nWHOLE SESSION')
})

test('compress：真实锚点起点不吞掉起点之前的工作', () => {
  const { session, compressCallId } = buildSplitScenario()
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'LATE', callId: compressCallId, logger: silentLogger })
  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'assistant', 'user', 'assistant'])
  assert.equal(shape[2].text, 'EARLY WORK')
  assert.equal(shape[3].text, '[Compressed: a1 → here]\nLATE')
})

test('compress：链式压缩累积摘要', () => {
  const { session } = buildAnchorScenario()
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'FIRST', callId: 'call-compress-1', logger: silentLogger })

  const anchor2 = appendAssistant(session, {
    turn: 4,
    step: 1,
    calls: [{ callId: 'call-anchor-2', name: 'add_anchor', args: { anchor: 'a2' } }],
  })
  appendToolResult(session, { turn: 4, step: 1, callId: 'call-anchor-2', callSeq: anchor2.callSeqs.get('call-anchor-2') })
  appendAssistant(session, { turn: 5, step: 1, text: 'SECOND WORK' })
  appendAssistant(session, {
    turn: 6,
    step: 1,
    calls: [{ callId: 'call-compress-2', name: 'compress', args: {} }],
  })

  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a2',
    summary: 'SECOND',
    callId: 'call-compress-2',
    logger: silentLogger,
  })
  assert.equal(result.absorbed, 0)
  const summaries = derivedShape(session).filter((m) => m.source === 'dcp-compress')
  assert.deepEqual(summaries.map((m) => m.text), [
    '[Compressed: a1 → here]\nFIRST',
    '[Compressed: a2 → here]\nSECOND',
  ])
})

test('compress：session_start 跨越先前压缩时吸收其摘要', () => {
  const { session } = buildAnchorScenario()
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'FIRST', callId: 'call-compress-1', logger: silentLogger })
  appendAssistant(session, { turn: 4, step: 1, text: 'MORE WORK' })
  appendAssistant(session, {
    turn: 5,
    step: 1,
    calls: [{ callId: 'call-compress-2', name: 'compress', args: {} }],
  })
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'session_start',
    summary: 'ALL',
    force: true,
    callId: 'call-compress-2',
    logger: silentLogger,
  })
  assert.equal(result.absorbed, 1)
  const summaries = derivedShape(session).filter((m) => m.source === 'dcp-compress')
  assert.deepEqual(summaries.map((m) => m.text), ['[Compressed: session_start → here]\nALL'])
})

test('compress：未知锚点列出可用锚点；已消耗锚点给出专门错误', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'nope', summary: 'S', callId: compressCallId, logger: silentLogger }),
    (error) => /unknown startAnchor "nope"/.test(error.message) && /"session_start", "a1"/.test(error.message),
  )
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: compressCallId, logger: silentLogger })
  appendAssistant(session, {
    turn: 4,
    step: 1,
    calls: [{ callId: 'call-compress-2', name: 'compress', args: {} }],
  })
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: 'call-compress-2', logger: silentLogger }),
    /was consumed by an earlier compression/,
  )
})

test('compress：空区间报错（锚点与 compress 同一 assistant 消息）', () => {
  const session = createSession('empty-range')
  appendSystem(session)
  appendPrompt(session)
  appendAssistant(session, {
    turn: 1,
    step: 1,
    calls: [
      { callId: 'call-anchor-1', name: 'add_anchor', args: { anchor: 'a1' } },
      { callId: 'call-compress-1', name: 'compress', args: {} },
    ],
  })
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: 'call-compress-1', logger: silentLogger }),
    /contains no messages to compress/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: 'S', force: true, callId: 'call-compress-1', logger: silentLogger }),
    /contains no messages to compress/,
  )
})

test('compress：空 system 节点也纳入完整 shadowedSeqs（消息计数不含它）', () => {
  const session = createSession('empty-system-node')
  appendSystem(session)
  appendPrompt(session)
  const anchorStep = appendAssistant(session, {
    turn: 1,
    step: 1,
    calls: [{ callId: 'call-anchor-1', name: 'add_anchor', args: { anchor: 'a1' } }],
  })
  appendToolResult(session, { turn: 1, step: 1, callId: 'call-anchor-1', callSeq: anchorStep.callSeqs.get('call-anchor-1') })
  appendAssistant(session, { turn: 2, step: 1, text: 'WORK' })
  // 空 system 节点不产生派生消息，但仍是一个表面节点，替换必须覆盖它。
  session.append('system/message', {
    turn: 3,
    step: 1,
    message: createSystemMessage(''),
  }, { surfaceOp: 'append' })
  appendAssistant(session, {
    turn: 3,
    step: 1,
    calls: [{ callId: 'call-compress-1', name: 'compress', args: {} }],
  })

  const before = session.seq
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a1',
    summary: 'S',
    callId: 'call-compress-1',
    logger: silentLogger,
  })
  assert.equal(result.shadowedMessages, 3)
  assert.equal(result.shadowedSeqs.length, 4)
  const replacement = session.snapshotEvents()[before + 1]
  assert.deepEqual(replacement.sourceEventSeqs, result.shadowedSeqs)
  assert.equal(replacement.surfaceOp.endSeq, result.shadowedSeqs[result.shadowedSeqs.length - 1])
  assert.deepEqual(derivedShape(session).map((m) => m.role), ['system', 'user', 'user', 'assistant'])
})

test('compress：非直接调用（run_code 子调用）报错', () => {
  const { session } = buildAnchorScenario()
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: 'ptc:call:1', logger: silentLogger }),
    /direct model tool call/,
  )
})

test('compress：区间跨过 system-prompt 更新时拒绝（不允许遮蔽 in-history 提示词）', () => {
  const session = createSession('system-span')
  appendSystem(session)
  appendPrompt(session)
  const anchorStep = appendAssistant(session, {
    turn: 1,
    step: 1,
    calls: [{ callId: 'call-anchor-1', name: 'add_anchor', args: { anchor: 'a1' } }],
  })
  appendToolResult(session, { turn: 1, step: 1, callId: 'call-anchor-1', callSeq: anchorStep.callSeqs.get('call-anchor-1') })
  session.append('system/message', {
    turn: 2,
    step: 1,
    message: createSystemMessage('UPDATED PROMPT'),
  }, { surfaceOp: 'append' })
  appendAssistant(session, {
    turn: 2,
    step: 1,
    calls: [{ callId: 'call-compress-1', name: 'compress', args: {} }],
  })
  const before = session.seq
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: 'call-compress-1', logger: silentLogger }),
    /crosses a system-prompt update/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: 'S', force: true, callId: 'call-compress-1', logger: silentLogger }),
    /crosses a system-prompt update/,
  )
  assert.equal(session.seq, before)
})

test('compress：参数校验与 tokenMeter 缺席', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: '', summary: 'S', callId: compressCallId, logger: silentLogger }),
    /non-empty string "startAnchor"/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: '  ', callId: compressCallId, logger: silentLogger }),
    /non-empty string "summary"/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter: undefined, startAnchor: 'a1', summary: 'S', callId: compressCallId, logger: silentLogger }),
    /token meter service/,
  )
})

test('compress：校验失败不产生任何事件', () => {
  const { session, compressCallId } = buildAnchorScenario()
  const before = session.seq
  assert.throws(() => compressFromAnchor({ session, tokenMeter, startAnchor: 'missing', summary: 'S', callId: compressCallId, logger: silentLogger }))
  assert.throws(() => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: 'S', callId: compressCallId, logger: silentLogger }))
  assert.equal(session.seq, before)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('scanAllAnchorCalls：包含已消耗锚点的全量记录', () => {
  const { session } = buildAnchorScenario()
  assert.deepEqual(scanAllAnchorCalls(session.snapshotEvents()).map((a) => a.name), ['a1'])
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: 'S', callId: 'call-compress-1', logger: silentLogger })
  assert.deepEqual(scanAllAnchorCalls(session.snapshotEvents()).map((a) => a.name), ['a1'])
})
