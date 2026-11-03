// Anchor model + compression core tests against a real detached dsh Session.
//
// The chain semantics come from omp 99a4f87: absorbed blocks are not validated
// away, they stay in the log and become index lines under the new summary.
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
import { expandBlock, scanCompressionBlocks } from '../lib/anchor/blocks.js'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { createSystemMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  appendAssistant,
  appendToolResult,
  buildAnchorScenario,
  createSession,
  derivedShape,
  entriesOf,
  slotSummary,
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
    calls: [{ callId: 'call-compress-1', name: 'compress', args: { startAnchor: 'a1', summary: slotSummary() } }],
  })
  return { session, compressCallId: 'call-compress-1' }
}

/** Append a second anchor + work + a compress call, returning its id. */
function appendSecondCompression(session, { anchorName = 'a2', anchorCallId = 'call-anchor-2', compressCallId = 'call-compress-2' } = {}) {
  const anchorStep = appendAssistant(session, {
    turn: 10,
    step: 1,
    calls: [{ callId: anchorCallId, name: 'add_anchor', args: { anchor: anchorName } }],
  })
  appendToolResult(session, { turn: 10, step: 1, callId: anchorCallId, callSeq: anchorStep.callSeqs.get(anchorCallId) })
  appendAssistant(session, { turn: 11, step: 1, text: 'SECOND WORK' })
  appendAssistant(session, {
    turn: 12,
    step: 1,
    calls: [{ callId: compressCallId, name: 'compress', args: {} }],
  })
  return compressCallId
}

test('scanLiveAnchors：session_start + 真实锚点，消耗后消失', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.deepEqual(scanLiveAnchors(entriesOf(session)).map((a) => a.name), ['session_start', 'a1'])
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: compressCallId, logger: silentLogger })
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
  assert.equal(firstUserEntryIndex(entriesOf(session)), 2)

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
  const summary = slotSummary()
  // omp 99a4f87 移除了 force：session_start 直接可用（旧块仍可 expand 取回）。
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'session_start',
    summary,
    callId: 'call-compress-1',
    logger: silentLogger,
  })
  assert.equal(result.shadowedMessages, 3)
  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'user', 'user', 'assistant'])
  assert.equal(shape[1].text, 'INJECTED CONTEXT')
  assert.equal(shape[2].text, 'REAL PROMPT')
  assert.equal(shape[3].text, `[Compressed: session_start → here] (block 1)\n${summary}`)
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
      () => compressFromAnchor({ session, tokenMeter, startAnchor: badName, summary: slotSummary(), callId: compressCallId, logger: silentLogger }),
      /unknown startAnchor/,
    )
  }
})

test('recordAnchor：本调用自身不算重复，其他同名（含已消耗）拒绝', () => {
  const { session, anchorCallId, compressCallId } = buildAnchorScenario()
  assert.deepEqual(recordAnchor({ session, anchor: 'a1', callId: anchorCallId }), { anchor: 'a1' })
  assert.throws(() => recordAnchor({ session, anchor: 'a1', callId: 'call-other' }), /already exists/)
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: compressCallId, logger: silentLogger })
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
  const summary = slotSummary({ '已完成': '- 改完压缩核心 证据: lib/anchor/compress.js:1' })
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a1',
    summary,
    callId: compressCallId,
    logger: silentLogger,
  })
  assert.equal(result.anchor, 'a1')
  assert.equal(result.blockId, 1)
  assert.equal(result.shadowedMessages, 3) // anchor assistant + result + WORK
  assert.equal(result.absorbed, 0)
  assert.deepEqual(result.chain, [])
  assert.deepEqual(result.problems, [])
  assert.equal(session.seq, before + 2)

  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'user', 'assistant'])
  assert.equal(shape[1].text, 'Original user prompt')
  assert.equal(shape[2].source, 'dcp-compress')
  assert.equal(shape[2].text, `[Compressed: a1 → here] (block 1)\n${summary}`)
  const replacement = session.snapshotEvents()[before + 1]
  assert.equal(replacement.data.source.blockId, 1)
  assert.deepEqual(replacement.data.source.parentIds, [])
  assert.equal(replacement.data.source.startToolCallId, 'call-anchor-1')
  const compressMessage = session.deriveMessages().at(-1)
  assert.equal(compressMessage.content[0].type, 'tool-call')
  assert.equal(compressMessage.content[0].id, compressCallId)

  // 影子计价事件紧邻替换事件
  const events = session.snapshotEvents()
  const prune = events[before]
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

test('compress：session_start 不再需要 force，保留原始 prompt', () => {
  const { session, compressCallId } = buildSplitScenario()
  const summary = slotSummary()
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'session_start',
    summary,
    callId: compressCallId,
    logger: silentLogger,
  })
  assert.equal(result.shadowedMessages, 4) // EARLY WORK + anchor + result + LATE WORK
  assert.deepEqual(derivedShape(session).map((m) => m.role), ['system', 'user', 'user', 'assistant'])
  assert.equal(derivedShape(session)[1].text, 'Original user prompt')
  assert.equal(derivedShape(session)[2].text, `[Compressed: session_start → here] (block 1)\n${summary}`)
})

test('compress：真实锚点起点不吞掉起点之前的工作', () => {
  const { session, compressCallId } = buildSplitScenario()
  const summary = slotSummary()
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary, callId: compressCallId, logger: silentLogger })
  const shape = derivedShape(session)
  assert.deepEqual(shape.map((m) => m.role), ['system', 'user', 'assistant', 'user', 'assistant'])
  assert.equal(shape[2].text, 'EARLY WORK')
  assert.equal(shape[3].text, `[Compressed: a1 → here] (block 1)\n${summary}`)
})

test('compress：链式压缩累积摘要，两个块各自存活', () => {
  const { session } = buildAnchorScenario()
  const first = slotSummary()
  const second = slotSummary({ '当前任务': '- 第二段工作 证据: a.md:2' })
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: first, callId: 'call-compress-1', logger: silentLogger })
  const compress2 = appendSecondCompression(session)

  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a2',
    summary: second,
    callId: compress2,
    logger: silentLogger,
  })
  assert.equal(result.blockId, 2)
  assert.equal(result.absorbed, 0)
  assert.deepEqual(result.chain, [])
  const summaries = derivedShape(session).filter((m) => m.source === 'dcp-compress')
  assert.deepEqual(summaries.map((m) => m.text), [
    `[Compressed: a1 → here] (block 1)\n${first}`,
    `[Compressed: a2 → here] (block 2)\n${second}`,
  ])
})

test('compress：跨过先前压缩时吸收为链，旧块文本保留且可 expand', () => {
  const { session } = buildAnchorScenario()
  const first = slotSummary({ '已完成': '- 实现 `compress.js` 证据: lib/anchor/compress.js:1' })
  // 新摘要只写很薄的内容：链式语义下不再因「没接住定位」被拒。
  const thin = slotSummary({ '当前任务': '- 继续链式改造 证据: docs/anchor-compression.md:1' })
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: first, callId: 'call-compress-1', logger: silentLogger })

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
    summary: thin,
    callId: 'call-compress-2',
    logger: silentLogger,
  })
  assert.equal(result.blockId, 2)
  assert.equal(result.absorbed, 1)
  assert.deepEqual(result.absorbedBlockIds, [1])
  assert.deepEqual(result.chain.map((block) => block.blockId), [1])
  assert.deepEqual(result.problems, [])

  const summaries = derivedShape(session).filter((m) => m.source === 'dcp-compress')
  assert.equal(summaries.length, 1)
  assert.ok(summaries[0].text.startsWith('[Compressed: session_start → here] (block 2)'))
  assert.ok(summaries[0].text.includes(thin))
  assert.ok(summaries[0].text.includes('已并入本块的旧压缩块（索引，全文未丢失）:'))
  assert.ok(summaries[0].text.includes('- 块 1「a1」：测试上下文压缩 — 全文: expand({block: 1})'))

  // 被吸收的块不在表面，但日志里仍在，expand 取回全文。
  const expanded = expandBlock({ session, block: 1 })
  assert.equal(expanded.block, 1)
  assert.ok(expanded.content.includes('[block 1「a1」 — 全文]'))
  assert.ok(expanded.content.includes(first))
  assert.ok(!expanded.content.includes('更深一层'))

  const expanded2 = expandBlock({ session, block: 2 })
  assert.ok(expanded2.content.includes('更深一层（索引）:'))
  assert.ok(expanded2.content.includes('- 块 1「a1」：测试上下文压缩 — 全文: expand({block: 1})'))
})

test('compress：摘要只做 advisory 提示，不拒绝压缩', () => {
  const { session, compressCallId } = buildAnchorScenario()
  const summary = '先做 A，然后做 B\n证据: notes.md:4'
  const before = session.seq
  const result = compressFromAnchor({
    session,
    tokenMeter,
    startAnchor: 'a1',
    summary,
    callId: compressCallId,
    logger: silentLogger,
  })
  assert.ok(result.problems.length > 0)
  assert.equal(session.seq, before + 2)
  assert.equal(derivedShape(session).filter((m) => m.source === 'dcp-compress')[0].text,
    `[Compressed: a1 → here] (block 1)\n${summary}`)
})

test('compress：未知锚点列出可用锚点；已消耗锚点给出专门错误', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'nope', summary: slotSummary(), callId: compressCallId, logger: silentLogger }),
    (error) => /unknown startAnchor "nope"/.test(error.message) && /"session_start", "a1"/.test(error.message),
  )
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: compressCallId, logger: silentLogger })
  appendAssistant(session, {
    turn: 4,
    step: 1,
    calls: [{ callId: 'call-compress-2', name: 'compress', args: {} }],
  })
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'call-compress-2', logger: silentLogger }),
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
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger }),
    /contains no messages to compress/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger }),
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
    summary: slotSummary(),
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
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'ptc:call:1', logger: silentLogger }),
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
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger }),
    /crosses a system-prompt update/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'session_start', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger }),
    /crosses a system-prompt update/,
  )
  assert.equal(session.seq, before)
})

test('compress：参数校验与 tokenMeter 缺席', () => {
  const { session, compressCallId } = buildAnchorScenario()
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: '', summary: slotSummary(), callId: compressCallId, logger: silentLogger }),
    /non-empty string "startAnchor"/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: '  ', callId: compressCallId, logger: silentLogger }),
    /non-empty string "summary"/,
  )
  assert.throws(
    () => compressFromAnchor({ session, tokenMeter: undefined, startAnchor: 'a1', summary: slotSummary(), callId: compressCallId, logger: silentLogger }),
    /token meter service/,
  )
})

test('compress：校验失败不产生任何事件', () => {
  const { session, compressCallId } = buildAnchorScenario()
  const before = session.seq
  assert.throws(() => compressFromAnchor({ session, tokenMeter, startAnchor: 'missing', summary: slotSummary(), callId: compressCallId, logger: silentLogger }))
  assert.equal(session.seq, before)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('scanAllAnchorCalls：包含已消耗锚点的全量记录', () => {
  const { session } = buildAnchorScenario()
  assert.deepEqual(scanAllAnchorCalls(session.snapshotEvents()).map((a) => a.name), ['a1'])
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger })
  assert.deepEqual(scanAllAnchorCalls(session.snapshotEvents()).map((a) => a.name), ['a1'])
})

test('scanCompressionBlocks：从日志重导出块，blockId 与 source 一致', () => {
  const { session } = buildAnchorScenario()
  compressFromAnchor({ session, tokenMeter, startAnchor: 'a1', summary: slotSummary(), callId: 'call-compress-1', logger: silentLogger })
  const blocks = scanCompressionBlocks(session.snapshotEvents())
  assert.equal(blocks.length, 1)
  assert.equal(blocks[0].blockId, 1)
  assert.equal(blocks[0].anchor, 'a1')
  assert.equal(blocks[0].startToolCallId, 'call-anchor-1')
  assert.deepEqual(blocks[0].parentIds, [])
})
