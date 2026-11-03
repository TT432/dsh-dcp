// Compression-block chain tests: log-derived blocks, legacy (pre-chain) ids,
// ancestry, and expand rendering/errors.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createUserMessage } from '@deepseek-ai/dsh-llm'
import {
  ancestorIndex,
  ancestorsOf,
  expandBlock,
  extractSummaryBody,
  findBlock,
  nextBlockId,
  scanCompressionBlocks,
} from '../lib/anchor/blocks.js'
import { createSession } from './helpers.mjs'

/** Append a compression node directly to the log (no surface replacement). */
function appendCompressionNode(session, { anchor, summary, blockId, parentIds, startToolCallId }) {
  const source = {
    kind: 'dcp-compress',
    anchor,
    ...(blockId === undefined ? {} : { blockId }),
    ...(parentIds === undefined ? {} : { parentIds }),
    ...(startToolCallId === undefined ? {} : { startToolCallId }),
  }
  const text = `[Compressed: ${anchor} → here]${blockId === undefined ? '' : ` (block ${blockId})`}\n${summary}`
  return session.append('user/message', createUserMessage({
    content: [{ type: 'text', text }],
    source,
  }), { surfaceOp: 'append' })
}

test('extractSummaryBody：去掉头行与祖先索引尾注', () => {
  const body = '## 当前任务\n- 做 A 证据: a.md:1'
  const withIndex = `[Compressed: a1 → here] (block 1)\n${body}\n\n已并入本块的旧压缩块（索引，全文未丢失）:\n- 块 0「a0」：旧事 — 全文: expand({block: 0})`
  assert.equal(extractSummaryBody(withIndex), body)
  assert.equal(extractSummaryBody(`[Compressed: a1 → here]\n${body}`), body)
  assert.equal(extractSummaryBody('只有一行'), '')
})

test('scanCompressionBlocks：显式 blockId 保留，旧块按日志顺序补号', () => {
  const session = createSession('legacy-blocks')
  appendCompressionNode(session, { anchor: 'old', summary: '旧摘要' })
  appendCompressionNode(session, { anchor: 'a5', summary: '新摘要', blockId: 5, parentIds: [1] })
  const blocks = scanCompressionBlocks(session.snapshotEvents())
  assert.deepEqual(blocks.map((block) => block.blockId), [1, 5])
  assert.deepEqual(blocks.map((block) => block.anchor), ['old', 'a5'])
  assert.deepEqual(blocks[1].parentIds, [1])
  assert.equal(nextBlockId(blocks), 6)
  assert.equal(findBlock(blocks, 5).summary, '新摘要')
})

test('scanCompressionBlocks：忽略普通 user 消息与非压缩来源', () => {
  const session = createSession('non-compression')
  session.append('user/message', createUserMessage({
    content: [{ type: 'text', text: '普通消息' }],
    source: { kind: 'user' },
  }), { surfaceOp: 'append' })
  assert.deepEqual(scanCompressionBlocks(session.snapshotEvents()), [])
})

test('ancestorsOf：传递祖先、去重、父块缺失时跳过', () => {
  const blocks = [
    { blockId: 1, parentIds: [] },
    { blockId: 2, parentIds: [1] },
    { blockId: 3, parentIds: [2, 1] },
    { blockId: 4, parentIds: [99] },
  ]
  assert.deepEqual(ancestorsOf(blocks, blocks[0]).map((b) => b.blockId), [])
  assert.deepEqual(ancestorsOf(blocks, blocks[2]).map((b) => b.blockId), [2, 1])
  assert.deepEqual(ancestorsOf(blocks, blocks[3]).map((b) => b.blockId), [])
})

test('ancestorIndex：为索引行提供 id/锚点/gist', () => {
  const blocks = [
    { blockId: 1, anchor: 'a1', parentIds: [], summary: '## 当前任务\n- 旧任务 证据: a.md:1' },
    { blockId: 2, anchor: 'a2', parentIds: [1], summary: '' },
  ]
  assert.deepEqual(ancestorIndex(blocks, blocks[1]), [{ blockId: 1, anchor: 'a1', gist: '旧任务' }])
})

test('expandBlock：未知块报错并列出已知块', () => {
  const session = createSession('expand-unknown')
  appendCompressionNode(session, { anchor: 'a1', summary: '摘要', blockId: 2 })
  assert.throws(() => expandBlock({ session, block: 9 }), /unknown compression block 9\. Known blocks: 2/)
  assert.throws(() => expandBlock({ session, block: 'x' }), /requires an integer/)
  const expanded = expandBlock({ session, block: '2' })
  assert.equal(expanded.block, 2)
  assert.ok(expanded.content.includes('[block 2「a1」 — 全文]'))
})
