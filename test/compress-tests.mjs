// LLM-DCP 压缩核心测试：课程调度 / token 估算 / 提示词构造 / MDP 迭代（mock LLM）
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { compressWithAgent, buildCompressPrompt, curriculumRatio, estimateTokens } from '../lib/dcp/compress.js'

// ---------- 课程调度 ----------
test('curriculumRatio：从宽松到严格', () => {
  assert.equal(curriculumRatio(1, 4, 0.4), 0.85) // 1 - 0.6*0.25 = 0.85
  assert.equal(curriculumRatio(4, 4, 0.4), 0.4)
  assert.equal(curriculumRatio(1, 1, 0.3), 0.3)
})

test('estimateTokens：CJK 加权', () => {
  assert.ok(estimateTokens('hello world this is a test') > 0)
  // 中文按每字符 ~1 token
  assert.ok(estimateTokens('这是一个测试句子') >= 7)
})

// ---------- 提示词 ----------
test('buildCompressPrompt：包含目标比例与关键信息', () => {
  const { system, user } = buildCompressPrompt({
    text: 'some text here',
    targetRatio: 0.5,
    step: 1, totalSteps: 2,
    keyPoints: 'KEEP: 42, Alice',
  })
  assert.ok(system.includes('DCP-Agent'))
  assert.ok(user.includes('50%'))
  assert.ok(user.includes('KEEP: 42, Alice'))
  assert.ok(user.includes('some text here'))
})

// ---------- mock LLM ----------
class MockAssembler {
  constructor() { this.parts = [] }
  push(chunk) { if (chunk.type === 'text-delta') this.parts.push(chunk.text) }
  get finish() { return { kind: 'stop' } }
  blocks() { return [{ type: 'text', text: this.parts.join('') }] }
}
function makeMockLlm(script) {
  // script: 每次调用返回的下一个文本
  let i = 0
  return {
    BlockAssembler: MockAssembler,
    calls: [],
    stream: async function* (opts) {
      const text = typeof script === 'function' ? script(opts, i) : script[i % script.length]
      this.calls.push(opts)
      i++
      yield { type: 'block-start', index: 0, blockType: 'text' }
      if (text) yield { type: 'text-delta', index: 0, text }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: text || '' } }
      yield { type: 'finish', reason: 'stop' }
    },
  }
}

// 真实 BlockAssembler + 0.2.x 的 StreamChunk 协议；用来验证插件组装的请求契约与
// 真实分片（block-start/text-delta/block-end/usage/finish）能配合工作。
function makeProtocolLlm(script, finishReason = { kind: 'stop' }) {
  let i = 0
  return {
    BlockAssembler,
    calls: [],
    stream: async function* (opts) {
      const text = typeof script === 'function' ? script(opts, i) : script[i % script.length]
      this.calls.push(opts)
      i++
      yield { type: 'block-start', index: 0, blockType: 'text' }
      if (text) yield { type: 'text-delta', index: 0, text }
      yield { type: 'block-end', index: 0, block: { type: 'text', text: text || '' } }
      yield { type: 'usage', usage: { inputTokens: 10, outputTokens: 5 } }
      yield { type: 'finish', reason: finishReason }
    },
  }
}
const stubCtx = { llm: null }

test('compressWithAgent：两步迭代压缩 + 关键信息传递', async () => {
  const original = 'This is a long document with many details. '.repeat(20)
  // 第一步压缩到 ~70%，第二步到 ~50%
  const llm = makeMockLlm([
    original.slice(0, Math.round(original.length * 0.7)),
    original.slice(0, Math.round(original.length * 0.5)),
  ])
  stubCtx.llm = llm
  const res = await compressWithAgent({
    ctx: stubCtx, llm,
    text: original,
    targetRatio: 0.5, steps: 2,
    keyPoints: 'KEEP 42',
  })
  assert.ok(res.compressedChars < original.length)
  assert.ok(Math.abs(res.ratio - 0.5) < 0.05, 'ratio=' + res.ratio)
  assert.equal(res.stepsUsed, 2)
  assert.equal(res.perStep.length, 2)
  assert.equal(llm.calls.length, 2)
  // 每步提示词包含关键信息
  for (const call of llm.calls) assert.ok(call.messages[0].content[0].text.includes('KEEP 42'))
})

test('compressWithAgent：第一步即达标则提前终止', async () => {
  const original = 'short text here'.repeat(5)
  const llm = makeMockLlm([original.slice(0, 20)]) // 一步压到 ~40%
  stubCtx.llm = llm
  const res = await compressWithAgent({ ctx: stubCtx, llm, text: original, targetRatio: 0.5, steps: 3 })
  assert.equal(res.stepsUsed, 1)
  assert.equal(llm.calls.length, 1)
})

test('compressWithAgent：第一步空输出 → 报错', async () => {
  const llm = makeMockLlm([''])
  stubCtx.llm = llm
  await assert.rejects(
    compressWithAgent({ ctx: stubCtx, llm, text: 'hello world '.repeat(10), targetRatio: 0.5 }),
    /empty output/
  )
})

test('compressWithAgent：第一步无进展 → 报错', async () => {
  const text = 'no progress here '.repeat(10)
  const llm = makeMockLlm([text]) // 输出与输入相同
  stubCtx.llm = llm
  await assert.rejects(
    compressWithAgent({ ctx: stubCtx, llm, text, targetRatio: 0.5 }),
    /no-progress/
  )
})

test('compressWithAgent：单步目标压缩', async () => {
  const original = 'A single step compression target. '.repeat(10)
  const llm = makeMockLlm([original.slice(0, Math.round(original.length * 0.6))])
  stubCtx.llm = llm
  const res = await compressWithAgent({ ctx: stubCtx, llm, text: original, targetRatio: 0.6, steps: 1 })
  assert.equal(res.stepsUsed, 1)
  assert.ok(Math.abs(res.ratio - 0.6) < 0.05)
})

test('compressWithAgent：onStep 回调', async () => {
  const original = 'Callback test text. '.repeat(10)
  const llm = makeMockLlm([original.slice(0, Math.round(original.length * 0.7)), original.slice(0, Math.round(original.length * 0.5))])
  stubCtx.llm = llm
  const seen = []
  const res = await compressWithAgent({
    ctx: stubCtx, llm, text: original, targetRatio: 0.4, steps: 2,
    onStep: (info) => seen.push(info.step),
  })
  assert.deepEqual(seen, [1, 2])
  assert.ok(res.perStep.length === 2)
})

// ---------- 0.2.x 运行时契约（真实 BlockAssembler + StreamChunk） ----------
test('compressWithAgent：真实 BlockAssembler 能消费标准分片协议', async () => {
  const original = 'Real assembler protocol text. '.repeat(20)
  const llm = makeProtocolLlm([
    original.slice(0, Math.round(original.length * 0.7)),
    original.slice(0, Math.round(original.length * 0.5)),
  ])
  stubCtx.llm = llm
  const res = await compressWithAgent({ ctx: stubCtx, llm, text: original, targetRatio: 0.5, steps: 2 })
  assert.equal(res.stepsUsed, 2)
  assert.equal(llm.calls.length, 2)
  assert.equal(res.text, original.slice(0, Math.round(original.length * 0.5)).trim())
})

test('compressWithAgent：请求使用 RequestUserInput（无 id/source）且无非法 purpose', async () => {
  const original = 'Request contract text. '.repeat(10)
  const llm = makeProtocolLlm([original.slice(0, Math.round(original.length * 0.5))])
  stubCtx.llm = llm
  await compressWithAgent({ ctx: stubCtx, llm, text: original, targetRatio: 0.5, steps: 1 })
  const call = llm.calls[0]
  // 0.2.x 的 GenerateOptions.messages 接受无身份的 RequestUserInput；
  // MessageSource 只允许 user/model/tool/system-prompt，不能伪造 plugin kind。
  assert.deepEqual(Object.keys(call.messages[0]).sort(), ['content', 'role'])
  assert.equal(call.messages[0].role, 'user')
  assert.equal(call.messages[0].id, undefined)
  assert.equal(call.messages[0].source, undefined)
  assert.equal(typeof call.system, 'string')
  // purpose 在 0.2.x 是 'compaction' | 'session-title' 的封闭联合，普通插件调用不应设置。
  assert.equal('purpose' in call, false)
})

test('compressWithAgent：真实 finish(kind=error) → 明确报错', async () => {
  const llm = makeProtocolLlm(
    ['irrelevant'],
    { kind: 'error', failure: { message: 'auth failed', code: 'AUTH' } },
  )
  stubCtx.llm = llm
  await assert.rejects(
    compressWithAgent({ ctx: stubCtx, llm, text: 'hello world '.repeat(10), targetRatio: 0.5 }),
    /llm call failed: auth failed/,
  )
})
