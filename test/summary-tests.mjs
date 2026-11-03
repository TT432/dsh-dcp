// Summary shape tests: the parser is advisory (omp 99a4f87) — it never throws,
// collects deviations as problems, and still yields bullets for the ancestor
// index. Evidence pointers use the relaxed regex from omp 0583104.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { SUMMARY_SLOTS, formatProblems, gistOf, parseSummary } from '../lib/anchor/summary.js'
import { slotSummary } from './helpers.mjs'

test('parseSummary：六个建议槽位按顺序解析', () => {
  const parsed = parseSummary(slotSummary())
  assert.deepEqual([...parsed.bySlot.keys()], SUMMARY_SLOTS)
  assert.equal(parsed.bySlot.get('当前任务')[0].fact, '测试上下文压缩')
  assert.equal(parsed.bySlot.get('当前任务')[0].evidence, 'test/helpers.mjs:1')
  assert.equal(parsed.bySlot.get('已完成')[0].fact, '无')
  assert.equal(parsed.bullets.length, 6)
  assert.deepEqual(parsed.problems, [])
})

test('parseSummary：证据正则放宽——中文标点后无空格、全角冒号、重复「证据」取最后一处', () => {
  const noSpace = parseSummary(`## 当前任务\n- 完成压缩。证据: lib/anchor/summary.js:1`)
  assert.equal(noSpace.bySlot.get('当前任务')[0].fact, '完成压缩。')
  assert.equal(noSpace.bySlot.get('当前任务')[0].evidence, 'lib/anchor/summary.js:1')
  assert.deepEqual(noSpace.problems.filter((p) => p.includes('证据')), [])

  const fullwidth = parseSummary(`## 当前任务\n- 完成（注）证据：docs/design.md:2`)
  assert.equal(fullwidth.bySlot.get('当前任务')[0].evidence, 'docs/design.md:2')

  const repeated = parseSummary(`## 当前任务\n- 记录「证据: 旧说法」 证据: new.md:3`)
  assert.equal(repeated.bySlot.get('当前任务')[0].fact, '记录「证据: 旧说法」')
  assert.equal(repeated.bySlot.get('当前任务')[0].evidence, 'new.md:3')
})

test('parseSummary：结构偏差只记问题，不抛错', () => {
  const cases = [
    ['自由叙述开场白', (problems) => problems.some((p) => p.includes('没有槽位标题'))],
    ['## 当前任务\n- 任务 证据: a.md:1\n## 未知槽\n- x 证据: y', (problems) => problems.some((p) => p.includes('不在建议列表'))],
    ['## 当前任务\n- 任务 证据: a.md:1\n## 当前任务\n- 重复 证据: b.md:1', (problems) => problems.some((p) => p.includes('重复出现'))],
    ['## 当前任务\n- 任务 证据: a.md:1', (problems) => problems.some((p) => p.includes('缺少槽位'))],
    [slotSummary({ '当前任务': '- 无' }), (problems) => problems.some((p) => p.includes('写明实际任务'))],
    ['## 当前任务\n任务 证据: a.md:1', (problems) => problems.some((p) => p.includes('以「- 」开头'))],
    ['## 当前任务\n- 任务没有证据', (problems) => problems.some((p) => p.includes('证据: <定位>'))],
  ]
  for (const [text, predicate] of cases) {
    const parsed = parseSummary(text)
    assert.ok(predicate(parsed.problems), `${text}\n${parsed.problems.join('\n')}`)
  }
})

test('parseSummary：完全自由格式也保留为条目（不丢模型写的内容）', () => {
  const parsed = parseSummary('先做 A，然后做 B\n证据: notes.md:4')
  assert.equal(parsed.bullets.length, 2)
  assert.equal(parsed.bullets[0].slot, null)
  assert.equal(parsed.bullets[0].fact, '先做 A，然后做 B')
  assert.equal(parsed.bullets[1].evidence, 'notes.md:4')
  assert.ok(parsed.problems.length > 0)
})

test('gistOf：优先当前任务，其次任意事实，过长截断', () => {
  assert.equal(gistOf(slotSummary({ '当前任务': '- 实现链式摘要 证据: a.md:1' })), '实现链式摘要')
  assert.equal(gistOf('自由格式：只有一条事实 证据: a.md:1'), '自由格式：只有一条事实')
  const long = gistOf(slotSummary({ '当前任务': `- ${'x'.repeat(200)} 证据: a.md:1` }), 20)
  assert.equal(long.length, 21)
  assert.ok(long.endsWith('…'))
  assert.equal(gistOf(''), '')
})

test('formatProblems：干净时为空，否则给出「已压缩，未拒绝」提示', () => {
  assert.equal(formatProblems([]), '')
  assert.equal(formatProblems(undefined), '')
  const text = formatProblems(['第 2 行：建议以「证据: <定位>」结尾'])
  assert.ok(text.includes('摘要形状提示（已压缩，未拒绝）'))
  assert.ok(text.includes('- 第 2 行：建议以「证据: <定位>」结尾'))
})
