// Summary schema tests: fixed slots, evidence pointers, strict tier, and
// baseline inheritance (absorbing a prior block must carry its locators).
import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  INHERITED_SLOTS,
  SUMMARY_SLOTS,
  collectIdentityKeys,
  parseSummary,
  validateInheritance,
  validateSummary,
} from '../lib/anchor/summary.js'
import { fullSummary, slotSummary } from './helpers.mjs'

test('parseSummary：六个固定槽位按顺序解析', () => {
  const parsed = parseSummary(slotSummary())
  assert.deepEqual([...parsed.bySlot.keys()], SUMMARY_SLOTS)
  assert.equal(parsed.bySlot.get('当前任务')[0].fact, '测试上下文压缩')
  assert.equal(parsed.bySlot.get('当前任务')[0].evidence, 'test/helpers.mjs:1')
  assert.equal(parsed.bySlot.get('已完成')[0].fact, '无')
  assert.equal(parsed.bullets.length, 6)
})

test('parseSummary：结构错误逐类报错', () => {
  const cases = [
    ['自由叙述开场白', /固定槽位结构/],
    [`## 当前任务\n- 任务 证据: a.md:1\n## 未知槽\n- x 证据: y`, /未知槽位/],
    [`## 当前任务\n- 任务 证据: a.md:1\n## 已完成\n- 完成 证据: b.md:1\n## 当前任务\n- 重复 证据: c.md:1`, /重复出现|顺序错误/],
    [`## 已完成\n- 完成 证据: b.md:1\n## 当前任务\n- 任务 证据: a.md:1`, /顺序错误/],
    [`## 当前任务\n- 任务 证据: a.md:1\n## 已完成\n- 完成`, /证据/],
    [`## 当前任务\n任务 证据: a.md:1`, /以「- 」开头/],
    [`## 当前任务\n- 任务 证据: a.md:1\n## 已完成\n- 完成 证据: 无`, /证据写成「无」/],
  ]
  for (const [text, pattern] of cases) {
    assert.throws(() => parseSummary(text), pattern, text)
  }
})

test('parseSummary：缺少槽位、空槽位、当前任务为「无」都拒绝', () => {
  assert.throws(() => parseSummary(`## 当前任务\n- 任务 证据: a.md:1`), /缺少槽位/)
  const emptySlot = slotSummary().replace('## 约束\n- 无', '## 约束')
  assert.throws(() => parseSummary(emptySlot), /没有条目/)
  assert.throws(() => parseSummary(slotSummary({ '当前任务': '- 无' })), /必须是实际任务/)
})

test('parseSummary：事实超过 600 字拒绝（摘要是指针不是副本）', () => {
  const long = slotSummary({ '已完成': `- ${'x'.repeat(601)} 证据: a.md:1` })
  assert.throws(() => parseSummary(long), /事实过长/)
})

test('validateSummary：普通层接受宽松证据，strict 层要求真实定位', () => {
  const loose = slotSummary({ '已完成': '- 完成 证据: 某处' })
  assert.doesNotThrow(() => validateSummary(loose))
  assert.throws(
    () => validateSummary(fullSummary({ '已完成': '- 完成 证据: 某处' }), { full: true }),
    /可核对的证据定位/,
  )

  assert.doesNotThrow(() => validateSummary(fullSummary(), { full: true }))
  assert.throws(
    () => validateSummary(fullSummary({ '未完成': '- 无' }), { full: true }),
    /写明实际内容/,
  )
})

test('collectIdentityKeys：反引号标识符、文件路径、CONSTANT', () => {
  const keys = collectIdentityKeys('改了 `summary.js` 与 lib/anchor/compress.js，常量 DCP_REMIND_TOOLCALLS')
  assert.ok(keys.includes('summary.js'))
  assert.ok(keys.includes('lib/anchor/compress.js'))
  assert.ok(keys.includes('dcp_remind_toolcalls'))
})

test('validateInheritance：接住旧块的定位才允许吸收', () => {
  const previous = [{
    blockId: 1,
    startName: 'a1',
    summary: slotSummary({ '已完成': '- 实现 `summary.js` 证据: lib/anchor/summary.js:1' }),
  }]
  // 丢失定位 → 拒绝，错误里列出定位
  assert.throws(
    () => validateInheritance(previous, parseSummary(fullSummary({ '已完成': '- 完成别处 证据: test/x.mjs:1' }))),
    /基线继承/,
  )
  // 保留定位（可换槽位）→ 通过
  assert.doesNotThrow(() => validateInheritance(
    previous,
    parseSummary(fullSummary({ '已完成': '- 吸收旧块：`summary.js` 证据: lib/anchor/summary.js:1' })),
  ))
  // 已废弃但保留定位 → 通过
  assert.doesNotThrow(() => validateInheritance(
    previous,
    parseSummary(fullSummary({ '已完成': '- `summary.js` 已废弃: 已迁移 证据: lib/anchor/summary.js:1' })),
  ))
})

test('validateInheritance：只检查 已完成/约束/文件与命令，且跳过不可解析旧块', () => {
  assert.deepEqual([...INHERITED_SLOTS], ['已完成', '约束', '文件与命令'])
  const previous = [{
    blockId: 1,
    startName: 'a1',
    // 旧格式（非槽位）无法解析 → 不阻塞
    summary: 'free-form summary',
  }]
  assert.doesNotThrow(() => validateInheritance(previous, parseSummary(slotSummary())))

  // 当前任务 / 未完成 的定位不要求继承
  const taskOnly = [{
    blockId: 2,
    startName: 'a2',
    summary: slotSummary({ '当前任务': '- 任务 `live.js` 证据: lib/live.js:1' }),
  }]
  assert.doesNotThrow(() => validateInheritance(taskOnly, parseSummary(slotSummary())))
})
