/**
 * Summary schema — fixed slots + evidence pointers.
 *
 * A free-form summary is a lossy paraphrase: whatever the model omits is gone,
 * and nothing in the text lets a later turn tell a remembered fact from an
 * invented one. The schema makes two properties checkable instead of
 * aspirational:
 *
 *   1. FIXED SLOTS — every summary carries the same six sections in the same
 *      order. 当前任务 is mandatory, so a compression always records what is
 *      actually being worked on; 未完成 records the next action.
 *   2. EVIDENCE PER FACT — every bullet ends with `证据: <locator>`, an
 *      externalized pointer (file:line, command, session id, tool result).
 *      Facts point at durable artifacts instead of pasting them, so the
 *      summary stays small and a later turn can re-read the source.
 *
 * `full` (session_start) compressions get the stricter tier: 已完成 and 未完成
 * must carry real content, and every evidence locator must be locatable.
 * That is the "validation cost" — full-session compression is the amnesia
 * footgun, so it is deliberately the most expensive summary to write.
 *
 * This module is the dsh port of `~/.omp/agent/extensions/dcp/src/summary.ts`
 * (commits 2e18809 + 3caea33). Behavior is kept identical; only the type
 * annotations are gone.
 */

/** Slot headers, in the order they must appear. */
export const SUMMARY_SLOTS = ['当前任务', '已完成', '未完成', '约束', '文件与命令', '在跑作业与句柄']

/** Placeholder for a slot with nothing to report. */
const NONE = '无'

/** A fact longer than this is a paste, not a pointer — facts belong outside. */
const FACT_MAX = 600
const EVIDENCE_MAX = 200

const SLOT_LIST = SUMMARY_SLOTS.join(' / ')
const EVIDENCE_RE = /^(.+)\s证据\s*[:：]\s*(\S.*)$/s
/** Locator shapes accepted at the strict tier: path, file:line, session id, URI, command. */
const LOCATOR_RE = /[:#/\\@]/

/** Copy-paste template for the tool description. */
export const SUMMARY_TEMPLATE = SUMMARY_SLOTS.map((slot) => `## ${slot}\n- 事实 证据: <定位>`).join('\n')

function fail(message) {
  throw new Error(message)
}

/**
 * Parse and structurally validate a summary.
 *
 * Throws with a message naming the offending line and the required shape;
 * callers must validate before any state mutation.
 *
 * @param {string} text - model-authored summary.
 * @returns {{ bullets: { slot: string, fact: string, evidence: string }[], bySlot: Map<string, object[]> }} parsed summary.
 */
export function parseSummary(text) {
  const lines = String(text).replace(/\r\n?/g, '\n').split('\n')
  const bySlot = new Map()
  const bullets = []
  let current = null
  let lastSlotIndex = -1

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    const where = `第 ${i + 1} 行`
    if (line.length === 0) continue

    if (line.startsWith('#')) {
      if (!line.startsWith('## ')) fail(`${where}：槽位标题必须写成「## <槽位名>」`)
      const name = line.slice(3).trim()
      const index = SUMMARY_SLOTS.indexOf(name)
      if (index < 0) fail(`${where}：未知槽位「${name}」。允许的槽位只有：${SLOT_LIST}`)
      if (bySlot.has(name)) fail(`${where}：槽位「${name}」重复出现`)
      if (index <= lastSlotIndex) {
        fail(`${where}：槽位「${name}」顺序错误。槽位必须按此顺序出现：${SLOT_LIST}`)
      }
      lastSlotIndex = index
      current = name
      bySlot.set(current, [])
      continue
    }

    if (current === null) {
      fail(`${where}：摘要必须是固定槽位结构，第一条内容之前必须先是「## ${SUMMARY_SLOTS[0]}」；不要写开场白或自由叙述`)
    }
    if (!line.startsWith('- ')) {
      fail(`${where}：槽位「${current}」下的每条内容都必须以「- 」开头（事实或「- 无」）`)
    }
    const body = line.slice(2).trim()
    if (body.length === 0) fail(`${where}：空条目；槽位没有内容时写「- ${NONE}」`)

    if (body === NONE) {
      const bullet = { slot: current, fact: NONE, evidence: '' }
      bullets.push(bullet)
      bySlot.get(current).push(bullet)
      continue
    }

    const match = EVIDENCE_RE.exec(body)
    if (!match) {
      fail(`${where}：每条事实都必须以「证据: <定位>」结尾（例如「- 改了 transform.ts:44 证据: src/transform.ts:44」）；没有可引用证据的内容不要写进摘要`)
    }
    const fact = match[1].trim()
    const evidence = match[2].trim()
    if (fact.length === 0) fail(`${where}：缺少事实正文（只剩证据）`)
    if (fact.length > FACT_MAX) {
      fail(`${where}：事实过长（${fact.length} > ${FACT_MAX} 字）。事实要外置：写指向文件的定位，不要粘贴原文`)
    }
    if (evidence.length > EVIDENCE_MAX) {
      fail(`${where}：证据定位过长（${evidence.length} > ${EVIDENCE_MAX} 字）`)
    }
    if (evidence === NONE) fail(`${where}：证据写成「${NONE}」没有意义；没有证据的事实不要写，槽位为空就写「- ${NONE}」`)

    const bullet = { slot: current, fact, evidence }
    bullets.push(bullet)
    bySlot.get(current).push(bullet)
  }

  const missing = SUMMARY_SLOTS.filter((slot) => !bySlot.has(slot))
  if (missing.length > 0) {
    fail(`摘要缺少槽位：${missing.join(' / ')}。固定槽位（顺序固定）：${SLOT_LIST}`)
  }
  for (const slot of SUMMARY_SLOTS) {
    if (bySlot.get(slot).length === 0) fail(`槽位「${slot}」没有条目；没有内容时写「- ${NONE}」`)
  }
  const task = bySlot.get(SUMMARY_SLOTS[0])
  if (task.every((b) => b.fact === NONE)) {
    fail(`槽位「${SUMMARY_SLOTS[0]}」必须是实际任务，不能写「${NONE}」`)
  }

  return { bullets, bySlot }
}

/**
 * Validate a summary for a compress call. Throws on any schema violation.
 * `full` adds the strict tier used for session_start compression.
 *
 * @param {string} text - model-authored summary.
 * @param {{ full?: boolean }} [options] - `full` selects the strict tier.
 * @returns parsed summary.
 */
export function validateSummary(text, options = {}) {
  const parsed = parseSummary(text)
  if (!options.full) return parsed

  for (const slot of ['已完成', '未完成']) {
    const list = parsed.bySlot.get(slot)
    if (list.length === 1 && list[0].fact === NONE) {
      fail(`force 全量压缩要求槽位「${slot}」写明实际内容，不能是「${NONE}」：全量压缩丢掉的是全部工作上下文，摘要必须能独立支撑下一轮`)
    }
  }
  for (const bullet of parsed.bullets) {
    if (bullet.fact === NONE) continue
    if (!LOCATOR_RE.test(bullet.evidence)) {
      fail(`force 全量压缩要求可核对的证据定位，「${bullet.slot}」条目「${bullet.fact.slice(0, 40)}」的证据「${bullet.evidence}」不含任何定位（需要 path、file:line、会话 id、URI 或命令）`)
    }
  }
  return parsed
}

// ---------------------------------------------------------------------------
// baseline inheritance (absorbing a prior block must not drop its facts)
// ---------------------------------------------------------------------------

/**
 * Slots whose facts outlive a compression. 当前任务 and 未完成 describe the live
 * edge of the work and legitimately change; 已完成 / 约束 / 文件与命令 describe
 * durable state and must never vanish silently when the block holding them is
 * absorbed.
 *
 * Session 01a18dc1 lost a finished implementation exactly this way: block 2
 * recorded it (files, verification), block 3 replaced block 2 with an
 * analysis-only summary, and the following turn re-proposed work that was
 * already done and committed. Slot structure alone does not prevent it —
 * every block was well-formed; the content was replaced, not merged.
 */
export const INHERITED_SLOTS = ['已完成', '约束', '文件与命令']

/**
 * Locator-ish tokens inside a fact: backticked identifiers, file paths and
 * CONSTANTS. Anything a later turn could re-find in the repository. Prose
 * without a locator is not checkable, so it is not checked — the schema already
 * asks facts to point at artifacts rather than paste them.
 */
const KEY_PATTERNS = [
  /`([^`\n]{2,120})`/g,
  /[A-Za-z0-9_.\/\\-]+\.(?:ts|tsx|js|mjs|cjs|json|md|py|toml|ya?ml|txt)\b/g,
  /\b[A-Z][A-Z0-9]*(?:[-_][A-Z0-9]+)+\b/g,
]

/** A superseded fact must say so in its own bullet. */
const DISCARD_MARK = '废弃'

function normalizeKey(key) {
  return key.trim().replace(/\\/g, '/').toLowerCase()
}

/**
 * Durable locators mentioned by one fact.
 * @param {string} fact - fact text without its evidence segment.
 * @returns {string[]} normalized locators.
 */
export function collectIdentityKeys(fact) {
  const out = new Set()
  for (const re of KEY_PATTERNS) {
    re.lastIndex = 0
    for (let m = re.exec(fact); m !== null; m = re.exec(fact)) {
      const key = normalizeKey(m[1] ?? m[0])
      if (key.length >= 2 && key.length <= 120) out.add(key)
    }
  }
  return [...out]
}

/**
 * Refuse a compression that absorbs prior blocks while dropping the locatable
 * facts they recorded. A fact survives when its locator appears anywhere in the
 * new summary — moving slots and merging bullets are fine, and a bullet that
 * says the old fact is superseded naturally contains its locator.
 *
 * Blocks written before the slot schema cannot be parsed; they are skipped —
 * history never blocks a compression.
 *
 * @param {{ blockId: number, startName: string, summary: string }[]} previous - absorbed blocks.
 * @param {{ bullets: object[], bySlot: Map<string, object[]> }} next - parsed new summary.
 * @throws when a locator recorded by an absorbed block is missing.
 */
export function validateInheritance(previous, next) {
  if (previous.length === 0) return
  const haystack = next.bullets.map((b) => normalizeKey(`${b.fact} ${b.evidence}`)).join('\n')
  const problems = []

  for (const block of previous) {
    let parsed
    try {
      parsed = parseSummary(block.summary)
    } catch {
      continue // pre-schema block — nothing structurally checkable
    }
    for (const slot of INHERITED_SLOTS) {
      for (const bullet of parsed.bySlot.get(slot)) {
        if (bullet.fact === NONE) continue
        const uncovered = collectIdentityKeys(bullet.fact).filter((key) => !haystack.includes(key))
        if (uncovered.length > 0) {
          const snippet = bullet.fact.length > 60 ? `${bullet.fact.slice(0, 60)}…` : bullet.fact
          problems.push(`  - block ${block.blockId}「${block.startName}」/ ${slot}：「${snippet}」缺 ${uncovered.join(', ')}`)
        }
      }
    }
  }

  if (problems.length === 0) return
  fail(
    `本次压缩会吸收 ${previous.length} 个既有压缩块，但新摘要没有接住它们记录的事实（基线继承）。` +
      `下列定位必须出现在新摘要中（可换槽位、可合并进其他条目；确已失效就在条目里写「${DISCARD_MARK}: <原因>」，同样要带上定位）：\n` +
      `${problems.join('\n')}\n` +
      `吸收旧块时摘要必须合并而不是重写。`,
  )
}
