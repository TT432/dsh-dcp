/**
 * Summary shape — advisory, never a gate.
 *
 * The slot list and the `证据:` convention live in the tool description and in
 * the compress result as *warnings*: a summary that ignores them still
 * compresses. dsh inherited omp's gates at first (fixed slots, evidence
 * whitelist, the `session_start` strict tier, locator inheritance), and they
 * were removed together with the chain change:
 *
 *   - a rejection costs a full turn and the model re-sends the same draft;
 *     omp measured 17 of 20 real compressions rejected in one session, 9 of
 *     them by a locator whitelist that refused real commit hashes, real
 *     root-level filenames and the `git log` form its own error text listed
 *     as acceptable;
 *   - the checks measured the *shape* of a summary, not the value of what it
 *     kept, and a passing summary could still assert a wrong fact;
 *   - the chain makes the old text reachable (`expand`), so a thin summary is
 *     recoverable instead of fatal. "Merge, don't rewrite" is no longer a rule
 *     the model can fail.
 *
 * What remains is structural parsing, used for two things the code actually
 * needs: rendering an index line for an ancestor block, and telling the model
 * (in the compress result) where its own summary deviates from the requested
 * shape. Nothing is refused.
 *
 * This module is the dsh port of `~/.omp/agent/extensions/dcp/src/summary.ts`
 * (commits 2e18809 → 0583104 → 99a4f87). Behavior is kept identical; only the
 * type annotations are gone.
 */

/** Slot headers, in the recommended order. */
export const SUMMARY_SLOTS = ['当前任务', '已完成', '未完成', '约束', '文件与命令', '在跑作业与句柄']

/** Placeholder for a slot with nothing to report. */
const NONE = '无'

const SLOT_LIST = SUMMARY_SLOTS.join(' / ')

/**
 * `证据:` is a convention, not a whitelist: any text after it is a locator.
 * Greedy `(.*)` splits at the LAST `证据` in the line, so a fact that quotes
 * the word itself still parses; `\s*` tolerates CN punctuation with no space
 * before it (omp commit 0583104: `。证据:` was rejected by the old pattern).
 */
const EVIDENCE_RE = /^(.*)\s*证据\s*[:：]\s*(\S.*)$/s

/** Copy-paste template for the tool description. */
export const SUMMARY_TEMPLATE = SUMMARY_SLOTS.map((slot) => `## ${slot}\n- 事实 证据: <定位>`).join('\n')

/**
 * Parse a summary into bullets, collecting shape deviations instead of
 * throwing. Text that does not follow the slot structure at all is kept as
 * free-form bullets so nothing the model wrote is lost from the index.
 *
 * @param {string} text - model-authored summary.
 * @returns {{ bullets: { slot: string | null, fact: string, evidence: string }[], bySlot: Map<string, object[]>, problems: string[] }}
 *   parsed summary; `problems` are advisory messages for the compress result.
 */
export function parseSummary(text) {
  const lines = String(text ?? '').replace(/\r\n?/g, '\n').split('\n')
  const bySlot = new Map()
  const bullets = []
  const problems = []
  let current = null
  let sawHeader = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    if (line.length === 0) continue
    const where = `第 ${i + 1} 行`

    if (line.startsWith('#')) {
      sawHeader = true
      const name = line.replace(/^#+\s*/u, '').trim()
      const index = SUMMARY_SLOTS.indexOf(name)
      if (index < 0) {
        problems.push(`${where}：槽位「${name}」不在建议列表里（${SLOT_LIST}）`)
        current = null
        continue
      }
      if (bySlot.has(name)) problems.push(`${where}：槽位「${name}」重复出现`)
      current = name
      if (!bySlot.has(name)) bySlot.set(name, [])
      continue
    }

    const isBullet = line.startsWith('- ')
    const body = isBullet ? line.slice(2).trim() : line
    if (!isBullet && current !== null) {
      problems.push(`${where}：槽位「${current}」下的内容建议以「- 」开头`)
    }
    if (body.length === 0) continue

    const match = EVIDENCE_RE.exec(body)
    const fact = (match ? match[1] : body).trim()
    const evidence = match ? match[2].trim() : ''
    if (fact.length > 0 && fact !== NONE && evidence.length === 0) {
      problems.push(`${where}：建议以「证据: <定位>」结尾，方便下一轮重读原始出处`)
    }
    const bullet = { slot: current, fact, evidence }
    bullets.push(bullet)
    if (current !== null) bySlot.get(current).push(bullet)
  }

  if (!sawHeader) {
    problems.push(`摘要没有槽位标题；建议使用 ${SLOT_LIST}`)
  } else {
    const missing = SUMMARY_SLOTS.filter((slot) => !bySlot.has(slot) || bySlot.get(slot).length === 0)
    if (missing.length > 0) problems.push(`摘要缺少槽位：${missing.join(' / ')}`)
    const task = bySlot.get(SUMMARY_SLOTS[0])
    if (task !== undefined && task.every((bullet) => bullet.fact === NONE || bullet.fact.length === 0)) {
      problems.push(`槽位「${SUMMARY_SLOTS[0]}」建议写明实际任务`)
    }
  }

  return { bullets, bySlot, problems }
}

/**
 * Advisory warning block appended to a compress result, or "" when clean.
 * @param {readonly string[]} problems - deviations collected by `parseSummary`.
 * @returns {string} model-facing advisory text.
 */
export function formatProblems(problems) {
  if (!Array.isArray(problems) || problems.length === 0) return ''
  return `\n摘要形状提示（已压缩，未拒绝）：\n${problems.map((problem) => `- ${problem}`).join('\n')}`
}

/**
 * One-line gist of a block for the ancestor index: its 当前任务 bullet, else
 * its first bullet, trimmed to keep the index cheap.
 *
 * @param {string} summary - model-authored summary body.
 * @param {number} [max] - maximum gist length.
 * @returns {string} single-line gist (empty when the summary has no fact).
 */
export function gistOf(summary, max = 160) {
  const parsed = parseSummary(summary)
  const task = parsed.bySlot.get(SUMMARY_SLOTS[0])?.find((bullet) => bullet.fact !== NONE && bullet.fact.length > 0)
  const source =
    task?.fact ??
    parsed.bullets.find((bullet) => bullet.fact !== NONE && bullet.fact.length > 0)?.fact ??
    ''
  const flat = source.replace(/\s+/gu, ' ').trim()
  if (flat.length <= max) return flat
  return `${flat.slice(0, max)}…`
}
