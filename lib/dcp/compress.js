// LLM-DCP (Dynamic Compressing Prompts, arXiv:2504.11004) 的 dsh 实现。
//
// 论文核心：把 prompt 压缩建模为马尔可夫决策过程（MDP）——
//   状态 s_t = 当前（逐步压缩后的）prompt；
//   动作 a_t = 对每个 token 的保留/删除决策；
//   转移 T   = 按动作删除 token 得到 s_{t+1}；
//   奖励 R   = α·(1/ρ) + β·BertScore(关键信息保留) − γ·KL(输出分布) − 越界惩罚。
// DCP-Agent 是逐 token 二分类策略网络（论文用 xlm-roberta-large + PPO + HPC 课程训练）。
//
// 本插件在 dsh 中的等价实现（LLM-as-DCP-Agent）：
//   - 用 ctx.llm 的可配置 LLM 充当 DCP-Agent 的策略网络；
//   - 按论文 MDP 框架**迭代多步**压缩：每一步基于上一步的结果（动态上下文）删除冗余、
//     保留关键内容（动作），目标压缩率随轨迹逐步收紧（HPC 课程思想）;
//   - 奖励函数的三个要素映射为：压缩率目标（ρ）、关键信息保留（提示词显式要求 +
//     可选的保留性校验）、输出质量（质量指令 + 非空/结构校验）。
//   - 论文"不依赖黑盒 LLM 训练"的目标在本实现中由推理端直接使用 LLM 承担。
//
// 说明：dsh 未暴露 tokenizer，token 数为字符近似估算（英文≈4 字符/token，CJK 加权），
// 压缩率控制以字符比例为准，报告中的 token 数为估算值。

export function estimateTokens(text) {
  if (!text) return 0
  let cjk = 0
  let other = 0
  for (const ch of text) {
    const code = ch.codePointAt(0)
    if ((code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf) || (code >= 0x3040 && code <= 0x30ff) || (code >= 0xac00 && code <= 0xd7af)) cjk++
    else other++
  }
  return Math.ceil(other / 4 + cjk * 1.0)
}

// 渐进的压缩率调度（HPC 课程思想：从宽松到严格，逐步利用上一步的结果）
export function curriculumRatio(step, totalSteps, targetRatio) {
  if (totalSteps <= 1) return targetRatio
  // 第 1 步目标 ≈ 1.0（轻微压缩），最后一步 = targetRatio
  const progress = step / totalSteps
  return Math.max(targetRatio, 1 - (1 - targetRatio) * progress)
}

// 构造 DCP-Agent 的压缩指令
export function buildCompressPrompt({
  text,
  targetRatio,       // 本步目标保留比例 (0,1]
  step, totalSteps,
  keyPoints,         // 必须保留的要点（可选）
}) {
  const system = [
    'You are DCP-Agent, a prompt/context compressor (LLM-DCP, arXiv:2504.11004).',
    'Compress the given text by removing redundant tokens while preserving all key information.',
    'Key requirements:',
    '1. Preserve essential content: named entities, numbers, dates, identifiers, code symbols, task instructions, and any user-specified key points.',
    '2. Keep the semantic content and structure (headings, steps, arguments) intact where it matters.',
    '3. Fluency and grammar may be sacrificed — LLMs tolerate compressed prompts well; brevity wins.',
    '4. Output ONLY the compressed text. No explanations, no notes, no quotes, no markdown fences.',
  ].join('\n')

  const pct = Math.round(targetRatio * 100)
  const stepNote = totalSteps > 1
    ? 'This is step ' + step + ' of ' + totalSteps + ' of an iterative compression (each step builds on the previous result). Compress further toward the final target.'
    : ''
  const keyNote = keyPoints && keyPoints.trim()
    ? 'MUST-keep key points (preserve these verbatim or in full detail):\n' + keyPoints.trim()
    : 'Preserve all key information: entities, numbers, identifiers, instructions, arguments.'

  const user = [
    'Compress the following text to about ' + pct + '% of its original length (' + Math.max(1, Math.round(targetRatio * 100)) + '% retention).',
    stepNote,
    keyNote,
    '\n--- TEXT TO COMPRESS ---\n' + text,
  ].filter(Boolean).join('\n')
  return { system, user }
}

// 一次 LLM 调用（DCP-Agent 的动作）
async function agentStep({ ctx, llm, options, system, user }) {
  const assembler = new llm.BlockAssembler()
  const streamOpts = {
    ...options,
    messages: [llm.createUserMessage({
      content: [{ type: 'text', text: user }],
      source: { kind: 'plugin', plugin: 'dsh-dcp' },
    })],
    system,
  }
  for await (const chunk of ctx.llm.stream(streamOpts)) {
    options.signal?.throwIfAborted()
    assembler.push(chunk)
  }
  const finish = assembler.finish
  if (finish.kind === 'error' || finish.kind === 'aborted') {
    const err = new Error('dcp-compress: llm call failed: ' + (finish.failure?.message ?? finish.kind))
    err.code = finish.failure?.code
    throw err
  }
  if (finish.kind === 'max-tokens') throw new Error('dcp-compress: llm output reached max-tokens')
  if (finish.kind === 'tool-calls') throw new Error('dcp-compress: llm unexpectedly requested a tool call')
  const blocks = assembler.blocks()
  const out = blocks.filter((b) => b.type === 'text').map((b) => b.text).join('').trim()
  if (!out) throw new Error('dcp-compress: agent produced empty output at step')
  return out
}

// 校验一步压缩结果：允许轻微超预算（下一步继续），拒绝空/退化输出
function validateStep({ original, compressed, targetChars }) {
  if (!compressed) return { ok: false, reason: 'empty' }
  if (compressed.trim() === original.trim()) return { ok: false, reason: 'no-progress' }
  if (compressed.length >= original.length) return { ok: false, reason: 'no-reduction' }
  return { ok: true }
}

/**
 * 主入口：按 LLM-DCP 的 MDP 框架迭代压缩。
 * @param {object} deps
 * @param {object} deps.ctx     - Cordis 上下文（ctx.llm 可用）
 * @param {object} deps.llm     - '@deepseek-ai/dsh-llm' 模块（BlockAssembler / createUserMessage）
 * @param {string} deps.text    - 待压缩文本
 * @param {number} deps.targetRatio - 目标保留比例 (0,1]，默认 0.5
 * @param {number} deps.steps   - MDP 轨迹最大步数，默认 2
 * @param {string} [deps.keyPoints] - 必须保留的要点
 * @param {object} [deps.route] - { provider?, model?, maxTokens?, signal?, sessionId? }
 * @param {(info: object) => void} [deps.onStep] - 每步回调（日志/报告）
 */
export async function compressWithAgent({
  ctx, llm, text,
  targetRatio = 0.5,
  steps = 2,
  keyPoints,
  route = {},
  onStep,
}) {
  if (!text || !text.trim()) throw new Error('dcp-compress: text is required')
  if (!(targetRatio > 0 && targetRatio <= 1)) throw new Error('dcp-compress: targetRatio must be in (0, 1]')
  const nSteps = Math.max(1, Math.min(5, Math.floor(steps || 2)))

  const original = text
  const originalChars = original.length
  const originalTokens = estimateTokens(original)
  const targetChars = Math.max(1, Math.round(originalChars * targetRatio))

  const options = {
    provider: route.provider,
    model: route.model,
    maxTokens: route.maxTokens,
    sessionId: route.sessionId,
    purpose: 'dcp-compress',
    signal: route.signal,
  }

  let current = original
  const perStep = []
  let stepsUsed = 0
  for (let t = 1; t <= nSteps; t++) {
    const ratio = curriculumRatio(t, nSteps, targetRatio)
    const stepTargetChars = Math.max(1, Math.round(originalChars * ratio))
    const { system, user } = buildCompressPrompt({
      text: current,
      targetRatio: ratio,
      step: t,
      totalSteps: nSteps,
      keyPoints,
    })
    const start = Date.now()
    const compressed = await agentStep({ ctx, llm, options, system, user })
    const elapsedMs = Date.now() - start
    const check = validateStep({ original, compressed, targetChars: stepTargetChars })
    if (!check.ok) {
      if (t === 1) throw new Error('dcp-compress: agent failed to compress at step 1: ' + check.reason)
      // 非第一步失败：终止并返回已压缩结果（保守回退）
      stepsUsed = t - 1
      break
    }
    const chars = compressed.length
    const ratioAchieved = originalChars > 0 ? chars / originalChars : 0
    perStep.push({
      step: t,
      targetRatio: ratio,
      chars,
      ratioAchieved,
      elapsedMs,
    })
    onStep?.({
      step: t,
      current: compressed,
      ratio,
      chars,
      ratioAchieved,
    })
    current = compressed
    stepsUsed = t
    // 已达标：提前终止（避免过度压缩）
    if (ratioAchieved <= targetRatio * 1.15) break
  }

  return {
    text: current,
    originalChars,
    originalTokens,
    compressedChars: current.length,
    compressedTokens: estimateTokens(current),
    ratio: originalChars > 0 ? current.length / originalChars : 0,
    targetRatio,
    steps: nSteps,
    stepsUsed,
    perStep,
  }
}
