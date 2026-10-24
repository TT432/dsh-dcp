// dcp_compress：LLM-DCP (Dynamic Compressing Prompts, arXiv:2504.11004) 的 dsh 工具。
// 用 ctx.llm 的可配置 LLM 充当 DCP-Agent，按论文的 MDP 框架迭代压缩长 prompt/上下文：
// 多步逐步压缩（每步基于上一步的动态上下文）、目标压缩率随轨迹收紧（HPC 课程思想）、
// 关键信息显式保留（实体/数字/标识符/指令要点）。
import { defineTool } from '@deepseek-ai/dsh-tools'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { compressWithAgent } from '../dcp/compress.js'

export function registerCompressTool(ctx) {
  ctx.tools.register(defineTool({
    name: 'dcp_compress',
    description: 'LLM-DCP (Dynamic Compressing Prompts, arXiv:2504.11004) 压缩器：用 LLM 充当 DCP-Agent，' +
      '按 MDP 框架多步迭代压缩长 prompt/上下文——逐步删除冗余 token、保留关键信息（实体/数字/标识符/指令），' +
      '目标压缩率随轨迹收紧。适合在调用长上下文模型前压缩提示词、文档或会话上下文。',
    parameters: {
      text: { type: 'string', required: true, description: '待压缩的文本（prompt / 上下文 / 文档）' },
      targetRatio: { type: 'number', description: '目标保留比例 0-1（默认 0.5，越小压缩越狠）' },
      steps: { type: 'integer', description: 'MDP 迭代步数 1-5（默认 2，多步逐步压缩、每步基于上一步结果）' },
      keyPoints: { type: 'string', description: '必须保留的关键要点（换行分隔；实体/数字/指令等）' },
      provider: { type: 'string', description: 'LLM provider 覆盖（默认用 harness 配置）' },
      model: { type: 'string', description: 'LLM model 覆盖（需与 provider 同时给出）' },
      maxTokens: { type: 'integer', description: '单次生成最大 token 数（默认由 provider 决定）' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{
        type: 'text',
        text: renderCompress(value),
      }],
    },
    async execute(args, exec) {
      const llm = { BlockAssembler }
      const route = resolveRoute(args, exec)
      if (!route.provider || !route.model) {
        // ctx.llm.stream 需要显式 provider/model 选择 adapter；缺失时无法自动路由。
        // 显式参数优先，其次当前会话的实际请求路由（agent loop 每轮记录 request/context）。
        throw new Error('dcp-compress: 无法确定 LLM 路由（provider/model 缺失）：' +
          '请显式传 provider/model，或在 agent 会话中调用以便从会话路由解析')
      }
      return compressWithAgent({
        ctx,
        llm,
        text: args.text,
        targetRatio: args.targetRatio ?? 0.5,
        steps: args.steps ?? 2,
        keyPoints: args.keyPoints,
        route,
      })
    },
  }))
}

// 解析 LLM 路由：显式参数优先；否则回退到当前 agent 会话实际使用的请求路由
// （dsh-agent-loop 每轮请求都会把 { provider, model, contextWindow } 记录为
// request/context 事件，session.requestContext() 返回最新一条）。这样
// "默认用 harness 配置" 在无 agent 的上下文（headless 等）里仍会清晰报错。
function resolveRoute(args, exec) {
  const sessionRoute = exec?.agent?.session?.requestContext?.()
  return {
    provider: args.provider ?? sessionRoute?.provider,
    model: args.model ?? sessionRoute?.model,
    maxTokens: args.maxTokens,
    signal: exec?.signal,
  }
}

function renderCompress(v) {
  const lines = []
  lines.push('==== LLM-DCP 压缩结果 ====')
  lines.push('长度: ' + v.originalChars + ' 字符 (≈' + v.originalTokens + ' tokens) → ' + v.compressedChars + ' 字符 (≈' + v.compressedTokens + ' tokens)')
  lines.push('压缩率: 保留 ' + (v.ratio * 100).toFixed(1) + '% (目标 ' + (v.targetRatio * 100).toFixed(0) + '%, 压缩比 ' + (v.ratio > 0 ? (1 / v.ratio).toFixed(1) : '∞') + 'x)')
  lines.push('MDP 步数: ' + v.stepsUsed + '/' + v.steps + (v.perStep.length ? '（每步: ' + v.perStep.map(s => s.step + ': ' + (s.ratioAchieved * 100).toFixed(0) + '%').join(' → ') + '）' : ''))
  lines.push('')
  lines.push('---- 压缩后文本 ----')
  lines.push(v.text)
  return lines.join('\n')
}
