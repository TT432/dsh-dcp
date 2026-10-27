// dsh-dcp：dsh 的 DCP（Dynamic Context Pruning）插件。
//
// 两条能力线：
//   1. 锚点压缩（对齐本机 omp 的 dcp 扩展，~/.omp/agent/extensions/dcp）：
//      add_anchor / compress 两个工具 + 超阈值提醒。规格见 docs/anchor-compression.md。
//      压缩是 append-only 的表面替换（compaction/prune 影子计价 + user/message replace），
//      原始会话事件永不删除；resume 后派生历史自动保持压缩结果。
//   2. LLM-DCP（arXiv:2504.11004）提示词压缩工具 dcp_compress：用可配置 LLM 充当
//      DCP-Agent 多步迭代压缩文本，与锚点压缩是不同能力，保留不动。
import { Logger, isDebugEnabled } from './anchor/logger.js'
import { parseRemindThreshold, registerReminder } from './anchor/remind.js'
import { registerAnchorTools } from './tools/anchor.js'
import { registerCompressTool } from './tools/compress.js'

export const name = 'dsh-dcp'

// cordis v4：访问 ctx.tools / ctx.llm 前必须声明 inject，否则抛
// "cannot get property ... without inject"。tokenMeter 是可选服务，用 ctx.get() 取。
export const inject = ['tools', 'llm']

export function apply(ctx) {
  const logger = new Logger({ debug: isDebugEnabled(process.env.DCP_DEBUG), ctx })
  registerCompressTool(ctx)
  registerAnchorTools(ctx, { logger })
  registerReminder(ctx, { threshold: parseRemindThreshold(process.env.DCP_REMIND_THRESHOLD), logger })
  logger.info('DCP initialized (anchor compression + LLM-DCP prompt compression)', {
    debug: logger.debugEnabled,
    remindThreshold: parseRemindThreshold(process.env.DCP_REMIND_THRESHOLD),
  })
  // 注册均为可逆副作用，插件 dispose 时自动注销
}
