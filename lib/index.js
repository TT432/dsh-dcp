// dsh-dcp：LLM-DCP (Dynamic Compressing Prompts, arXiv:2504.11004) 的 dsh 插件。
// 提供面向模型的工具：
//   - dcp_compress: 用可配置 LLM 充当 DCP-Agent，按 MDP 框架多步迭代压缩长 prompt/上下文
//
// 论文: Xu et al., "LLM-DCP: Dynamic Compressing Prompts via LLM as DCP-Agent",
//       https://arxiv.org/abs/2504.11004
import { registerCompressTool } from './tools/compress.js'

export const name = 'dsh-dcp'

// cordis v4：访问 ctx.tools / ctx.llm 前必须声明 inject，否则抛
// "cannot get property ... without inject"
export const inject = ['tools', 'llm']

export function apply(ctx) {
  registerCompressTool(ctx)
  // 注册均为可逆副作用，插件 dispose 时自动注销
}
