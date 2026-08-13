// dsh-dcp：DCP (Dynamic Context Parallelism, SOSP'25, arXiv:2510.10620) 的 dsh 插件。
// 提供两个面向模型的工具：
//   - dcp_plan:      对一批序列生成动态上下文并行规划（块生成→超图放置→division 调度→模拟执行）
//   - dcp_benchmark: 批量抽样 micro-benchmark，汇总 vs 静态 CP/DP 基线的加速比与通信降低
//
// 论文: Jiang et al., "DCP: Addressing Input Dynamism In Long-Context Training via Dynamic
//       Context Parallelism", SOSP 2025, https://arxiv.org/abs/2510.10620
import { registerPlanTool } from './tools/plan.js'
import { registerBenchmarkTool } from './tools/benchmark.js'
import { registerCompressTool } from './tools/compress.js'

export const name = 'dsh-dcp'

// cordis v4：访问 ctx.tools / ctx.llm 前必须声明 inject，否则抛
// "cannot get property ... without inject"
export const inject = ['tools', 'llm']

export function apply(ctx) {
  registerPlanTool(ctx)
  registerBenchmarkTool(ctx)
  registerCompressTool(ctx)
  // 注册均为可逆副作用，插件 dispose 时自动注销
}
