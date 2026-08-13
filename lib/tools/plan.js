// dcp_plan：对一批序列生成 DCP 动态上下文并行规划（块生成→超图放置→division 调度→模拟），
// 并与静态 CP 基线（zigzag）和 DP 基线对比。
import { defineTool } from '@deepseek-ai/dsh-tools'
import { planBatch } from '../dcp/planner.js'
import { makeContext, fmtBytes, fmtMs, fmtX, fmtPct } from '../dcp/core.js'

function maskSpecFromArgs(a) {
  switch (a.mask) {
    case 'causal': return { type: 'causal' }
    case 'lambda': return { type: 'lambda', sink: a.sink ?? 64, window: a.window ?? 4096 }
    case 'causal_blockwise': return { type: 'causal_blockwise', blockSize: a.maskBlockSize ?? 256, windowBlocks: a.windowBlocks ?? 2, sinkBlocks: a.sinkBlocks ?? 1 }
    case 'shared_question': return { type: 'shared_question', questionTokens: a.questionTokens ?? 1638, answerTokens: a.answerTokens ?? 1638, nAnswers: a.nAnswers ?? 4 }
    default: return { type: 'causal' }
  }
}

export function renderPlan(args, value) {
  const lines = []
  const m = value.meta
  const sim = value.simulation
  lines.push('==== DCP 动态上下文并行规划 ====')
  lines.push('批次: ' + m.nSequences + ' 条序列 / ' + m.nTokensTotal + ' tokens, mask=' + m.mask)
  lines.push('集群: ' + m.cluster.nodes + ' 节点 x ' + m.cluster.devicesPerNode + ' 设备 = ' + m.cluster.nDevices + ' GPU')
  lines.push('模型: H=' + m.model.nHeads + ' G=' + m.model.nQueryGroups + ' D=' + m.model.headDim +
    ', blockSize=' + m.model.blockSize + ', headBlockSize=' + m.model.headBlockSize)
  const b = value.blocks
  lines.push('块: Q=' + b.nQBlocks + ' KV=' + b.nKVBlocks + ' Out/LSE=' + b.nOutputBlocks +
    ' work=' + b.nWorkBlocks + ' colocation组=' + b.nColocationGroups +
    ' (平均 work 稀疏度 ' + fmtPct(b.avgWorkSparsity) + ')')
  const ps = value.placement.stats
  lines.push('通信: 总 ' + fmtBytes(value.metrics.totalCommBytes) +
    ' (跨节点 ' + fmtBytes(ps.interNodeBytes) + ' + 节点内 ' + fmtBytes(ps.intraNodeBytes) + ')' +
    '; zigzag 基线 ' + fmtBytes(value.metrics.zigzagTotalCommBytes))
  lines.push('负载: 计算失衡 ' + value.metrics.dcpComputeImbalance.toFixed(3) +
    ' (zigzag ' + value.metrics.zigzagComputeImbalance.toFixed(2) + ', dp ' + value.metrics.dcpComputeImbalance.toFixed(2) + ')' +
    '; 内存失衡 ' + value.metrics.dcpMemImbalance.toFixed(3))
  lines.push('')
  lines.push('---- 模拟执行 (attention 层) ----')
  lines.push('DCP    : ' + fmtMs(sim.dcpMs) + (sim.dcpBwMs != null ? ' (fwd ' + fmtMs(sim.dcpFwdMs) + ' + bwd ' + fmtMs(sim.dcpBwMs) + ')' : ''))
  lines.push('zigzag : ' + fmtMs(sim.zigzagMs) + '   dp: ' + fmtMs(sim.dpMs))
  lines.push('加速比 : vs zigzag ' + fmtX(sim.speedupVsZigzag) + (sim.speedupVsDp != null ? '  vs dp ' + fmtX(sim.speedupVsDp) : ''))
  lines.push('设备耗时分布 (ms): ' + sim.perDeviceMs.map(x => x.toFixed(2)).join(' '))
  lines.push('')
  lines.push('---- 调度 ----')
  lines.push('divisions=' + value.schedule.nDivisions + ', 输出传输=' + value.schedule.nOutputTransfers + ', reduction=' + value.schedule.nReductions)
  lines.push('每设备指令计划示例 (设备0):')
  const plan0 = value.executionPlan[0] || []
  for (const ins of plan0.slice(0, 8)) {
    if (ins.type === 'attention') lines.push('  attention x' + ins.work.length + ' blocks')
    else if (ins.type === 'comm_launch') lines.push('  comm_launch ' + ins.ops.length + ' ops')
    else if (ins.type === 'comm_wait') lines.push('  comm_wait')
    else if (ins.type === 'reduction') lines.push('  reduction x' + ins.outIds.length)
    else lines.push('  ' + ins.type)
  }
  if (plan0.length > 8) lines.push('  ... 共 ' + plan0.length + ' 条指令')
  return [{ type: 'text', text: lines.join('\n') }]
}

export function registerPlanTool(ctx) {
  ctx.tools.register(defineTool({
    name: 'dcp_plan',
    description: 'DCP (SOSP\'25 Dynamic Context Parallelism) 规划器：给定一批序列长度与注意力 mask，' +
      '生成细粒度块划分、超图划分的设备放置、division 调度与执行计划，并用 roofline+通信成本模型' +
      '模拟 attention 层执行，对比静态 CP（zigzag）与 DP 基线的加速比。论文：arxiv 2510.10620。',
    parameters: {
      seqlens: { type: 'array', items: { type: 'integer' }, required: true, description: '每条序列的 token 长度' },
      nHeads: { type: 'integer', required: true, description: '注意力头数（张量并行后的 Q 头数）' },
      nQueryGroups: { type: 'integer', required: true, description: 'KV 组数（GQA）' },
      headDim: { type: 'integer', required: true, description: '单头维度' },
      mask: { type: 'string', enum: ['causal', 'lambda', 'causal_blockwise', 'shared_question'], description: '注意力 mask 类型（默认 causal）' },
      sink: { type: 'integer', description: 'lambda mask 的 attention sink 数（默认 64）' },
      window: { type: 'integer', description: 'lambda mask 的滑窗大小（默认 4096）' },
      maskBlockSize: { type: 'integer', description: 'causal_blockwise 的块大小（默认 256）' },
      windowBlocks: { type: 'integer', description: 'causal_blockwise 的滑窗块数（默认 2）' },
      sinkBlocks: { type: 'integer', description: 'causal_blockwise 的 sink 块数（默认 1）' },
      questionTokens: { type: 'integer', description: 'shared_question 的问题长度（默认 1638）' },
      answerTokens: { type: 'integer', description: 'shared_question 每个回答长度（默认 1638）' },
      nAnswers: { type: 'integer', description: 'shared_question 回答数（默认 4）' },
      blockSize: { type: 'integer', description: 'seq 维块大小（默认按启发式选择）' },
      headBlockSize: { type: 'integer', description: 'head 维块大小（默认 headsPerQueryGroup）' },
      nodes: { type: 'integer', description: '节点数（默认 1）' },
      devicesPerNode: { type: 'integer', description: '每节点设备数（默认 1）' },
      intraBandwidthGbps: { type: 'number', description: '节点内带宽 Gbps（默认 4800）' },
      interBandwidthGbps: { type: 'number', description: '节点间带宽 Gbps（默认 400）' },
      compEpsilon: { type: 'number', description: '计算失衡容差（默认 0.2，节点内按 0.1 收紧）' },
      memEpsilon: { type: 'number', description: '内存失衡容差（默认 0.2，按 0.05 收紧）' },
      nDivisions: { type: 'integer', description: 'division 数（默认 4）' },
      includeBackward: { type: 'boolean', description: '是否包含反向传播模拟（默认 false）' },
      maxTflops: { type: 'number', description: 'GPU 峰值 TFLOPS（默认 280）' },
      maxMemBwGBs: { type: 'number', description: 'GPU 内存带宽 GB/s（默认 1500）' },
    },
    output: {
      schema: { type: 'json' },
      render: renderPlan,
    },
    async execute(args) {
      const ctx = makeContext({
        nodes: args.nodes ?? 1,
        devicesPerNode: args.devicesPerNode ?? 1,
        intraBandwidthGbps: args.intraBandwidthGbps ?? 4800,
        interBandwidthGbps: args.interBandwidthGbps ?? 400,
      })
      return planBatch({
        seqlens: args.seqlens,
        nHeads: args.nHeads,
        nQueryGroups: args.nQueryGroups,
        headDim: args.headDim,
        maskSpec: maskSpecFromArgs(args),
        blockSize: args.blockSize,
        headBlockSize: args.headBlockSize,
        context: ctx,
        compEpsilon: args.compEpsilon ?? 0.2,
        memEpsilon: args.memEpsilon ?? 0.2,
        nDivisions: args.nDivisions ?? 4,
        includeBackward: args.includeBackward ?? false,
        maxTflops: args.maxTflops ?? 280,
        maxMemBwGBs: args.maxMemBwGBs ?? 1500,
      })
    },
  }))
}
