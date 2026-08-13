// dcp_benchmark：按给定的序列长度分布抽样若干批次，逐批运行 DCP 规划并汇总
// 与静态基线（zigzag / dp）的加速比与通信对比（对应论文 §7 的 micro-benchmark 流程）。
import { defineTool } from '@deepseek-ai/dsh-tools'
import { planBatch } from '../dcp/planner.js'
import { makeContext, fmtBytes, fmtMs, fmtX } from '../dcp/core.js'

// 论文 Fig.2 风格的长尾分布：大部分短序列 + 少量长序列（log-normal 近似）
export function sampleSeqlens(rng, nSeqs, minLen, maxLen) {
  const out = []
  for (let i = 0; i < nSeqs; i++) {
    // log-normal: logLen ~ N(mu, sigma)；mu 取 min/max 的中点偏下
    const mu = Math.log(minLen) + 0.35 * (Math.log(maxLen) - Math.log(minLen))
    const sigma = 1.2
    let logLen = mu + sigma * gauss(rng)
    let L = Math.round(Math.exp(logLen))
    L = Math.min(maxLen, Math.max(minLen, L))
    out.push(L)
  }
  return out
}

function gauss(rng) {
  let u = 0, v = 0
  while (u === 0) u = rng()
  while (v === 0) v = rng()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function mulberry32(seed) {
  let a = seed >>> 0
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function registerBenchmarkTool(ctx) {
  ctx.tools.register(defineTool({
    name: 'dcp_benchmark',
    description: 'DCP micro-benchmark：抽样多个批次（默认 LongDataCollections 式长尾序列长度分布），' +
      '逐批运行 DCP 动态上下文并行规划，汇总 vs 静态 CP（zigzag）与 DP 的平均加速比、通信降低幅度与负载均衡。',
    parameters: {
      nBatches: { type: 'integer', description: '批次数量（默认 8）' },
      nSeqsPerBatch: { type: 'integer', description: '每批序列数（默认 24）' },
      minSeqLen: { type: 'integer', description: '序列长度下限（默认 1024）' },
      maxSeqLen: { type: 'integer', description: '序列长度上限（默认 65536）' },
      seed: { type: 'integer', description: '随机种子（默认 42）' },
      nHeads: { type: 'integer', description: '注意力头数（默认 8）' },
      nQueryGroups: { type: 'integer', description: 'KV 组数（默认 2）' },
      headDim: { type: 'integer', description: '单头维度（默认 128）' },
      mask: { type: 'string', enum: ['causal', 'lambda', 'causal_blockwise', 'shared_question'], description: 'mask 类型（默认 causal）' },
      sink: { type: 'integer', description: 'lambda sink（默认 64）' },
      window: { type: 'integer', description: 'lambda 窗口（默认 4096）' },
      nodes: { type: 'integer', description: '节点数（默认 1）' },
      devicesPerNode: { type: 'integer', description: '每节点设备数（默认 8）' },
      blockSize: { type: 'integer', description: '块大小（默认 512）' },
      headBlockSize: { type: 'integer', description: 'head 块大小（默认 4）' },
      nDivisions: { type: 'integer', description: 'division 数（默认 4）' },
    },
    output: {
      schema: { type: 'json' },
      render: (_args, value) => [{
        type: 'text',
        text: renderBenchmark(value),
      }],
    },
    async execute(args) {
      const nBatches = args.nBatches ?? 8
      const nSeqs = args.nSeqsPerBatch ?? 24
      const rng = mulberry32(args.seed ?? 42)
      const maskSpec = (() => {
        switch (args.mask ?? 'causal') {
          case 'causal': return { type: 'causal' }
          case 'lambda': return { type: 'lambda', sink: args.sink ?? 64, window: args.window ?? 4096 }
          case 'causal_blockwise': return { type: 'causal_blockwise', blockSize: 256, windowBlocks: 2, sinkBlocks: 1 }
          case 'shared_question': return { type: 'shared_question', questionTokens: 1638, answerTokens: 1638, nAnswers: 4 }
          default: return { type: 'causal' }
        }
      })()
      const ctx = makeContext({
        nodes: args.nodes ?? 1,
        devicesPerNode: args.devicesPerNode ?? 8,
      })
      const agg = {
        nBatches: 0,
        dcpMs: [], zigzagMs: [], dpMs: [],
        speedupVsZigzag: [], speedupVsDp: [],
        commDcp: [], commZigzag: [],
        dcpCompImb: [], dcpMemImb: [],
        tokens: [],
        perBatch: [],
      }
      for (let b = 0; b < nBatches; b++) {
        const seqlens = sampleSeqlens(rng, nSeqs, args.minSeqLen ?? 1024, args.maxSeqLen ?? 65536)
        try {
          const p = planBatch({
            seqlens,
            nHeads: args.nHeads ?? 8,
            nQueryGroups: args.nQueryGroups ?? 2,
            headDim: args.headDim ?? 128,
            blockSize: args.blockSize ?? 512,
            headBlockSize: args.headBlockSize ?? 4,
            maskSpec,
            context: ctx,
            nDivisions: args.nDivisions ?? 4,
          })
          agg.nBatches++
          agg.dcpMs.push(p.simulation.dcpMs)
          agg.zigzagMs.push(p.simulation.zigzagMs)
          agg.dpMs.push(p.simulation.dpMs)
          agg.speedupVsZigzag.push(p.simulation.speedupVsZigzag)
          agg.speedupVsDp.push(p.simulation.speedupVsDp)
          agg.commDcp.push(p.metrics.totalCommBytes)
          agg.commZigzag.push(p.metrics.zigzagTotalCommBytes)
          agg.dcpCompImb.push(p.metrics.dcpComputeImbalance)
          agg.dcpMemImb.push(p.metrics.dcpMemImbalance)
          agg.tokens.push(p.meta.nTokensTotal)
          agg.perBatch.push({
            batch: b,
            nSeqs: seqlens.length,
            tokens: p.meta.nTokensTotal,
            dcpMs: p.simulation.dcpMs,
            zigzagMs: p.simulation.zigzagMs,
            dpMs: p.simulation.dpMs,
            speedupVsZigzag: p.simulation.speedupVsZigzag,
            speedupVsDp: p.simulation.speedupVsDp,
            commDcp: p.metrics.totalCommBytes,
            commZigzag: p.metrics.zigzagTotalCommBytes,
          })
        } catch (e) {
          agg.perBatch.push({ batch: b, error: String(e && e.message || e) })
        }
      }
      return finalizeAggregate(agg, maskSpec, args)
    },
  }))
}

function mean(arr) {
  if (arr.length === 0) return 0
  return arr.reduce((a, b) => a + b, 0) / arr.length
}

function finalizeAggregate(agg, maskSpec, args) {
  const n = agg.nBatches
  const geomean = (arr) => {
    if (arr.length === 0) return 0
    const l = arr.reduce((a, b) => a + Math.log(Math.max(b, 1e-9)), 0)
    return Math.exp(l / arr.length)
  }
  return {
    config: {
      mask: maskSpec.type,
      nBatches: agg.nBatches,
      nSeqsPerBatch: args.nSeqsPerBatch ?? 24,
      cluster: args.nodes ?? 1 + 'x' + (args.devicesPerNode ?? 8),
      blockSize: args.blockSize ?? 512,
    },
    aggregate: {
      nBatches: agg.nBatches,
      meanTokens: Math.round(mean(agg.tokens)),
      meanDcpMs: mean(agg.dcpMs),
      meanZigzagMs: mean(agg.zigzagMs),
      meanDpMs: mean(agg.dpMs),
      meanSpeedupVsZigzag: mean(agg.speedupVsZigzag),
      geomeanSpeedupVsZigzag: geomean(agg.speedupVsZigzag),
      meanSpeedupVsDp: mean(agg.speedupVsDp),
      meanCommDcp: mean(agg.commDcp),
      meanCommZigzag: mean(agg.commZigzag),
      commReductionVsZigzag: mean(agg.commZigzag) > 0 ? 1 - mean(agg.commDcp) / mean(agg.commZigzag) : 0,
      meanDcpCompImbalance: mean(agg.dcpCompImb),
      meanDcpMemImbalance: mean(agg.dcpMemImb),
      worstSpeedupVsZigzag: Math.min(...agg.speedupVsZigzag),
      bestSpeedupVsZigzag: Math.max(...agg.speedupVsZigzag),
    },
    perBatch: agg.perBatch,
  }
}

function renderBenchmark(value) {
  const a = value.aggregate
  const lines = []
  lines.push('==== DCP Micro-Benchmark (' + value.config.mask + ', ' + value.config.nBatches + ' 批 x ' + value.config.nSeqsPerBatch + ' 序列, 集群 ' + value.config.cluster + ') ====')
  lines.push('平均 tokens/批: ' + a.meanTokens)
  lines.push('平均耗时: DCP ' + fmtMs(a.meanDcpMs) + ' | zigzag ' + fmtMs(a.meanZigzagMs) + ' | dp ' + fmtMs(a.meanDpMs))
  lines.push('加速比 (vs zigzag): 平均 ' + fmtX(a.meanSpeedupVsZigzag) + ' (几何平均 ' + fmtX(a.geomeanSpeedupVsZigzag) + ', 最差 ' + fmtX(a.worstSpeedupVsZigzag) + ', 最好 ' + fmtX(a.bestSpeedupVsZigzag) + ')')
  lines.push('加速比 (vs dp): ' + fmtX(a.meanSpeedupVsDp))
  lines.push('通信: DCP ' + fmtBytes(a.meanCommDcp) + ' vs zigzag ' + fmtBytes(a.meanCommZigzag) + ' (降低 ' + (a.commReductionVsZigzag * 100).toFixed(1) + '%)')
  lines.push('负载失衡: 计算 ' + a.meanDcpCompImbalance.toFixed(3) + ', 内存 ' + a.meanDcpMemImbalance.toFixed(3))
  return lines.join('\n')
}