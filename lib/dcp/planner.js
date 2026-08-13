// 端到端规划：块生成 → 超图划分放置 → division 调度 → 模拟执行 → 与静态基线对比。
// 对应论文 Fig.8 的 planner 模块（block generation + hypergraph partitioning + scheduling），
// 并用论文同款成本模型做微基准模拟（Fig.13/14 的估计）。
import { makeContext, DType, sum, max as maxOf } from './core.js'
import { generateWorkload } from './blockgen.js'
import { placeBlocks } from './partition.js'
import { scheduleDivisions } from './scheduler.js'
import { simulatePlan } from './simulator.js'
import { simulateZigzag, simulateDp } from './baselines.js'
import { AttnRooflineCostModel, CommunicationCostModel } from './costmodel.js'
import { maskLabel } from './masks.js'

export function planBatch({
  seqlens,
  blockSize,
  headBlockSize,
  nHeads,
  nQueryGroups,
  headDim,
  dtype = DType.BF16,
  maskSpec = { type: 'causal' },
  context = null,
  compEpsilon = 0.2,
  memEpsilon = 0.2,
  interNodeCompEpsilonFactor = 5,
  nDivisions = 4,
  includeBackward = false,
  maxTflops = 280,
  maxMemBwGBs = 1500,
} = {}) {
  const ctx = context || makeContext()
  if (!seqlens || seqlens.length === 0) throw new Error('seqlens must be a non-empty array')
  for (const L of seqlens) if (!Number.isInteger(L) || L <= 0) throw new Error('seqlens must be positive integers')

  const isGqa = nHeads > nQueryGroups
  if (!headBlockSize) headBlockSize = isGqa ? nHeads / nQueryGroups : nHeads
  const bs = blockSize || pickBlockSize(seqlens, ctx.nTotalDevices)

  const costModel = new AttnRooflineCostModel({ maxTflops, maxMemBwGBs })
  const commCostModel = new CommunicationCostModel({
    intraBandwidthGbps: ctx.intraBandwidthGbps,
    interBandwidthGbps: ctx.interBandwidthGbps,
    intraLatencyMs: ctx.intraLatencyMs,
    interLatencyMs: ctx.interLatencyMs,
  })

  const workload = generateWorkload({
    seqlens, blockSize: bs, headBlockSize,
    nHeads, nQueryGroups, headDim, dtype, maskSpec, costModel,
  })

  const placement = placeBlocks(ctx, workload, { compEpsilon, memEpsilon, interNodeCompEpsilonFactor })
  const schedule = scheduleDivisions(ctx, workload, placement, { nDivisions })
  const sim = simulatePlan(ctx, workload, placement, schedule, { commCostModel, costModel, includeBackward })
  const zigzag = simulateZigzag(ctx, workload, { commCostModel, costModel })
  const dp = simulateDp(ctx, workload, { costModel })

  const totalCommBytes = placement.stats.totalBytes + schedule.totalOutputCommBytes
  const dcpTotal = sim.totalMs
  const speedupVsZigzag = dcpTotal > 0 ? zigzag.totalMs / dcpTotal : null
  const speedupVsDp = dcpTotal > 0 ? dp.totalMs / dcpTotal : null

  return {
    meta: {
      seqlens,
      nSequences: seqlens.length,
      nTokensTotal: workload.nTokensTotal,
      mask: maskLabel(maskSpec),
      maskSpec,
      cluster: { nodes: ctx.nNodes, devicesPerNode: ctx.nDevicesPerNode, nDevices: ctx.nTotalDevices },
      model: { nHeads, nQueryGroups, headDim, dtype, blockSize: bs, headBlockSize },
      hyperparams: { compEpsilon, memEpsilon, interNodeCompEpsilonFactor, nDivisions, maxTflops, maxMemBwGBs },
    },
    blocks: {
      blockSize: bs,
      nQBlocks: workload.inputs.filter(m => m.type === 'Q').length,
      nKVBlocks: workload.inputs.filter(m => m.type === 'KV').length,
      nSeqBlocks: workload.nSeqBlocksTotal,
      nHeadBlocksQ: workload.nHeadBlocksQ,
      nHeadBlocksKV: workload.nHeadBlocksKV,
      nInputBlocks: workload.nInputs,
      nOutputBlocks: workload.nOutputs,
      nWorkBlocks: workload.nWork,
      nColocationGroups: workload.colocationConstraints.length,
      avgWorkSparsity: workload.nWork ? sum(workload.work.map(w => w.sparsity)) / workload.nWork : 0,
    },
    placement: {
      workToDevice: placement.workToDevice,
      inputToDevice: placement.inputToDevice,
      outputToDevice: placement.outputToDevice,
      stats: placement.stats,
    },
    schedule: {
      nDivisions: schedule.nDivisions,
      divisionStats: schedule.divisionStats,
      nOutputTransfers: schedule.outputTransfers.length,
      nReductions: schedule.reductions.length,
      totalInputCommBytes: schedule.totalInputCommBytes,
      totalOutputCommBytes: schedule.totalOutputCommBytes,
    },
    executionPlan: schedule.planPerDevice,
    simulation: {
      dcpMs: sim.totalMs,
      dcpFwdMs: sim.fwdMs,
      dcpBwMs: sim.bwMs,
      perDeviceMs: sim.perDeviceMs,
      breakdown: sim.breakdown,
      zigzagMs: zigzag.totalMs,
      zigzagPerDeviceMs: zigzag.perDeviceMs,
      dpMs: dp.totalMs,
      dpPerDeviceMs: dp.perDeviceMs,
      speedupVsZigzag,
      speedupVsDp,
    },
    metrics: {
      totalCommBytes,
      interNodeCommBytes: placement.stats.interNodeBytes,
      intraNodeCommBytes: placement.stats.intraNodeBytes,
      zigzagTotalCommBytes: zigzag.totalCommBytes,
      dcpComputeImbalance: computeImbalance(placement.stats.deviceCompute),
      dcpMemImbalance: computeImbalance(placement.stats.deviceMem),
      zigzagComputeImbalance: zigzag.computeImbalance,
      zigzagMemImbalance: zigzag.memImbalance,
      dpComputeImbalance: dp.computeImbalance,
      dpMemImbalance: dp.memImbalance,
      dcpTotalCommTimeMs: sim.breakdown ? sum(sim.breakdown.filter(Boolean).map(b => b.commMs)) : 0,
    },
  }
}

export function computeImbalance(loads) {
  const s = sum(loads)
  if (s <= 0) return 1
  const avg = s / loads.length
  return maxOf(loads) / avg
}

// 简易块大小启发式：目标总块数在 [8*devices, 64*devices] 之间
export function pickBlockSize(seqlens, nDevices) {
  const total = sum(seqlens)
  const targetBlocks = Math.min(Math.max(8 * nDevices, 64), 1024)
  const bs = Math.ceil(total / targetBlocks)
  // 对齐到 2 的幂附近（论文搜索 512/1024/2048/4096）
  const powers = [256, 512, 1024, 2048, 4096, 8192]
  for (const p of powers) if (bs <= p) return p
  return 8192
}