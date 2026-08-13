// 模拟执行：按 division 结构构建每设备时间线，通信与下一 division 计算重叠。
// 模型与论文 §4.3 一致：comm(div i+1) 与 compute(div i) 重叠；
// 每设备耗时 T_d = c_0 + sum_{i=1..T-1} max(c_i, k_i) + 输出通信 + reduction。
// 迭代总耗时 = max 设备耗时（可选 fwd+bwd）。
import { deviceKey } from './core.js'

export function simulatePlan(context, workload, placement, schedule, {
  commCostModel,
  costModel,
  includeBackward = false,
} = {}) {
  const nDevices = context.nTotalDevices
  const devs = []
  for (let n = 0; n < context.nNodes; n++)
    for (let d = 0; d < context.nDevicesPerNode; d++) devs.push([n, d])

  // 每设备每 division 的计算/通信耗时
  const computePerDiv = new Array(nDevices)
  const commPerDiv = new Array(nDevices)
  for (let i = 0; i < nDevices; i++) {
    computePerDiv[i] = []
    commPerDiv[i] = []
  }
  for (const [di, div] of schedule.divisions.entries()) {
    const compute = new Array(nDevices).fill(0)
    const comm = new Array(nDevices).fill(0)
    for (const w of div.work) {
      const d = devIdxOf(placement.workToDevice[w], context)
      compute[d] += workload.work[w].costMs
    }
    for (const c of div.comm) {
      const d = devIdxOf(c.dst, context)
      comm[d] += commCostModel.cost(c.src, c.dst, c.bytes)
    }
    for (let i = 0; i < nDevices; i++) {
      computePerDiv[i].push(compute[i])
      commPerDiv[i].push(comm[i])
    }
  }

  // 输出通信与 reduction
  const outCommPerDevice = new Array(nDevices).fill(0)
  for (const t of schedule.outputTransfers) {
    const d = devIdxOf(t.src, context)
    outCommPerDevice[d] += commCostModel.cost(t.src, t.dst, t.bytes)
  }
  // reduction 近似：输出设备上每个被 reduction 的输出块一次融合 kernel，成本按 0 计（被通信覆盖）
  const reductionPerDevice = new Array(nDevices).fill(0)
  for (const { device } of schedule.reductions) {
    reductionPerDevice[Number(device)] = 0
  }

  const perDeviceMs = new Array(nDevices)
  const breakdown = new Array(nDevices)
  for (let d = 0; d < nDevices; d++) {
    const cs = computePerDiv[d]
    const ks = commPerDiv[d]
    if (cs.length === 0) { perDeviceMs[d] = 0; breakdown[d] = null; continue }
    let total = cs[0]
    for (let i = 1; i < cs.length; i++) total += Math.max(cs[i], ks[i])
    // 输出通信与最后一个 division 的计算重叠（官方 pipeline 将输出阶段纳入 stage 内），只暴露残余
    const lastCompute = cs[cs.length - 1] ?? 0
    total += Math.max(0, outCommPerDevice[d] - lastCompute) + reductionPerDevice[d]
    breakdown[d] = {
      computeMs: cs.reduce((a, b) => a + b, 0),
      commMs: ks.reduce((a, b) => a + b, 0) + outCommPerDevice[d],
      overlappedMs: Math.max(0, ks.slice(1).reduce((a, b, i) => a + Math.min(b, cs[i + 1]), 0)),
    }
    perDeviceMs[d] = total
  }

  let totalMs = Math.max(...perDeviceMs)
  let bwTotalMs = null
  let perDeviceBwMs = null
  if (includeBackward) {
    // 近似：bwd 用同一 division 结构，成本取 bwd roofline（官方为每个迭代单独生成 bw plan）
    perDeviceBwMs = new Array(nDevices)
    for (let d = 0; d < nDevices; d++) {
      const cs = new Array(schedule.divisions.length).fill(0)
      for (const [di, div] of schedule.divisions.entries()) {
        for (const w of div.work) {
          if (devIdxOf(placement.workToDevice[w], context) === d) {
            const m = workload.work[w]
            const c = costModel.getCost({
              seqlensQ: [workload.blockSize], seqlensK: [workload.blockSize],
              rangesForTokens: workload.rangesPerSeq.get(m.seqId),
              qStart: m.qBlockId * workload.blockSize,
              qEnd: Math.min((m.qBlockId + 1) * workload.blockSize, workload.seqlens[m.seqId]),
              kvStart: m.kvBlockId * workload.blockSize,
              kvEnd: Math.min((m.kvBlockId + 1) * workload.blockSize, workload.seqlens[m.seqId]),
              paddedQ: workload.blockSize, paddedK: workload.blockSize,
              nHeads: workload.nHeads, nQueryGroups: workload.nQueryGroups, headDim: workload.headDim,
              isForward: false,
            })
            cs[di] += Math.max(c.flopsTimeMs, c.memTimeMs)
          }
        }
      }
      const ks = commPerDiv[d]
      let total = cs[0] ?? 0
      for (let i = 1; i < cs.length; i++) total += Math.max(cs[i], ks[i] ?? 0)
      const lastC = cs[cs.length - 1] ?? 0
      total += Math.max(0, outCommPerDevice[d] - lastC)
      perDeviceBwMs[d] = total
    }
    bwTotalMs = Math.max(...perDeviceBwMs)
  }

  return {
    perDeviceMs,
    totalMs: bwTotalMs === null ? totalMs : totalMs + bwTotalMs,
    fwdMs: totalMs,
    bwMs: bwTotalMs,
    breakdown,
    computePerDiv,
    commPerDiv,
    outCommPerDevice,
  }
}

export function devIdxOf(dev, context) {
  return dev[0] * context.nDevicesPerNode + dev[1]
}