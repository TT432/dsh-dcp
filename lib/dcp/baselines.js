// 静态基线，用于对比 DCP 的收益（对应论文 §7 的 micro-benchmark 对照）：
//   - zigzag：把每条序列均匀切成 2R 块（R=设备数），块 c 放在设备 c（c<R）或 2R-1-c（c>=R），
//     仅沿 seq 维并行（RingAttention 风格），KV 以 ring 方式流转，通信与计算重叠；
//   - dp：无上下文并行，序列按 token 贪心打包到设备，零通信但可能计算/内存失衡（论文 Fig.5(b)）。
import { deviceKey, max as maxOf, sum } from './core.js'
import { blockPairActive } from './masks.js'

// 每 token 的 Q+KV+Out 字节（用于设备内存估计）
export function perTokenBytes(workload) {
  const d = workload.dtype
  return (workload.nHeads + 2 * workload.nQueryGroups + workload.nHeads) * workload.headDim * d
}

// zigzag 块归属：块 c -> 设备 id
export function zigzagOwner(c, R) {
  return c < R ? c : 2 * R - 1 - c
}

export function simulateZigzag(context, workload, { commCostModel, costModel }) {
  const R = context.nTotalDevices
  const perDeviceCompute = new Array(R).fill(0)
  const perDeviceComm = new Array(R).fill(0)
  const perDeviceCommBytes = new Array(R).fill(0)
  const perDeviceMem = new Array(R).fill(0)
  const perDeviceMs = new Array(R).fill(0)
  const perToken = perTokenBytes(workload)

  for (let seqId = 0; seqId < workload.nSequences; seqId++) {
    const L = workload.seqlens[seqId]
    const nChunks = 2 * R
    const chunkTokens = Math.ceil(L / nChunks)
    const ranges = workload.rangesPerSeq.get(seqId)
    const tokensOf = (c) => Math.min(chunkTokens, L - c * chunkTokens)
    const startOf = (c) => c * chunkTokens

    // 每个设备的 owned Q chunks（本序列）
    const qChunksPerDevice = new Array(R)
    for (let d = 0; d < R; d++) qChunksPerDevice[d] = []
    for (let c = 0; c < nChunks; c++) qChunksPerDevice[zigzagOwner(c, R)].push(c)

    // 每个 (q chunk, kv chunk) 的注意力成本
    const costCache = new Map()
    const attnCost = (c, j) => {
      const key = c * 10000 + j
      let v = costCache.get(key)
      if (v === undefined) {
        const qStart = startOf(c), qEnd = Math.min(qStart + chunkTokens, L)
        const kStart = startOf(j), kEnd = Math.min(kStart + chunkTokens, L)
        const active = blockPairActive(ranges, qStart, qEnd, kStart, kEnd)
        if (!active) { v = 0 }
        else {
          const r = costModel.getCost({
            seqlensQ: [qEnd - qStart], seqlensK: [kEnd - kStart],
            rangesForTokens: ranges, qStart, qEnd, kvStart: kStart, kvEnd: kEnd,
            paddedQ: qEnd - qStart, paddedK: kEnd - kStart,
            nHeads: workload.nHeads, nQueryGroups: workload.nQueryGroups, headDim: workload.headDim,
          })
          v = Math.max(r.flopsTimeMs, r.memTimeMs)
        }
        costCache.set(key, v)
      }
      return v
    }

    for (let d = 0; d < R; d++) {
      const qChunks = qChunksPerDevice[d]
      if (qChunks.length === 0) continue
      // 需要的 kv chunk 集合
      const needed = new Set()
      for (const c of qChunks) {
        const qStart = startOf(c), qEnd = Math.min(qStart + chunkTokens, L)
        for (let j = 0; j < nChunks; j++) {
          const kStart = startOf(j), kEnd = Math.min(kStart + chunkTokens, L)
          if (blockPairActive(ranges, qStart, qEnd, kStart, kEnd)) needed.add(j)
        }
      }
      // 该设备的 token 数与内存
      let tokens = 0
      for (const c of qChunks) tokens += tokensOf(c)
      perDeviceMem[d] += tokens * perToken

      // 计算与通信（ring 模型：每步一个 kv chunk 到达）
      let total = 0
      for (const j of [...needed].sort((a, b) => a - b)) {
        const owner = zigzagOwner(j, R)
        let computeStep = 0
        for (const c of qChunks) computeStep += attnCost(c, j)
        let commStep = 0
        if (owner !== d) {
          const bytes = tokensOf(j) * 2 * workload.nQueryGroups * workload.headDim * workload.dtype
          commStep = commCostModel.cost([Math.floor(owner / (context.nDevicesPerNode || 1)), owner % (context.nDevicesPerNode || 1)], [Math.floor(d / (context.nDevicesPerNode || 1)), d % (context.nDevicesPerNode || 1)], bytes)
          perDeviceCommBytes[d] += bytes
          perDeviceComm[d] += commStep
        }
        total += Math.max(commStep, computeStep)
        perDeviceCompute[d] += computeStep
      }
      perDeviceMs[d] += total
    }
  }
  const totalMs = maxOf(perDeviceMs)
  return {
    name: 'zigzag',
    perDeviceMs,
    totalMs,
    perDeviceCompute,
    perDeviceComm,
    perDeviceCommBytes,
    perDeviceMem,
    maxCommBytes: maxOf(perDeviceCommBytes),
    totalCommBytes: sum(perDeviceCommBytes),
    computeImbalance: maxOf(perDeviceCompute) / (sum(perDeviceCompute) / R),
    memImbalance: maxOf(perDeviceMem) / (sum(perDeviceMem) / R),
  }
}

export function simulateDp(context, workload, { costModel }) {
  const R = context.nTotalDevices
  const perDeviceMs = new Array(R).fill(0)
  const perDeviceMem = new Array(R).fill(0)
  const perToken = perTokenBytes(workload)

  // 贪心打包：序列按 token 降序，放入当前负载最小设备
  const load = new Array(R).fill(0)
  const seqs = workload.seqlens.map((L, i) => ({ L, i })).sort((a, b) => b.L - a.L)
  const seqCost = new Array(workload.nSequences)
  for (const { L, i } of seqs) {
    const ranges = workload.rangesPerSeq.get(i)
    const r = costModel.getCost({
      seqlensQ: [L], seqlensK: [L],
      rangesForTokens: ranges, qStart: 0, qEnd: L, kvStart: 0, kvEnd: L,
      paddedQ: L, paddedK: L,
      nHeads: workload.nHeads, nQueryGroups: workload.nQueryGroups, headDim: workload.headDim,
    })
    seqCost[i] = Math.max(r.flopsTimeMs, r.memTimeMs)
  }
  for (const { L, i } of seqs) {
    let best = 0
    for (let d = 1; d < R; d++) if (load[d] < load[best]) best = d
    load[best] += L
    perDeviceMs[best] += seqCost[i]
    perDeviceMem[best] += L * perToken
  }
  const totalMs = maxOf(perDeviceMs)
  return {
    name: 'dp',
    perDeviceMs,
    totalMs,
    perDeviceCompute: perDeviceMs,
    perDeviceComm: new Array(R).fill(0),
    perDeviceCommBytes: new Array(R).fill(0),
    perDeviceMem,
    maxCommBytes: 0,
    totalCommBytes: 0,
    computeImbalance: maxOf(perDeviceMs) / (sum(perDeviceMs) / R),
    memImbalance: maxOf(perDeviceMem) / (sum(perDeviceMem) / R),
  }
}
