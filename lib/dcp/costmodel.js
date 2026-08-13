// 成本模型：通信成本 + attention roofline 成本。
// 忠实移植官方 dcp/core/cost_model.py：
//   - CommunicationCostModel: 按 (src, dst) 与带宽计费，latency ms + size(bit)/1e6 / BW(Gbps) => ms
//   - AttnRooflineCostModel: FlashAttention 基准同款 flops/mem-access 公式（fwd 4N^2HD，bwd x2.5）
import { DType } from './core.js'

export class CommunicationCostModel {
  constructor({
    intraBandwidthGbps = 4800,
    interBandwidthGbps = 400,
    intraLatencyMs = 0,
    interLatencyMs = 0,
  } = {}) {
    this.intraBandwidthGbps = intraBandwidthGbps
    this.interBandwidthGbps = interBandwidthGbps
    this.intraLatencyMs = intraLatencyMs
    this.interLatencyMs = interLatencyMs
  }

  // src/dst: [nodeId, deviceId]; size: bytes; 返回 ms
  cost(src, dst, size) {
    if (src[0] === dst[0]) {
      if (src[1] === dst[1]) return 0
      return this.intraLatencyMs + (size * 8 / 1e6) / this.intraBandwidthGbps
    }
    return this.interLatencyMs + (size * 8 / 1e6) / this.interBandwidthGbps
  }

  // 传输耗时（不含 latency 的纯带宽部分），用于重叠分析
  bandwidthTime(src, dst, size) {
    if (src[0] === dst[0]) {
      if (src[1] === dst[1]) return 0
      return (size * 8 / 1e6) / this.intraBandwidthGbps
    }
    return (size * 8 / 1e6) / this.interBandwidthGbps
  }
}

export class AttnRooflineCostModel {
  constructor({ maxTflops = 280, maxMemBwGBs = 1500 } = {}) {
    this.maxTflops = maxTflops      // TFLOPS/s
    this.maxMemBwGBs = maxMemBwGBs  // GB/s
  }

  // 官方 _get_flops: 4 * sum(q*k) * nheads * headdim * sparsity; bwd x2.5
  flops(seqlensQ, seqlensK, nHeads, headDim, sparsity, mode = 'fwd') {
    let s = 0
    for (let i = 0; i < seqlensQ.length; i++) s += seqlensQ[i] * seqlensK[i]
    const f = 4 * s * nHeads * headDim * sparsity
    return mode === 'fwd' ? f : 2.5 * f
  }

  // 官方 _get_mem_access（FlashAttention 的访存模型）
  memAccess(seqlensQ, seqlensK, nQueryGroups, headDim, sparsity, mode = 'fwd') {
    const tokenSize = 2 * nQueryGroups * headDim
    let total = 0
    if (mode === 'fwd') {
      const mBlock = 64
      for (let i = 0; i < seqlensQ.length; i++) {
        total += Math.ceil(seqlensQ[i] / mBlock) * seqlensK[i] * tokenSize
      }
    } else {
      const nBlock = 128
      total += seqlensQ.reduce((a, b) => a + b, 0) * tokenSize // dO
      for (let i = 0; i < seqlensQ.length; i++) {
        total += 4 * seqlensQ[i] * Math.ceil(seqlensK[i] / nBlock) * tokenSize
      }
    }
    return total * sparsity
  }

  // 根据 per-token range 计算 mask sparsity。
  // rangesForTokens: 长度为 qStart..qEnd 的逐 token range 列表（含 padding 行用空数组）
  // 与官方一致：actual = 每个 q 行在 kv 区间内的有效元素数之和；total = paddedQ * paddedK
  maskSparsity(rangesForTokens, qStart, qEnd, kvStart, kvEnd, paddedQ, paddedK) {
    let actual = 0
    for (let t = qStart; t < qEnd; t++) {
      const ranges = rangesForTokens[t] || []
      for (const [s, e] of ranges) {
        const lo = Math.max(s, kvStart)
        const hi = Math.min(e, kvEnd)
        if (hi > lo) actual += hi - lo
      }
    }
    const total = paddedQ * paddedK
    return total > 0 ? actual / total : 0
  }

  // 返回 { flopsTimeMs, memTimeMs }；rangesForTokens 为全序列逐 token range。
  // qStart/qEnd 为 q 块区间，kvStart/kvEnd 为 kv 块区间，paddedQ/paddedK 用于分母。
  getCost({
    seqlensQ = [1],
    seqlensK = [1],
    rangesForTokens,
    qStart = 0,
    qEnd = seqlensQ[0],
    kvStart = 0,
    kvEnd = seqlensK[0],
    paddedQ = seqlensQ[0],
    paddedK = seqlensK[0],
    nHeads,
    nQueryGroups,
    headDim,
    isForward = true,
  }) {
    const sparsity = this.maskSparsity(rangesForTokens, qStart, qEnd, kvStart, kvEnd, paddedQ, paddedK)
    const mode = isForward ? 'fwd' : 'bwd'
    const flops = this.flops(seqlensQ, seqlensK, nHeads, headDim, sparsity, mode)
    const mem = this.memAccess(seqlensQ, seqlensK, nQueryGroups, headDim, sparsity, mode)
    const flopsTimeMs = flops / this.maxTflops * 1e-9   // flops / (TFLOPS*1e12) s * 1e3 = flops/maxTflops * 1e-9 ms
    const memTimeMs = mem / this.maxMemBwGBs * 1e-6     // mem / (GB*1e9) s * 1e3 = mem/maxMemBwGBs * 1e-6 ms
    return { flopsTimeMs, memTimeMs, sparsity }
  }
}
