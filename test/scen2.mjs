import { planBatch } from '../lib/dcp/planner.js'
import { makeContext } from '../lib/dcp/core.js'

// 长尾混合批次（类似 LongAlign/LongDataCollections 的序列长度分布）
const ctx32 = makeContext({ nodes: 4, devicesPerNode: 8 })
const seqlens = [32768, 32768, 16384, 16384, 8192, 8192, 8192, 4096, 4096, 4096, 4096, 2048, 2048, 2048, 2048, 1024]
const p = planBatch({
  seqlens, nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 512, headBlockSize: 4, maskSpec: { type: 'causal' },
  context: ctx32, nDivisions: 4,
})
console.log('=== 长尾混合批次 (causal, 32 devices) ===')
console.log('tokens:', p.meta.nTokensTotal, 'work blocks:', p.blocks.nWorkBlocks)
console.log('dcp:', p.simulation.dcpMs.toFixed(3), 'ms; zigzag:', p.simulation.zigzagMs.toFixed(3), 'ms; dp:', p.simulation.dpMs.toFixed(3), 'ms')
console.log('speedup vs zigzag:', p.simulation.speedupVsZigzag.toFixed(2) + 'x')
console.log('comm: dcp', (p.metrics.totalCommBytes/1e6).toFixed(0), 'MB vs zigzag', (p.metrics.zigzagTotalCommBytes/1e6).toFixed(0), 'MB')
console.log('dcp comp imbalance:', p.metrics.dcpComputeImbalance.toFixed(3), 'mem imbalance:', p.metrics.dcpMemImbalance.toFixed(3))
console.log('per-device dcp ms (max/min):', Math.max(...p.simulation.perDeviceMs).toFixed(3), '/', Math.min(...p.simulation.perDeviceMs).toFixed(3))

// lambda mask 同批次
const p2 = planBatch({
  seqlens, nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 512, headBlockSize: 4,
  maskSpec: { type: 'lambda', sink: 64, window: 4096 },
  context: ctx32, nDivisions: 4,
})
console.log('=== 同批次 lambda mask ===')
console.log('dcp:', p2.simulation.dcpMs.toFixed(3), 'ms; zigzag:', p2.simulation.zigzagMs.toFixed(3), 'ms')
console.log('speedup vs zigzag:', p2.simulation.speedupVsZigzag.toFixed(2) + 'x')
console.log('comm: dcp', (p2.metrics.totalCommBytes/1e6).toFixed(0), 'MB vs zigzag', (p2.metrics.zigzagTotalCommBytes/1e6).toFixed(0), 'MB')
