import { planBatch } from '../lib/dcp/planner.js'
import { makeContext } from '../lib/dcp/core.js'

// 场景1：大量短序列（论文 scale=0.5 情形）——DCP 应显著快于 zigzag
const ctx32 = makeContext({ nodes: 4, devicesPerNode: 8 })
// 32 条 4K 序列（全局 128K tokens）
const seqlensShort = Array.from({ length: 32 }, () => 4096)
const p1 = planBatch({
  seqlens: seqlensShort, nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 512, headBlockSize: 4, maskSpec: { type: 'causal' },
  context: ctx32, nDivisions: 4,
})
console.log('=== 场景1: 32x4K 短序列, 32 设备, causal ===')
console.log('  dcp:', p1.simulation.dcpMs.toFixed(3), 'ms; zigzag:', p1.simulation.zigzagMs.toFixed(3), 'ms; dp:', p1.simulation.dpMs.toFixed(3), 'ms')
console.log('  speedup vs zigzag:', p1.simulation.speedupVsZigzag.toFixed(2) + 'x', '; vs dp:', p1.simulation.speedupVsDp.toFixed(2) + 'x')
console.log('  comm: dcp', (p1.metrics.totalCommBytes/1e6).toFixed(1), 'MB vs zigzag', (p1.metrics.zigzagTotalCommBytes/1e6).toFixed(1), 'MB')

// 场景2：共享问题 mask（稀疏）——DCP 应显著减少冗余通信
const p2 = planBatch({
  seqlens: [8192, 8192, 8192, 8192], nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 512, headBlockSize: 4,
  maskSpec: { type: 'shared_question', questionTokens: 1638, answerTokens: 1638, nAnswers: 4 },
  context: ctx32, nDivisions: 4,
})
console.log('=== 场景2: 4x8K 共享问题 mask, 32 设备 ===')
console.log('  avg work sparsity:', p2.blocks.avgWorkSparsity.toFixed(3))
console.log('  dcp:', p2.simulation.dcpMs.toFixed(3), 'ms; zigzag:', p2.simulation.zigzagMs.toFixed(3), 'ms')
console.log('  speedup vs zigzag:', p2.simulation.speedupVsZigzag.toFixed(2) + 'x')
console.log('  comm: dcp', (p2.metrics.totalCommBytes/1e6).toFixed(1), 'MB vs zigzag', (p2.metrics.zigzagTotalCommBytes/1e6).toFixed(1), 'MB')

// 场景3：lambda mask
const p3 = planBatch({
  seqlens: [16384, 16384, 16384, 16384, 16384], nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 512, headBlockSize: 4,
  maskSpec: { type: 'lambda', sink: 64, window: 4096 },
  context: ctx32, nDivisions: 4,
})
console.log('=== 场景3: 5x16K lambda mask(sink=64,window=4096), 32 设备 ===')
console.log('  avg work sparsity:', p3.blocks.avgWorkSparsity.toFixed(3))
console.log('  dcp:', p3.simulation.dcpMs.toFixed(3), 'ms; zigzag:', p3.simulation.zigzagMs.toFixed(3), 'ms')
console.log('  speedup vs zigzag:', p3.simulation.speedupVsZigzag.toFixed(2) + 'x')
console.log('  comm: dcp', (p3.metrics.totalCommBytes/1e6).toFixed(1), 'MB vs zigzag', (p3.metrics.zigzagTotalCommBytes/1e6).toFixed(1), 'MB')
