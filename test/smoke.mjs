import { planBatch } from '../lib/dcp/planner.js'
import { makeContext } from '../lib/dcp/core.js'

// 论文 Fig.5 场景：1 条长序列 + 2 条短序列，2 设备
const ctx = makeContext({ nodes: 1, devicesPerNode: 2 })
const plan = planBatch({
  seqlens: [4096, 4096, 32768],
  nHeads: 8, nQueryGroups: 2, headDim: 128,
  blockSize: 1024, headBlockSize: 4,
  maskSpec: { type: 'causal' },
  context: ctx,
  nDivisions: 4,
})
console.log('=== plan summary ===')
console.log('blocks:', JSON.stringify(plan.blocks))
console.log('work blocks:', plan.blocks.nWorkBlocks)
console.log('placement stats:', JSON.stringify(plan.placement.stats, (k, v) => typeof v === 'number' ? Math.round(v) : v))
console.log('schedule divisions:', plan.schedule.nDivisions)
console.log('simulation:', JSON.stringify({
  dcpMs: plan.simulation.dcpMs?.toFixed?.(3),
  zigzagMs: plan.simulation.zigzagMs?.toFixed?.(3),
  dpMs: plan.simulation.dpMs?.toFixed?.(3),
  speedupVsZigzag: plan.simulation.speedupVsZigzag?.toFixed?.(3),
  speedupVsDp: plan.simulation.speedupVsDp?.toFixed?.(3),
}))
console.log('metrics:', JSON.stringify({ ...plan.metrics, totalCommBytes: Math.round(plan.metrics.totalCommBytes), zigzagTotalCommBytes: Math.round(plan.metrics.zigzagTotalCommBytes) }))
console.log('per-device dcp ms:', plan.simulation.perDeviceMs.map(x => x.toFixed(3)))
console.log('per-device zigzag ms:', plan.simulation.zigzagPerDeviceMs.map(x => x.toFixed(3)))
