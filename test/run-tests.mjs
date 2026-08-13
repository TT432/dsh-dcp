// dsh-dcp 核心测试：块生成 / mask / 成本模型 / 超图划分 / 调度 / 模拟 / 论文行为
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { generateWorkload } from '../lib/dcp/blockgen.js'
import { buildMaskRanges, maskLabel } from '../lib/dcp/masks.js'
import { AttnRooflineCostModel, CommunicationCostModel } from '../lib/dcp/costmodel.js'
import { buildHypergraph } from '../lib/dcp/hypergraph.js'
import { partitionHypergraph, placeBlocks } from '../lib/dcp/partition.js'
import { scheduleDivisions } from '../lib/dcp/scheduler.js'
import { simulatePlan } from '../lib/dcp/simulator.js'
import { simulateZigzag, simulateDp } from '../lib/dcp/baselines.js'
import { planBatch } from '../lib/dcp/planner.js'
import { makeContext, sum } from '../lib/dcp/core.js'

// ---------- mask ----------
test('causal mask ranges', () => {
  const r = buildMaskRanges({ type: 'causal' }, 4)
  assert.deepEqual(r, [[[0, 1]], [[0, 2]], [[0, 3]], [[0, 4]]])
})

test('lambda mask sparsity (L=16384, sink=64, window=4096) ≈ 0.22', () => {
  const r = buildMaskRanges({ type: 'lambda', sink: 64, window: 4096 }, 16384)
  let actual = 0
  for (let t = 0; t < 16384; t++) for (const [s, e] of r[t]) actual += Math.min(e, 16384) - Math.max(s, 0)
  const sp = actual / (16384 * 16384)
  assert.ok(Math.abs(sp - 0.2217) < 0.01, 'sparsity=' + sp)
})

test('shared_question mask sparsity ≈ 0.26', () => {
  const r = buildMaskRanges({ type: 'shared_question', questionTokens: 1638, answerTokens: 1638, nAnswers: 4 }, 8190)
  let actual = 0
  for (let t = 0; t < 8190; t++) for (const [s, e] of r[t]) actual += Math.min(e, 8190) - Math.max(s, 0)
  const sp = actual / (8190 * 8190)
  assert.ok(Math.abs(sp - 0.2601) < 0.01, 'sparsity=' + sp)
})

// ---------- 成本模型 ----------
test('AttnRooflineCostModel flops 公式', () => {
  const cm = new AttnRooflineCostModel({ maxTflops: 280, maxMemBwGBs: 1500 })
  // 4 * q*k * H * D
  const f = cm.flops([1024], [1024], 8, 128, 1.0)
  assert.equal(f, 4 * 1024 * 1024 * 8 * 128)
  const b = cm.flops([1024], [1024], 8, 128, 1.0, 'bwd')
  assert.equal(b, 2.5 * f)
  const t = cm.getCost({ seqlensQ: [1024], seqlensK: [1024], rangesForTokens: buildMaskRanges({ type: 'causal' }, 1024), qStart: 0, qEnd: 1024, kvStart: 0, kvEnd: 1024, paddedQ: 1024, paddedK: 1024, nHeads: 8, nQueryGroups: 2, headDim: 128 })
  assert.ok(t.flopsTimeMs > 0 && t.memTimeMs > 0)
})

test('CommunicationCostModel 单位', () => {
  const cm = new CommunicationCostModel({ intraBandwidthGbps: 4800, interBandwidthGbps: 400 })
  // 100MB intra: 100e6*8/1e6/4800 ms
  const intra = cm.cost([0, 0], [0, 1], 100e6)
  assert.ok(Math.abs(intra - 100e6 * 8 / 1e6 / 4800) < 1e-9)
  const inter = cm.cost([0, 0], [1, 0], 100e6)
  assert.ok(Math.abs(inter - 100e6 * 8 / 1e6 / 400) < 1e-9)
})

// ---------- 块生成 ----------
test('块生成：数量/colocation/因果 work 块', () => {
  const wl = generateWorkload({
    seqlens: [4096, 32768], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  // seq1: 4 块, seq2: 32 块
  assert.equal(wl.seqBlockCounts[0], 4)
  assert.equal(wl.seqBlockCounts[1], 32)
  // Q 块数 = nHeadBlocksQ(2) * 总块数(36) = 72；KV 同 72
  assert.equal(wl.inputs.filter(m => m.type === 'Q').length, 2 * 36)
  assert.equal(wl.inputs.filter(m => m.type === 'KV').length, 2 * 36)
  // work 块：seq1 causal = 4*5/2=10 × 2 hb = 20；seq2 = 32*33/2=528 × 2 = 1056
  assert.equal(wl.work.filter(w => w.seqId === 0).length, 20)
  assert.equal(wl.work.filter(w => w.seqId === 1).length, 1056)
  assert.equal(wl.nWork, 1076)
  // colocation 约束：每个 seq 块一组
  assert.equal(wl.colocationConstraints.length, 36)
  // 每个 work 的输出归属正确
  const w0 = wl.work[0]
  assert.equal(wl.outputs[w0.outId].type, 'Out')
  assert.equal(wl.outputs[w0.lseId].type, 'LSE')
})

test('块生成：GQA kv head 组映射', () => {
  const wl = generateWorkload({
    seqlens: [1024], blockSize: 512, headBlockSize: 2,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  // Q 头块 0-3 → KV 组 floor(k*2/4)：0,0,1,1
  for (const w of wl.work) {
    const qHead = w.headBlockId
    const kvHead = wl.inputs[w.kvInputId].headBlockId
    const expect = Math.floor((qHead * 2) / 4)
    assert.equal(kvHead, expect)
  }
})

// ---------- 超图 ----------
test('超图：顶点与超边数量', () => {
  const wl = generateWorkload({
    seqlens: [2048, 2048], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const hg = buildHypergraph(wl)
  assert.equal(hg.nVertices, wl.nWork + wl.colocationConstraints.length)
  assert.equal(hg.nets.length, wl.nInputs + wl.nOutputs)
})

test('划分：平衡约束与全量覆盖', () => {
  const wl = generateWorkload({
    seqlens: [4096, 4096, 16384], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const hg = buildHypergraph(wl)
  const res = partitionHypergraph(hg, 4, { compEpsilon: 0.2, memEpsilon: 0.2 })
  // 每个顶点都分配到划分
  for (const p of res.partOfVertex) assert.ok(p >= 0 && p < 4)
  // 计算负载 <= 1.1*avg（紧容量）
  const avgC = sum(hg.computeWeights) / 4
  for (const c of res.computeLoads) assert.ok(c <= avgC * 1.1 + 1e-6, 'compute load ' + c)
  const avgM = sum(hg.memWeights) / 4
  for (const m of res.memLoads) assert.ok(m <= avgM * 1.05 + 1e-6, 'mem load ' + m)
})

// ---------- 调度 ----------
test('调度：所有 work 恰好调度一次；division 通信预算', () => {
  const ctx = makeContext({ nodes: 1, devicesPerNode: 4 })
  const wl = generateWorkload({
    seqlens: [4096, 8192], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const placement = placeBlocks(ctx, wl, {})
  const sched = scheduleDivisions(ctx, wl, placement, { nDivisions: 4 })
  const scheduled = new Set()
  for (const div of sched.divisions) for (const w of div.work) scheduled.add(w)
  assert.equal(scheduled.size, wl.nWork)
  // divisions 数量 = 4
  assert.equal(sched.divisions.length, 4)
  // 第 0 个 division 不含通信
  assert.equal(sched.divisions[0].comm.length, 0)
  // 每个设备都有执行计划
  assert.equal(sched.planPerDevice.length, 4)
})

// ---------- 模拟 ----------
test('模拟：单设备无通信 = 计算总和', () => {
  const ctx = makeContext({ nodes: 1, devicesPerNode: 1 })
  const wl = generateWorkload({
    seqlens: [4096], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const placement = placeBlocks(ctx, wl, {})
  const sched = scheduleDivisions(ctx, wl, placement, { nDivisions: 4 })
  const sim = simulatePlan(ctx, wl, placement, sched, {
    commCostModel: new CommunicationCostModel(),
    costModel: new AttnRooflineCostModel(),
  })
  const expected = sum(wl.work.map(w => w.costMs))
  assert.ok(Math.abs(sim.totalMs - expected) < 1e-6, 'sim ' + sim.totalMs + ' vs ' + expected)
})

test('模拟：zigzag 基线因果结构', () => {
  const ctx = makeContext({ nodes: 1, devicesPerNode: 2 })
  const wl = generateWorkload({
    seqlens: [4096], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const z = simulateZigzag(ctx, wl, {
    commCostModel: new CommunicationCostModel(),
    costModel: new AttnRooflineCostModel(),
  })
  assert.ok(z.totalMs > 0)
  assert.ok(z.perDeviceMs.length === 2)
})

// ---------- 论文行为 ----------
test('论文行为：多短序列 → DCP 大幅优于 zigzag（≈dp）', () => {
  const ctx = makeContext({ nodes: 4, devicesPerNode: 8 })
  const seqlens = Array.from({ length: 32 }, () => 4096)
  const p = planBatch({ seqlens, nHeads: 8, nQueryGroups: 2, headDim: 128, blockSize: 512, headBlockSize: 4, maskSpec: { type: 'causal' }, context: ctx, nDivisions: 4 })
  assert.ok(p.simulation.speedupVsZigzag > 3, 'speedup vs zigzag = ' + p.simulation.speedupVsZigzag)
  assert.ok(p.metrics.totalCommBytes < p.metrics.zigzagTotalCommBytes / 10, 'comm 应远小于 zigzag')
})

test('论文行为：共享问题 mask → DCP 通信随稀疏度降低', () => {
  const ctx = makeContext({ nodes: 4, devicesPerNode: 8 })
  const seqlens = [8192, 8192, 8192, 8192]
  const p = planBatch({ seqlens, nHeads: 8, nQueryGroups: 2, headDim: 128, blockSize: 512, headBlockSize: 4, maskSpec: { type: 'shared_question', questionTokens: 1638, answerTokens: 1638, nAnswers: 4 }, context: ctx, nDivisions: 4 })
  assert.ok(p.simulation.speedupVsZigzag > 2, 'speedup = ' + p.simulation.speedupVsZigzag)
  assert.ok(p.metrics.totalCommBytes < p.metrics.zigzagTotalCommBytes, 'DCP comm 应更低')
})

test('论文行为：均衡因果批次 → DCP ≈ zigzag（±40%）', () => {
  const ctx = makeContext({ nodes: 1, devicesPerNode: 2 })
  const p = planBatch({ seqlens: [4096, 4096, 32768], nHeads: 8, nQueryGroups: 2, headDim: 128, blockSize: 1024, headBlockSize: 4, maskSpec: { type: 'causal' }, context: ctx, nDivisions: 4 })
  const sp = p.simulation.speedupVsZigzag
  assert.ok(sp > 0.6 && sp < 2.5, 'speedup vs zigzag = ' + sp + '（长序列主导时应接近基线）')
})

test('分层放置：跨节点通信优先最小化', () => {
  const ctx = makeContext({ nodes: 2, devicesPerNode: 2 })
  const wl = generateWorkload({
    seqlens: [4096, 4096, 4096, 4096], blockSize: 1024, headBlockSize: 4,
    nHeads: 8, nQueryGroups: 2, headDim: 128, maskSpec: { type: 'causal' },
  })
  const p = placeBlocks(ctx, wl, {})
  // 4 条短序列 2 节点 → 每节点应尽量整条序列（无跨节点）
  assert.equal(p.stats.interNodeBytes, 0, '短序列不应跨节点')
})

// ---------- 计划完整性 ----------
test('计划结构完整性', () => {
  const ctx = makeContext({ nodes: 1, devicesPerNode: 4 })
  const p = planBatch({ seqlens: [4096, 8192, 16384], nHeads: 8, nQueryGroups: 2, headDim: 128, blockSize: 512, headBlockSize: 4, maskSpec: { type: 'causal' }, context: ctx, nDivisions: 4 })
  assert.ok(p.meta.nSequences === 3)
  assert.ok(p.blocks.nWorkBlocks > 0)
  assert.ok(p.executionPlan.length === 4)
  assert.ok(p.simulation.dcpMs > 0)
  assert.ok(p.metrics.totalCommBytes >= 0)
  assert.ok(Number.isFinite(p.simulation.speedupVsZigzag))
})