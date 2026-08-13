// 超图划分与分层放置。
// 官方使用 KaHyPar/PaToH 求解平衡超图划分（NP-hard，启发式）。
// 本实现提供纯 JS 的自带求解器：greedy 初始化 + Fiduccia-Mattheyses (FM) 局部搜索精化，
// 对 compute/memory 双约束做平衡（论文公式 §4.2：w(Pi) <= [1+eps, 1] * w(N)/R）。
// 分层：先按节点划分（最小化跨节点通信），再在节点内按设备划分。
import { sum, max } from './core.js'
import { buildHypergraph } from './hypergraph.js'

// ---------- 超图划分 ----------
// hg: buildHypergraph 的输出；k: 划分数
// 返回 { partOfVertex, cutCost, computeLoads, memLoads }
export function partitionHypergraph(hg, k, {
  compEpsilon = 0.2,
  memEpsilon = 0.2,
  maxPasses = 12,
  seed = 42,
} = {}) {
  const n = hg.nVertices
  if (process.env.DCP_DEBUG_PART) {
    console.log('[hg] n=', n, 'nGroups=', hg.nGroups, 'sumComp=', sum(hg.computeWeights), 'sumMem=', sum(hg.memWeights), 'nWorkField=', hg.nWork)
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(hg.computeWeights[i]) || !Number.isFinite(hg.memWeights[i])) {
        console.log('[hg] NON-FINITE at vertex', i, 'compW=', hg.computeWeights[i], 'memW=', hg.memWeights[i])
      }
    }
  }
  // work 顶点 = 计算权重 > 0 的顶点（子超图可能没有 nWork 字段）
  let nWork = 0
  for (let i = 0; i < n; i++) if (hg.computeWeights[i] > 0) nWork++
  if (k <= 1) {
    return {
      partOfVertex: new Array(n).fill(0),
      cutCost: 0,
      computeLoads: [sum(hg.computeWeights)],
      memLoads: [sum(hg.memWeights)],
    }
  }
  const compW = hg.computeWeights
  const memW = hg.memWeights
  const nets = hg.nets
  const vertexNets = hg.vertexNets

  const avgCompute = sum(compW) / k
  const avgMem = sum(memW) / k
  // 紧容量：论文“内存尽量完美平衡”，intra-node 计算 epsilon 用 0.1。
  // FM/修复/共置统一使用紧容量，避免松上限（默认 0.2）把平衡破坏后再修复的拉锯。
  const tightCompEpsilon = Math.min(compEpsilon, 0.1)
  const tightMemEpsilon = Math.min(memEpsilon, 0.05)
  const compCap = avgCompute * (1 + tightCompEpsilon) + 1e-12
  const memCap = avgMem * (1 + tightMemEpsilon) + 1e-12

  // 净切割成本：sum_e w_e * (lambda_e - 1)
  function netLambda(e, partOfVertex) {
    const seen = new Set()
    for (const v of nets[e].pins) seen.add(partOfVertex[v])
    return seen.size
  }
  function cutCost(partOfVertex) {
    let cost = 0
    for (let e = 0; e < nets.length; e++) {
      const lam = netLambda(e, partOfVertex)
      if (lam > 1) cost += nets[e].weight * (lam - 1)
    }
    return cost
  }

  // ---------- work 分组（按 (seq, headBlock, qBlock)，blockgen 生成顺序保证同组连续）----------
  // 同一输出块的所有贡献 work 必须同机（消除跨设备 reduction/输出通信）
  const workGroups = []
  {
    let curKey = null
    let curGroup = null
    for (let v = 0; v < nWork; v++) {
      const dataPinsSet = new Set()
      for (const e of vertexNets[v]) {
        for (const u of nets[e].pins) if (u >= nWork) dataPinsSet.add(u)
      }
      const key = [...dataPinsSet].sort((a, b) => a - b).join(',')
      if (key !== curKey) {
        curKey = key
        curGroup = { work: [], dataPins: [...dataPinsSet], compSum: 0 }
        workGroups.push(curGroup)
      }
      curGroup.work.push(v)
      curGroup.compSum += compW[v]
    }
  }

  // ---------- greedy 初始化 ----------
  let partOfVertex = new Array(n).fill(0)
  const memLoad = new Array(k).fill(0)
  const compLoad = new Array(k).fill(0)
  const partCounts = new Array(k).fill(0)
  // 顶点按 mem 降序（其次 compute 降序）处理，保证最大块先落位
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) =>
    memW[b] - memW[a] || compW[b] - compW[a] || a - b)

  // 通信感知：顶点 v 放入划分 p 时，其超边中数据（colocation 组）已在 p 内的净重占比。
  // 用于 greedy 初始化的 tie-break，让 work 顶点就近其数据，降低后续通信。
  const localWeightRatio = (v, p) => {
    let total = 0
    let local = 0
    for (const e of vertexNets[v]) {
      const w = nets[e].weight
      total += w
      // 超边内的数据顶点（在超图里排在 work 之后）
      for (const u of nets[e].pins) {
        if (u >= nWork && partOfVertex[u] === p) { local += w; break }
      }
    }
    return total > 0 ? local / total : 0
  }

  const cutOfVertex = (v, p) => {
    // 把 v 放入 p 带来的净切割增量
    let delta = 0
    for (const e of vertexNets[v]) {
      const pins = nets[e].pins
      const w = nets[e].weight
      let inP = false
      let others = 0
      for (const u of pins) {
        if (u === v) continue
        const pu = partOfVertex[u]
        if (pu === p) { inP = true; break }
        others++
      }
      // 简化：计算 (lambda-1) 的精确增量
      const lamBefore = new Set(pins.map(u => partOfVertex[u])).size
      const lamAfter = new Set(pins.map(u => (u === v ? p : partOfVertex[u]))).size
      delta += w * (lamAfter - lamBefore)
    }
    return delta
  }

  // 两阶段 greedy 初始化（PaToH 风格）：
  //  阶段 A：数据顶点（colocation 组）按内存平衡先落位；
  //  阶段 B：work 顶点按“本地化数据权重”优先（通信优先），平衡（计算上限）作为可行性约束。
  // 阶段 A（序列感知）：数据顶点按序列分组。
  // 论文 Fig.5(c) 的关键行为：能整条放下的序列整条放一个设备（DP），
  // 放不下的长序列再按组分散（CP）。组内（Q/KV/Out/LSE）恒同机（colocation）。
  const groupSeqId = (g) => hg.dataVertexGroup[g].seqId
  // 每个序列的组列表与总内存
  const seqGroups = new Map() // seqId -> [groupIdx...]
  for (let g = 0; g < hg.nGroups; g++) {
    const s = groupSeqId(g)
    if (!seqGroups.has(s)) seqGroups.set(s, [])
    seqGroups.get(s).push(g)
  }
  const seqMemOf = (s) => seqGroups.get(s).reduce((a, g) => a + hg.memWeights[nWork + g], 0)
  if (process.env.DCP_DEBUG_PART) {
    console.log('[phaseA] k=', k, 'memCap=', memCap, 'seqCount=', seqGroups.size, 'seqSizes=', [...seqGroups.keys()].map(s => s + ':' + seqMemOf(s).toFixed(1)).join(' '))
  }
  const seqOrder = [...seqGroups.keys()].sort((a, b) => seqMemOf(b) - seqMemOf(a))
  const groupAssigned = new Array(hg.nGroups).fill(false)
  for (const s of seqOrder) {
    const groups = seqGroups.get(s)
    const seqMem = seqMemOf(s)
    // 尝试整序列放入一个设备
    let bestP = -1, bestLoad = Infinity
    for (let p = 0; p < k; p++) {
      let seqLoad = seqMem
      // 当前组在 p 上的负载 + 序列总内存
      const curMem = memLoad[p]
      if (curMem + seqLoad <= memCap * 1.0001) {
        if (curMem < bestLoad) { bestLoad = curMem; bestP = p }
      }
    }
    if (process.env.DCP_DEBUG_PART) console.log('[phaseA] seq', s, 'seqMem=', seqMem.toFixed(2), 'bestP=', bestP, 'memLoad0=', memLoad[0].toFixed(0))
    if (bestP >= 0) {
      for (const g of groups) {
        const v = nWork + g
        partOfVertex[v] = bestP
        memLoad[bestP] += memW[v]
        compLoad[bestP] += compW[v]
        partCounts[bestP]++
        groupAssigned[g] = true
      }
      continue
    }
    // 放不下：把序列切成连续段动态填充——反复把最大连续块放入当前最轻且放得下的设备。
    // 连续段保持相邻块的局部性（sliding-window/lambda 等局部 mask 的关键）；
    // 每次填充目标节点剩余容量，尽量保持内存平衡（等价论文的参考 packing）。
    const groupMem = memW[nWork + groups[0]]
    let start = 0
    while (start < groups.length) {
      let best = 0
      for (let p = 1; p < k; p++) if (memLoad[p] < memLoad[best]) best = p
      const freeCap = Math.max(0, memCap * 1.0001 - memLoad[best])
      const segSize = Math.max(1, Math.min(groups.length - start, Math.floor(freeCap / groupMem)))
      const seg = groups.slice(start, start + segSize)
      start += segSize
      for (const g of seg) {
        const v = nWork + g
        partOfVertex[v] = best
        memLoad[best] += memW[v]
        compLoad[best] += compW[v]
        partCounts[best]++
        groupAssigned[g] = true
      }
    }
  }

  // 阶段 B（组级）：把每个 work 组整体放到“本地化数据权重”最大的可行划分。
  // 组优先就近其 Q/Out 数据（输出同机，消除输出通信），计算平衡作为可行性约束。
  for (const g of workGroups) {
    // 该组涉及的数据顶点 → 其所在划分（若已分配）的字节权重
    const localW = new Array(k).fill(0)
    for (const u of g.dataPins) {
      localW[partOfVertex[u]] += memW[u]
    }
    let best = -1
    let bestLocal = -Infinity
    let bestRatio = Infinity
    for (let p = 0; p < k; p++) {
      // 组的计算量
      let groupComp = 0
      for (const v of g.work) groupComp += compW[v]
      const memFeasible = true // 组只有 work 顶点，不占内存
      const compFeasible = compLoad[p] + groupComp <= compCap * 1.0001
      if (!compFeasible) continue
      if (localW[p] > bestLocal + 1e-9 ||
          (Math.abs(localW[p] - bestLocal) <= 1e-9 && (compLoad[p] + groupComp) / (avgCompute + 1e-12) < bestRatio - 1e-9)) {
        bestLocal = localW[p]
        bestRatio = (compLoad[p] + groupComp) / (avgCompute + 1e-12)
        best = p
      }
    }
    if (best < 0) {
      // 全部超容量（理论上不会发生：总容量 = 1.1 * 总计算）
      let minL = Infinity
      let groupComp = 0
      for (const v of g.work) groupComp += compW[v]
      for (let p = 0; p < k; p++) {
        const l = compLoad[p] + groupComp
        if (l < minL) { minL = l; best = p }
      }
    }
    for (const v of g.work) {
      partOfVertex[v] = best
      memLoad[best] += memW[v]
      compLoad[best] += compW[v]
      partCounts[best]++
    }
  }

  // ---------- FM 精化 ----------
  const netCounts = nets.map(net => {
    const cnt = new Array(k).fill(0)
    for (const v of net.pins) cnt[partOfVertex[v]]++
    return cnt
  })

  function gainOf(v, p) {
    let gain = 0
    for (const e of vertexNets[v]) {
      const cnt = netCounts[e]
      const cur = partOfVertex[v]
      const w = nets[e].weight
      const cntP = cnt[cur]
      const cntQ = cnt[p]
      if (cntP === 1 && cntQ > 0) gain += w
      else if (cntP > 1 && cntQ === 0) gain -= w
      // 其余情况不变
    }
    return gain
  }

  let bestCut = cutCost(partOfVertex)

  for (let pass = 0; pass < maxPasses; pass++) {
    const locked = new Array(n).fill(false)
    const savedState = {
      parts: partOfVertex.slice(),
      memLoad: memLoad.slice(),
      compLoad: compLoad.slice(),
      cut: bestCut,
    }
    let cumGain = 0
    let bestCumGain = 0
    let bestStateAt = savedState
    let moved = 0
    let improvement = false

    for (let step = 0; step < n - nWork; step++) {
      // 找最佳可行移动（只移动数据顶点——work 由 Phase B 组级放置，FM 逐顶点移动会碎片化输出）
      let bestV = -1, bestP = -1, bestG = -Infinity
      for (let v = nWork; v < n; v++) {
        if (locked[v]) continue
        const cur = partOfVertex[v]
        for (let p = 0; p < k; p++) {
          if (p === cur) continue
          if (memLoad[p] + memW[v] > memCap + 1e-9) continue
          if (compLoad[p] + compW[v] > compCap + 1e-9) continue
          const g = gainOf(v, p)
          if (g > bestG || (g === bestG && (bestV < 0 || memW[v] > memW[bestV]))) {
            bestG = g; bestV = v; bestP = p
          }
        }
      }
      if (bestV < 0) break
      // 执行移动
      const cur = partOfVertex[bestV]
      partOfVertex[bestV] = bestP
      memLoad[cur] -= memW[bestV]; memLoad[bestP] += memW[bestV]
      compLoad[cur] -= compW[bestV]; compLoad[bestP] += compW[bestV]
      for (const e of vertexNets[bestV]) {
        netCounts[e][cur]--
        netCounts[e][bestP]++
      }
      locked[bestV] = true
      cumGain += bestG
      moved++
      if (cumGain > bestCumGain + 1e-12) {
        bestCumGain = cumGain
        bestStateAt = {
          parts: partOfVertex.slice(),
          memLoad: memLoad.slice(),
          compLoad: compLoad.slice(),
          cut: savedState.cut - cumGain,
        }
        improvement = true
      }
      if (moved >= n) break
    }
    // 回滚到最佳前缀
    partOfVertex = bestStateAt.parts
    // 重建负载与 net counts
    for (let p = 0; p < k; p++) { memLoad[p] = bestStateAt.memLoad[p]; compLoad[p] = bestStateAt.compLoad[p] }
    for (let e = 0; e < nets.length; e++) {
      const cnt = new Array(k).fill(0)
      for (const v of nets[e].pins) cnt[partOfVertex[v]]++
      netCounts[e] = cnt
    }
    if (!improvement || bestCumGain <= 1e-12) break
    bestCut = bestStateAt.cut
  }

  // 顶点 v 从 p 移到 q 的割成本增量（闭包，使用当前 partOfVertex / nets / vertexNets）
  function computeCutDelta(v, p, q) {
    let delta = 0
    for (const e of vertexNets[v]) {
      const pins = nets[e].pins
      const w = nets[e].weight
      let lamBefore = 0, lamAfter = 0
      const seenBefore = new Set(), seenAfter = new Set()
      for (const u of pins) {
        seenBefore.add(partOfVertex[u])
        seenAfter.add(u === v ? q : partOfVertex[u])
      }
      delta += w * (seenAfter.size - seenBefore.size)
    }
    return delta
  }

  // [DEBUG] FM 后状态
  if (process.env.DCP_DEBUG_PART) {
    console.log('[fm] cut:', bestCut, 'comp:', compLoad.map(x=>x.toFixed(3)), 'mem:', memLoad.map(x=>x.toFixed(0)))
  }
  // ---------- work 就近数据共置精化（组级）----------
  // 每个 work 组整体移到“其数据所在划分”权重最大的可行划分
  {
    const iterLimit = 6
    for (let it = 0; it < iterLimit; it++) {
      let movedAny = false
      const order = workGroups
        .map((g, i) => [i, g.compSum])
        .sort((a, b) => b[1] - a[1])
      for (const [gi] of order) {
        const g = workGroups[gi]
        const cur = partOfVertex[g.work[0]]
        // 每个划分的本地化数据权重（该组的数据顶点）
        const localW = new Array(k).fill(0)
        for (const u of g.dataPins) localW[partOfVertex[u]] += memW[u]
        let bestP = cur
        let bestW = localW[cur]
        for (let p = 0; p < k; p++) {
          if (p === cur) continue
          if (compLoad[p] + g.compSum > compCap + 1e-9) continue
          if (localW[p] > bestW + 1e-9) { bestP = p; bestW = localW[p] }
        }
        if (bestP !== cur) {
          for (const v of g.work) {
            partOfVertex[v] = bestP
            compLoad[cur] -= compW[v]; compLoad[bestP] += compW[v]
            memLoad[cur] -= memW[v]; memLoad[bestP] += memW[v]
            for (const e of vertexNets[v]) {
              netCounts[e][cur]--
              netCounts[e][bestP]++
            }
          }
          movedAny = true
        }
      }
      if (!movedAny) break
    }
    bestCut = cutCost(partOfVertex)
    if (process.env.DCP_DEBUG_PART) console.log('[colocate] cut:', bestCut, 'comp:', compLoad.map(x=>x.toFixed(3)), 'mem:', memLoad.map(x=>x.toFixed(0)))
  }


  // ---------- 平衡修复 ----------
  // KaHyPar/PaToH 是多层求解器，平衡质量远好于单层 FM；这里追加一个修复 pass：
  // 反复把最重划分中“割增量最小”的顶点移到最轻划分，直至两个维度都进入 (1+eps) 容差。
  {
    // 修复目标：论文的平衡约束是上限 epsilon（intra-node 论文用 0.1）；
    // 修复只把超出该容差的失衡拉回，不追求超过 epsilon 的过度平衡（过度平衡会增加通信）。
    const eps = Math.min(tightCompEpsilon, tightMemEpsilon)
    const iterLimit = 500
    for (let it = 0; it < iterLimit; it++) {
      const avgC = sum(compLoad) / k
      const avgM = sum(memLoad) / k
      const rC = max(compLoad) / (avgC || 1)
      const rM = max(memLoad) / (avgM || 1)
      if (Math.max(rC, rM) <= 1 + eps) break
      const useMem = rM >= rC
      const loadOf = useMem ? (i) => memLoad[i] : (i) => compLoad[i]
      const wOf = useMem ? (v) => memW[v] : (v) => compW[v]
      const otherCap = useMem ? compCap : memCap
      const otherLoad = useMem ? compLoad : memLoad
      const otherW = useMem ? compW : memW
      let heavy = 0, light = 0
      for (let p = 1; p < k; p++) {
        if (loadOf(p) > loadOf(heavy)) heavy = p
        if (loadOf(p) < loadOf(light)) light = p
      }
      if (heavy === light) break
      // 在 heavy 上找移到 light 后“割增量最小”的单元：
      // 内存维度按数据顶点；计算维度按整个 work 组（保持输出同机）
      let best = -1, bestDelta = Infinity
      if (useMem) {
        // 优先移动内存权重最大的顶点（一次移动最大影响；割代价次要）
        let bestW2 = 0
        for (let v = nWork; v < n; v++) {
          if (partOfVertex[v] !== heavy) continue
          if (memLoad[light] + memW[v] > memCap + 1e-9) continue
          if (memLoad[heavy] - memW[v] < -1e-9) continue
          if (memW[v] <= bestW2) continue
          const delta = computeCutDelta(v, heavy, light)
          if (delta < 2 * memW[v]) { bestW2 = memW[v]; best = v } // 割增量小于 2 倍权重才接受
        }
        if (best < 0) {
          // 兜底：允许更高割代价
          for (let v = nWork; v < n; v++) {
            if (partOfVertex[v] !== heavy) continue
            if (memLoad[light] + memW[v] > memCap + 1e-9) continue
            if (memLoad[heavy] - memW[v] < -1e-9) continue
            if (memW[v] > bestW2) { bestW2 = memW[v]; best = v }
          }
        }
      } else {
        // 优先移动计算量最大的组（一次移动最大影响）
        let bestW2 = 0
        for (let gi = 0; gi < workGroups.length; gi++) {
          const g = workGroups[gi]
          if (partOfVertex[g.work[0]] !== heavy) continue
          if (compLoad[light] + g.compSum > compCap + 1e-9) continue
          if (compLoad[heavy] - g.compSum < -1e-9) continue
          if (g.compSum <= bestW2) continue
          // 组割增量 = 各顶点割增量之和
          let delta = 0
          for (const v of g.work) delta += computeCutDelta(v, heavy, light)
          if (delta < 2 * g.compSum) { bestW2 = g.compSum; best = gi }
        }
        if (best < 0) {
          for (let gi = 0; gi < workGroups.length; gi++) {
            const g = workGroups[gi]
            if (partOfVertex[g.work[0]] !== heavy) continue
            if (compLoad[light] + g.compSum > compCap + 1e-9) continue
            if (compLoad[heavy] - g.compSum < -1e-9) continue
            if (g.compSum > bestW2) { bestW2 = g.compSum; best = gi }
          }
        }
      }
      if (best < 0) { if (process.env.DCP_DEBUG_PART) console.log('[repair] no movable unit') ; break }
      // 执行移动
      if (useMem) {
        const v = best
        const cur = partOfVertex[v]
        partOfVertex[v] = light
        memLoad[cur] -= memW[v]; memLoad[light] += memW[v]
        compLoad[cur] -= compW[v]; compLoad[light] += compW[v]
        for (const e of vertexNets[v]) {
          netCounts[e][cur]--
          netCounts[e][light]++
        }
      } else {
        const g = workGroups[best]
        for (const v of g.work) {
          const cur = partOfVertex[v]
          partOfVertex[v] = light
          memLoad[cur] -= memW[v]; memLoad[light] += memW[v]
          compLoad[cur] -= compW[v]; compLoad[light] += compW[v]
          for (const e of vertexNets[v]) {
            netCounts[e][cur]--
            netCounts[e][light]++
          }
        }
      }
      if (process.env.DCP_DEBUG_PART && it % 20 === 0) console.log('[repair] it', it, 'comp:', compLoad.map(x=>x.toFixed(3)), 'mem:', memLoad.map(x=>x.toFixed(0)))
    }
    bestCut = cutCost(partOfVertex)
  }

  return {
    partOfVertex,
    cutCost: bestCut,
    computeLoads: compLoad,
    memLoads: memLoad,
  }
}


// ---------- 分层放置 ----------
// 返回 { workToDevice, inputToDevice, outputToDevice, stats }
// stats: { interNodeBytes, intraNodeBytes, totalBytes, nodeBalance, deviceBalance }
export function placeBlocks(context, workload, opts = {}) {
  const { nNodes, nDevicesPerNode } = context
  const nTotal = nNodes * nDevicesPerNode
  const hg = buildHypergraph(workload)
  const nWork = workload.nWork

  const workToDevice = new Array(nWork)
  const inputToDevice = new Array(workload.nInputs)
  const outputToDevice = new Array(workload.nOutputs)

  if (nTotal === 1) {
    for (let w = 0; w < nWork; w++) workToDevice[w] = [0, 0]
    for (let i = 0; i < workload.nInputs; i++) inputToDevice[i] = [0, 0]
    for (let o = 0; o < workload.nOutputs; o++) outputToDevice[o] = [0, 0]
    return {
      workToDevice, inputToDevice, outputToDevice,
      stats: {
        interNodeBytes: 0, intraNodeBytes: 0, totalBytes: 0,
        nodeCompute: [sum(hg.computeWeights)], nodeMem: [sum(hg.memWeights)],
        deviceCompute: [sum(hg.computeWeights)], deviceMem: [sum(hg.memWeights)],
      },
    }
  }

  const interEpsilon = opts.interNodeCompEpsilon ?? (opts.compEpsilon ?? 0.2) * (opts.interNodeCompEpsilonFactor ?? 5)
  const memEpsilon = opts.memEpsilon ?? 0.2
  const compEpsilon = opts.compEpsilon ?? 0.2

  // ---- Level 1: 节点间 ----
  let nodeOfVertex
  if (nNodes === 1) {
    nodeOfVertex = new Array(hg.nVertices).fill(0)
  } else {
    const p1 = partitionHypergraph(hg, nNodes, { compEpsilon: interEpsilon, memEpsilon })
    nodeOfVertex = p1.partOfVertex
  }

  // ---- Level 2: 节点内设备 ----
  const devOfVertex = new Array(hg.nVertices)
  const nodeCompute = new Array(nNodes).fill(0)
  const nodeMem = new Array(nNodes).fill(0)
  for (let v = 0; v < hg.nVertices; v++) {
    nodeCompute[nodeOfVertex[v]] += hg.computeWeights[v]
    nodeMem[nodeOfVertex[v]] += hg.memWeights[v]
  }
  const deviceCompute = new Array(nTotal).fill(0)
  const deviceMem = new Array(nTotal).fill(0)

  for (let node = 0; node < nNodes; node++) {
    if (nDevicesPerNode === 1) {
      for (let v = 0; v < hg.nVertices; v++) {
        if (nodeOfVertex[v] === node) {
          devOfVertex[v] = 0
          deviceCompute[node] += hg.computeWeights[v]
          deviceMem[node] += hg.memWeights[v]
        }
      }
      continue
    }
    // 构建节点内子图：只含本节点顶点；超边取本节点内的 pins
    const localVertices = []
    for (let v = 0; v < hg.nVertices; v++) if (nodeOfVertex[v] === node) localVertices.push(v)
    const subIdx = new Map(localVertices.map((v, i) => [v, i]))
    const subNets = []
    for (const net of hg.nets) {
      const pins = net.pins.filter(v => subIdx.has(v)).map(v => subIdx.get(v))
      if (pins.length >= 2) subNets.push({ pins, weight: net.weight })
    }
    // 子超图的数据顶点（orig >= nWork）在子索引中连续排在工作顶点之后
    const subDataGroups = []
    for (const v of localVertices) {
      if (v >= hg.nWork) subDataGroups.push(hg.dataVertexGroup[v - hg.nWork])
    }
    const subHg = {
      nVertices: localVertices.length,
      nGroups: subDataGroups.length,
      dataVertexGroup: subDataGroups,
      computeWeights: localVertices.map(v => hg.computeWeights[v]),
      memWeights: localVertices.map(v => hg.memWeights[v]),
      nets: subNets,
      vertexNets: (() => {
        const vn = new Array(localVertices.length)
        for (let i = 0; i < localVertices.length; i++) vn[i] = []
        for (let e = 0; e < subNets.length; e++) for (const p of subNets[e].pins) vn[p].push(e)
        return vn
      })(),
    }
    const p2 = partitionHypergraph(subHg, nDevicesPerNode, { compEpsilon, memEpsilon, seed: 42 + node })
    if (process.env.DCP_DEBUG_PART) {
      console.log('[level2 node', node, '] k=', nDevicesPerNode, 'nVertices=', subHg.nVertices, 'comp:', p2.computeLoads.map(x=>x.toFixed(3)).slice(0,8), 'mem:', p2.memLoads.map(x=>x.toFixed(0)).slice(0,8))
    }
    for (let i = 0; i < localVertices.length; i++) {
      const v = localVertices[i]
      devOfVertex[v] = p2.partOfVertex[i]
      deviceCompute[node * nDevicesPerNode + p2.partOfVertex[i]] += subHg.computeWeights[i]
      deviceMem[node * nDevicesPerNode + p2.partOfVertex[i]] += subHg.memWeights[i]
    }
  }

  // ---- 映射到 block ----
  for (let w = 0; w < nWork; w++) {
    const node = nodeOfVertex[w]
    workToDevice[w] = [node, devOfVertex[w]]
  }
  for (let i = 0; i < workload.nInputs; i++) {
    const v = hg.dataVertexByInput[i]
    const node = nodeOfVertex[v]
    inputToDevice[i] = [node, devOfVertex[v]]
  }
  for (let o = 0; o < workload.nOutputs; o++) {
    const v = hg.dataVertexByOutput[o]
    const node = nodeOfVertex[v]
    outputToDevice[o] = [node, devOfVertex[v]]
  }

  // ---- 通信量统计 ----
  const nodesOfNet = hg.nets.map(net => {
    const s = new Set()
    for (const v of net.pins) s.add(nodeOfVertex[v])
    return s
  })
  const devsOfNet = hg.nets.map(net => {
    const s = new Set()
    for (const v of net.pins) s.add(devOfVertex[v] + nodeOfVertex[v] * nDevicesPerNode)
    return s
  })
  let interNodeBytes = 0
  let totalBytes = 0
  for (let e = 0; e < hg.nets.length; e++) {
    const ln = nodesOfNet[e].size
    const ld = devsOfNet[e].size
    if (ln > 1) interNodeBytes += hg.nets[e].weight * (ln - 1)
    if (ld > 1) totalBytes += hg.nets[e].weight * (ld - 1)
  }
  const intraNodeBytes = Math.max(0, totalBytes - interNodeBytes)

  return {
    workToDevice, inputToDevice, outputToDevice,
    stats: {
      interNodeBytes, intraNodeBytes, totalBytes,
      nodeCompute, nodeMem,
      deviceCompute, deviceMem,
      nWork, nInputs: workload.nInputs, nOutputs: workload.nOutputs,
    },
  }
}

function buildHypergraphForPlacement(workload) {
  return buildHypergraph(workload)
}