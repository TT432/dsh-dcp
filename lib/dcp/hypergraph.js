// 超图构建。
// 忠实移植官方 dcp/graph_partition/graph.py 的建模方式：
//   - 顶点 = 计算块(work) + 数据块（按 colocation 约束合并为一个顶点）
//   - 超边 = 每个数据块连接其消费/产出它的所有计算块，权重 = 数据块字节数
//   - 顶点权重：计算块 [cost, 0]，数据块 [0, memBytes]（论文公式 §4.2）
// 官方用整数缩放喂给 KaHyPar/PaToH；本实现直接用原始浮点值，语义等价。

export function buildHypergraph(workload) {
  const nWork = workload.nWork
  const groups = workload.colocationConstraints // [{inputIds, outputIds}]
  const nGroups = groups.length
  const nVertices = nWork + nGroups

  const dataVertexByInput = new Array(workload.nInputs).fill(-1)
  const dataVertexByOutput = new Array(workload.nOutputs).fill(-1)
  const dataVertexMem = new Array(nGroups).fill(0)
  const dataVertexGroup = []
  for (let g = 0; g < nGroups; g++) {
    const { inputIds, outputIds } = groups[g]
    let mem = 0
    for (const id of inputIds) {
      dataVertexByInput[id] = nWork + g
      mem += workload.inputs[id].bytes
    }
    for (const id of outputIds) {
      dataVertexByOutput[id] = nWork + g
      mem += workload.outputs[id].bytes
    }
    dataVertexMem[g] = mem
    dataVertexGroup[g] = {
      seqId: workload.inputs[inputIds[0]].seqId,
      blockId: workload.inputs[inputIds[0]].blockId,
      inputIds: [...inputIds],
      outputIds: [...outputIds],
    }
  }

  const computeWeights = new Array(nVertices).fill(0)
  for (let w = 0; w < nWork; w++) computeWeights[w] = workload.work[w].costMs
  const memWeights = new Array(nVertices).fill(0)
  for (let g = 0; g < nGroups; g++) memWeights[nWork + g] = dataVertexMem[g]

  // 每个数据块 → 消费/产出它的 work 集合
  const inputConsumers = new Array(workload.nInputs)
  for (let i = 0; i < workload.nInputs; i++) inputConsumers[i] = []
  const outputProducers = new Array(workload.nOutputs)
  for (let i = 0; i < workload.nOutputs; i++) outputProducers[i] = []
  for (let w = 0; w < nWork; w++) {
    const [qId, kvId] = workload.workUnitInputMap[w]
    inputConsumers[qId].push(w)
    inputConsumers[kvId].push(w)
    const [outId, lseId] = workload.workUnitOutputMap[w]
    outputProducers[outId].push(w)
    outputProducers[lseId].push(w)
  }

  // 超边：输入边 + 输出边
  const nets = []
  for (let i = 0; i < workload.nInputs; i++) {
    const pins = [dataVertexByInput[i]]
    for (const w of inputConsumers[i]) pins.push(w)
    nets.push({ pins, weight: workload.inputs[i].bytes })
  }
  for (let o = 0; o < workload.nOutputs; o++) {
    const pins = [dataVertexByOutput[o]]
    for (const w of outputProducers[o]) pins.push(w)
    nets.push({ pins, weight: workload.outputs[o].bytes })
  }

  return {
    nVertices,
    nWork,
    nGroups,
    computeWeights,
    memWeights,
    nets,
    dataVertexByInput,
    dataVertexByOutput,
    dataVertexMem,
    dataVertexGroup,
    // 每个顶点的超边列表（FM 用）
    vertexNets: buildVertexNets(nets, nVertices),
  }
}

function buildVertexNets(nets, nVertices) {
  const vertexNets = new Array(nVertices)
  for (let v = 0; v < nVertices; v++) vertexNets[v] = []
  for (let e = 0; e < nets.length; e++) {
    for (const v of nets[e].pins) vertexNets[v].push(e)
  }
  return vertexNets
}
