// 计算/通信调度。
// 忠实论文 §4.3 的 Block Scheduling 伪代码与官方 dcp/core/pipeline.py：
//   - 每个设备把计算块分为 T 个 division（默认 T=4）
//   - 第 0 个 division：无需通信的计算块
//   - 中间 division：按 per-division 通信上限（总量的 1/T）贪心装入，设备间按当前计算负载最小者先调度
//   - 最后一个 division：剩余全部计算块
//   - 输出块在 divisions 结束后传输到其所在设备，并在目标设备做 reduction
import { deviceKey } from './core.js'

export function scheduleDivisions(context, workload, placement, {
  nDivisions = 4,
} = {}) {
  const T = Math.max(1, Math.floor(nDivisions))
  const nWork = workload.nWork
  const { workToDevice, inputToDevice, outputToDevice } = placement
  const inputSizes = workload.inputs.map(m => m.bytes)
  const outputSizes = workload.outputs.map(m => m.bytes)

  // 每个 work 需要的远端输入（去重：同一 (src,dst) 的多块合并计费在官方 cost model 中逐块计，
  // 但 division 上限按 (dst,src) 对统计）
  const workComms = new Array(nWork) // [{src, dst, bytes}]
  const workAllLocal = new Array(nWork)
  for (let w = 0; w < nWork; w++) {
    const dst = workToDevice[w]
    const comms = []
    let allLocal = true
    const seen = new Set()
    for (const inputId of workload.workUnitInputMap[w]) {
      const src = inputToDevice[inputId]
      if (src[0] !== dst[0] || src[1] !== dst[1]) {
        allLocal = false
        const key = deviceKey(src) + ':' + deviceKey(dst)
        if (!seen.has(key)) {
          seen.add(key)
          comms.push({ src, dst, bytes: inputSizes[inputId] })
        } else {
          // 同一对设备间的多输入块合并（官方按块计费，这里按块计费但限制按对统计）
          comms.find(c => deviceKey(c.src) === deviceKey(src) && deviceKey(c.dst) === deviceKey(dst)).bytes += inputSizes[inputId]
        }
      }
    }
    workComms[w] = comms
    workAllLocal[w] = allLocal
  }

  // (dst, src) -> 总字节（输入通信需求）
  const commRequirements = new Map() // dstKey:srcKey -> bytes
  const commPairs = new Map()        // dstKey:srcKey -> {src, dst, bytes}
  for (let w = 0; w < nWork; w++) {
    for (const c of workComms[w]) {
      const key = deviceKey(c.dst) + ':' + deviceKey(c.src)
      commRequirements.set(key, (commRequirements.get(key) ?? 0) + c.bytes)
      if (!commPairs.has(key)) commPairs.set(key, { src: c.src, dst: c.dst, bytes: 0 })
      commPairs.get(key).bytes += c.bytes
    }
  }
  const perDivLimit = new Map()
  for (const [key, bytes] of commRequirements) perDivLimit.set(key, bytes / T)

  const nDevices = context.nTotalDevices
  const devs = []
  for (let n = 0; n < context.nNodes; n++)
    for (let d = 0; d < context.nDevicesPerNode; d++) devs.push([n, d])
  const devIdx = new Map(devs.map((d, i) => [deviceKey(d), i]))

  const workByDevice = new Array(nDevices)
  for (let i = 0; i < nDevices; i++) workByDevice[i] = []
  for (let w = 0; w < nWork; w++) workByDevice[devIdx.get(deviceKey(workToDevice[w]))].push(w)

  // 每个 division 的当前 (dst,src) 用量（用于检查上限）
  const divisions = [] // [{work: [w...], comm: [{src,dst,bytes}...]}]
  const currentUsage = () => {
    const u = new Map()
    for (const [key] of perDivLimit) u.set(key, 0)
    return u
  }

  const scheduled = new Set()

  // ---- division 0：无需通信 ----
  // 注：中间 division 的通信预算按 division 各自重置（每 division 各 1/T 的预算）
  {
    const div = { work: [], comm: [] }
    for (let w = 0; w < nWork; w++) {
      if (workAllLocal[w] && !scheduled.has(w)) {
        div.work.push(w)
        scheduled.add(w)
      }
    }
    divisions.push(div)
  }

  // ---- 中间 division：1..T-2 ----
  for (let divIdx = 1; divIdx <= T - 2; divIdx++) {
    const div = { work: [], comm: [] }
    const usage = currentUsage()
    // 按论文：反复选择当前 division 计算负载最小的设备，装入其可调度块
    let progress = true
    while (progress) {
      progress = false
      // 设备按当前 division 计算负载升序
      const divCompute = new Array(nDevices).fill(0)
      for (const w of div.work) divCompute[devIdx.get(deviceKey(workToDevice[w]))] += workload.work[w].costMs
      const order = devs
        .map((d, i) => [i, divCompute[i]])
        .sort((a, b) => a[1] - b[1] || a[0] - b[0])
      for (const [di] of order) {
        // 该设备上未调度的块，按成本降序尝试装入
        const unscheduled = workByDevice[di].filter(w => !scheduled.has(w))
        if (unscheduled.length === 0) continue
        unscheduled.sort((a, b) => workload.work[b].costMs - workload.work[a].costMs)
        for (const w of unscheduled) {
          if (scheduled.has(w)) continue
          // 检查 per-div 通信上限
          let ok = true
          const touched = []
          for (const c of workComms[w]) {
            const key = deviceKey(c.dst) + ':' + deviceKey(c.src)
            if ((usage.get(key) ?? 0) + c.bytes > perDivLimit.get(key) + 1e-9) { ok = false; break }
            touched.push(key)
          }
          if (!ok) continue
          for (const c of workComms[w]) {
            const key = deviceKey(c.dst) + ':' + deviceKey(c.src)
            usage.set(key, usage.get(key) + c.bytes)
          }
          div.work.push(w)
          for (const c of workComms[w]) div.comm.push(c)
          scheduled.add(w)
          progress = true
        }
      }
    }
    divisions.push(div)
  }

  // ---- 最后一个 division：剩余全部 ----
  {
    const div = { work: [], comm: [] }
    for (let w = 0; w < nWork; w++) {
      if (!scheduled.has(w)) {
        div.work.push(w)
        for (const c of workComms[w]) div.comm.push(c)
      }
    }
    divisions.push(div)
  }

  // ---- 输出传输与 reduction ----
  // 每个输出块由若干 work 贡献；输出块放在 outputToDevice；贡献设备 != 输出设备 => 传输
  // 输出设备做 reduction（多个贡献时）
  const outputTransfers = [] // {src, dst, bytes}
  const reductionByDevice = new Map() // dstKey -> [{outId, nContributors}]
  const outContributors = new Map()   // outId -> [workId...]
  for (let w = 0; w < nWork; w++) {
    for (const outId of workload.workUnitOutputMap[w]) {
      if (!outContributors.has(outId)) outContributors.set(outId, [])
      outContributors.get(outId).push(w)
    }
  }
  for (const [outId, contributors] of outContributors) {
    const dst = outputToDevice[outId]
    const srcDevs = new Set(contributors.map(w => deviceKey(workToDevice[w])))
    if (srcDevs.size <= 1) continue
    const hasLocal = srcDevs.has(deviceKey(dst))
    // 每个非本地贡献设备发送部分和到输出设备（本地贡献者就地参与 reduction）
    for (const w of contributors) {
      const src = workToDevice[w]
      if (src[0] !== dst[0] || src[1] !== dst[1]) {
        outputTransfers.push({ src, dst, bytes: outputSizes[outId] })
      }
    }
    if (!reductionByDevice.has(deviceKey(dst))) reductionByDevice.set(deviceKey(dst), [])
    reductionByDevice.get(deviceKey(dst)).push({ outId, nContributors: contributors.length })
  }

  // ---- 每设备指令计划（5 类指令的紧凑表示）----
  // [{type:'attention', work:[ids]}, {type:'comm_launch', ops:[{src,dst,bytes}]}, {type:'comm_wait'},
  //  {type:'reduction', outIds:[...]}, {type:'copy', ...}]
  const planPerDevice = new Array(nDevices)
  const divisionStats = new Array(nDevices)
  for (let i = 0; i < nDevices; i++) {
    planPerDevice[i] = []
    divisionStats[i] = []
  }
  for (let d = 0; d < nDevices; d++) {
    for (let di = 0; di < divisions.length; di++) {
      const div = divisions[di]
      const myWork = div.work.filter(w => deviceKey(workToDevice[w]) === deviceKey(devs[d]))
      const myComm = div.comm.filter(c => deviceKey(c.dst) === deviceKey(devs[d]))
      let computeMs = 0
      for (const w of myWork) computeMs += workload.work[w].costMs
      divisionStats[d].push({ division: di, nWork: myWork.length, computeMs, commBytes: sumBytes(myComm) })
      if (myComm.length > 0) {
        planPerDevice[d].push({ type: 'comm_launch', ops: myComm })
        planPerDevice[d].push({ type: 'comm_wait' })
      }
      if (myWork.length > 0) {
        planPerDevice[d].push({ type: 'attention', work: myWork, masks: myWork.map(w => ({ seqId: workload.work[w].seqId, q: workload.work[w].qBlockId, kv: workload.work[w].kvBlockId, sparsity: workload.work[w].sparsity })) })
      }
    }
    // 输出阶段
    const myTransfers = outputTransfers.filter(t => deviceKey(t.src) === deviceKey(devs[d]))
    if (myTransfers.length > 0) {
      planPerDevice[d].push({ type: 'comm_launch', ops: myTransfers })
      planPerDevice[d].push({ type: 'comm_wait' })
    }
    const myReductions = reductionByDevice.get(deviceKey(devs[d])) ?? []
    if (myReductions.length > 0) {
      planPerDevice[d].push({ type: 'reduction', outIds: myReductions.map(r => r.outId) })
    }
  }

  return {
    nDivisions: divisions.length,
    divisions,
    outputTransfers,
    reductions: [...reductionByDevice.entries()].map(([k, v]) => ({ device: k, outputs: v })),
    planPerDevice,
    divisionStats,
    totalInputCommBytes: [...commRequirements.values()].reduce((a, b) => a + b, 0),
    totalOutputCommBytes: outputTransfers.reduce((a, t) => a + t.bytes, 0),
  }
}

function sumBytes(comms) {
  let s = 0
  for (const c of comms) s += c.bytes
  return s
}