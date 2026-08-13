// dsh-dcp 核心工具函数：单位换算、上下文、常用数学。
// 忠实于 DCP (SOSP'25) 官方实现的单位约定：成本模型输出毫秒 (ms)。

// ---------- 数据类型 ----------
// 与官方 DType.element_size 一致：bf16/fp16 = 2 字节，fp32 = 4 字节
export const DType = Object.freeze({
  BF16: 2,
  FP16: 2,
  FP32: 4,
})

export function bytesPerElement(dtype) {
  return dtype
}

// ---------- 集群上下文 ----------
// 官方 CommunicationCostModel 默认：节点内 4800 Gbps，节点间 400 Gbps
export function makeContext({
  nodes = 1,
  devicesPerNode = 1,
  intraBandwidthGbps = 4800,
  interBandwidthGbps = 400,
  intraLatencyMs = 0,
  interLatencyMs = 0,
} = {}) {
  return {
    nNodes: nodes,
    nDevicesPerNode: devicesPerNode,
    nTotalDevices: nodes * devicesPerNode,
    intraBandwidthGbps,
    interBandwidthGbps,
    intraLatencyMs,
    interLatencyMs,
  }
}

export function allDevices(ctx) {
  const out = []
  for (let n = 0; n < ctx.nNodes; n++) {
    for (let d = 0; d < ctx.nDevicesPerNode; d++) out.push([n, d])
  }
  return out
}

export function deviceKey(dev) {
  return dev[0] * 100000 + dev[1]
}

// ---------- 常用数学 ----------
export function sum(arr) {
  let s = 0
  for (const x of arr) s += x
  return s
}

export function max(arr) {
  let m = -Infinity
  for (const x of arr) if (x > m) m = x
  return m
}

export function min(arr) {
  let m = Infinity
  for (const x of arr) if (x < m) m = x
  return m
}

// ---------- 格式化 ----------
export function fmtBytes(n) {
  if (!Number.isFinite(n)) return String(n)
  if (n >= 1e12) return (n / 1e12).toFixed(2) + ' TB'
  if (n >= 1e9) return (n / 1e9).toFixed(2) + ' GB'
  if (n >= 1e6) return (n / 1e6).toFixed(2) + ' MB'
  if (n >= 1e3) return (n / 1e3).toFixed(2) + ' KB'
  return n.toFixed(0) + ' B'
}

export function fmtMs(ms) {
  if (!Number.isFinite(ms)) return String(ms)
  if (ms >= 1e3) return (ms / 1e3).toFixed(3) + ' s'
  if (ms >= 1) return ms.toFixed(2) + ' ms'
  return (ms * 1e3).toFixed(2) + ' µs'
}

export function fmtPct(x) {
  return (x * 100).toFixed(1) + '%'
}

export function fmtX(x) {
  return x.toFixed(2) + 'x'
}
