// 注意力 mask 模型。
// 官方 DCP 用 per-token 的 attend range（每行 1~2 个 [start, end) 区间）表示 mask；
// 此处实现论文 §7 使用的四种 mask：causal / lambda / causal_blockwise / shared_question。
//
// maskSpec 格式：
//   { type: 'causal' }
//   { type: 'lambda', sink, window }
//   { type: 'causal_blockwise', blockSize, windowBlocks, sinkBlocks }
//   { type: 'shared_question', questionTokens, answerTokens, nAnswers }
//     （sequence 长度 = questionTokens + nAnswers * answerTokens）

// 合并重叠/相邻区间，最多保留 maxRanges 个（取最长者）
export function mergeRanges(ranges, maxRanges = 2) {
  if (ranges.length === 0) return []
  const sorted = ranges
    .filter(([s, e]) => e > s)
    .sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const merged = []
  for (const [s, e] of sorted) {
    const last = merged[merged.length - 1]
    if (last && s <= last[1]) {
      last[1] = Math.max(last[1], e)
    } else {
      merged.push([s, e])
    }
  }
  if (merged.length <= maxRanges) return merged
  // 超出上限：按长度排序保留最长的 maxRanges 个（论文实现限制每 token 至多 2 个 range）
  merged.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]))
  return merged.slice(0, maxRanges)
}

// 生成指定 mask 的逐 token range 列表（长度 L）
export function buildMaskRanges(maskSpec, seqlen) {
  const type = maskSpec && maskSpec.type ? maskSpec.type : 'causal'
  const ranges = new Array(seqlen)
  for (let t = 0; t < seqlen; t++) ranges[t] = []
  switch (type) {
    case 'causal': {
      // token t 关注 [0, t+1)
      for (let t = 0; t < seqlen; t++) ranges[t] = [[0, t + 1]]
      break
    }
    case 'lambda': {
      const sink = maskSpec.sink ?? 64
      const window = maskSpec.window ?? 4096
      for (let t = 0; t < seqlen; t++) {
        const r1 = [0, Math.min(sink, t + 1)]
        const r2 = t >= sink ? [Math.max(sink, t - window), t] : null
        ranges[t] = mergeRanges(r2 ? [r1, r2] : [r1])
      }
      break
    }
    case 'causal_blockwise': {
      const B = maskSpec.blockSize ?? 256
      const winBlocks = maskSpec.windowBlocks ?? 2
      const sinkBlocks = maskSpec.sinkBlocks ?? 1
      const nb = Math.ceil(seqlen / B)
      for (let t = 0; t < seqlen; t++) {
        const b = Math.floor(t / B)
        if (b === nb - 1) {
          // 最后一个 block（含测试样例）关注全部前缀
          ranges[t] = [[0, t + 1]]
          continue
        }
        const withinStart = b * B
        const winStart = Math.max(0, b - winBlocks) * B
        const sinkEnd = Math.min(sinkBlocks * B, seqlen)
        const rr = []
        if (winStart < withinStart) rr.push([winStart, withinStart])
        if (sinkEnd > winStart) rr.push([0, Math.min(sinkEnd, winStart)]) // sink 与窗口起点相接则合并
        rr.push([withinStart, t + 1])
        ranges[t] = mergeRanges(rr)
      }
      break
    }
    case 'shared_question': {
      const q = maskSpec.questionTokens ?? 0
      const a = maskSpec.answerTokens ?? 0
      const n = maskSpec.nAnswers ?? 4
      for (let t = 0; t < seqlen; t++) {
        if (t < q) {
          ranges[t] = [[0, t + 1]]
        } else {
          const idx = Math.min(n - 1, Math.floor((t - q) / a))
          const aStart = q + idx * a
          const rr = [[0, q]]
          if (t > aStart) rr.push([aStart, t + 1])
          ranges[t] = mergeRanges(rr)
        }
      }
      break
    }
    default:
      throw new Error('Unknown mask type: ' + type)
  }
  return ranges
}

export function maskLabel(maskSpec) {
  const type = maskSpec && maskSpec.type ? maskSpec.type : 'causal'
  switch (type) {
    case 'causal': return 'causal'
    case 'lambda': return 'lambda(sink=' + (maskSpec.sink ?? 64) + ',window=' + (maskSpec.window ?? 4096) + ')'
    case 'causal_blockwise': return 'causal_blockwise(B=' + (maskSpec.blockSize ?? 256) + ',w=' + (maskSpec.windowBlocks ?? 2) + 'blk,sink=' + (maskSpec.sinkBlocks ?? 1) + 'blk)'
    case 'shared_question': return 'shared_question(q=' + (maskSpec.questionTokens ?? 0) + ',a=' + (maskSpec.answerTokens ?? 0) + 'x' + (maskSpec.nAnswers ?? 4) + ')'
    default: return type
  }
}

// (q块, kv块) 是否产生 work block：q 块内存在 token 的某个 range 与 kv 块区间相交
export function blockPairActive(rangesForTokens, qStart, qEnd, kvStart, kvEnd) {
  for (let t = qStart; t < qEnd; t++) {
    for (const [s, e] of rangesForTokens[t]) {
      if (Math.min(e, kvEnd) > Math.max(s, kvStart)) return true
    }
  }
  return false
}
