// 块生成：把每个序列按 (head, seq) 两个维度切成细粒度数据块，
// 依据 mask 构造计算块（work block），并给出 colocation 约束与逐块成本。
// 忠实移植官方 dcp/data/dataloader.py 的 _generate_workload（§4.1）。
import { DType } from './core.js'
import { buildMaskRanges, blockPairActive, maskLabel } from './masks.js'
import { AttnRooflineCostModel } from './costmodel.js'

// 生成 workload。返回结构化的 WorkloadSpec（无 torch 依赖的纯数据）。
export function generateWorkload({
  seqlens,             // 每条序列的真实 token 数
  blockSize,           // seq 维块大小（token）
  headBlockSize,       // head 维块大小（Q/Out 用）
  nHeads,
  nQueryGroups,
  headDim,
  dtype = DType.BF16,
  maskSpec = { type: 'causal' },
  costModel = null,
}) {
  if (!costModel) costModel = new AttnRooflineCostModel()
  const nSeqs = seqlens.length
  const isGqa = nHeads > nQueryGroups
  if (nHeads % nQueryGroups !== 0) throw new Error('nHeads must be divisible by nQueryGroups')
  if (nHeads % headBlockSize !== 0) throw new Error('nHeads must be divisible by headBlockSize')
  const headsPerQueryGroup = nHeads / nQueryGroups
  if (isGqa) {
    if (headBlockSize > headsPerQueryGroup || headsPerQueryGroup % headBlockSize !== 0) {
      throw new Error('for GQA, headBlockSize must divide headsPerQueryGroup')
    }
  }
  const headBlockSizeKV = isGqa ? 1 : headBlockSize
  const nHeadBlocksQ = nHeads / headBlockSize
  const nHeadBlocksKV = nQueryGroups / headBlockSizeKV

  const paddedSeqlens = seqlens.map((L) => Math.ceil(L / blockSize) * blockSize)
  const seqBlockCounts = paddedSeqlens.map((L) => L / blockSize)

  const inputs = []        // {type, seqId, headBlockId, blockId, nTokens, blockSize, headBlockSize, bytes, tokenDim}
  const outputs = []       // {type, seqId, headBlockId, blockId, nTokens, blockSize, bytes}
  const work = []          // {seqId, headBlockId, qBlockId, kvBlockId, qInputId, kvInputId, outId, lseId, costMs, flopsMs, memMs, sparsity}
  const workUnitInputMap = []
  const workUnitOutputMap = []
  const colocationConstraints = [] // [{inputIds: [], outputIds: []}] 每个 (seq, seq-block) 一组

  const rangesCache = new Map() // seqId -> ranges

  const getRanges = (seqId) => {
    let r = rangesCache.get(seqId)
    if (!r) {
      r = buildMaskRanges(maskSpec, seqlens[seqId])
      rangesCache.set(seqId, r)
    }
    return r
  }

  // 每个数据块的字节数（nTokens 为真实 token 数，与官方 numel(padded=False) 一致）
  const qBytes = (tokens) => tokens * headBlockSize * headDim * bytesPerElem(dtype)
  const kvBytes = (tokens) => tokens * 2 * headBlockSizeKV * headDim * bytesPerElem(dtype)
  const outBytes = (tokens) => tokens * headBlockSize * headDim * bytesPerElem(dtype)
  const lseBytes = (tokens) => tokens * headBlockSize * DType.FP32

  let cumInput = 0
  let cumOutput = 0
  for (let seqId = 0; seqId < nSeqs; seqId++) {
    const nBlocks = seqBlockCounts[seqId]
    const nTokens = seqlens[seqId]
    const actualTokens = (blk) => Math.min(blockSize, nTokens - blk * blockSize)

    // --- Q 输入块 ---
    const qInputIds = []
    for (let hb = 0; hb < nHeadBlocksQ; hb++) {
      for (let b = 0; b < nBlocks; b++) {
        const id = cumInput + hb * nBlocks + b
        inputs[id] = {
          type: 'Q', seqId, headBlockId: hb, blockId: b,
          nTokens: actualTokens(b), blockSize, headBlockSize,
          bytes: qBytes(actualTokens(b)),
        }
        qInputIds.push(id)
      }
    }
    // --- KV 输入块 ---
    const kvInputIds = []
    for (let hb = 0; hb < nHeadBlocksKV; hb++) {
      for (let b = 0; b < nBlocks; b++) {
        const id = cumInput + nHeadBlocksQ * nBlocks + hb * nBlocks + b
        inputs[id] = {
          type: 'KV', seqId, headBlockId: hb, blockId: b,
          nTokens: actualTokens(b), blockSize, headBlockSize: headBlockSizeKV,
          bytes: kvBytes(actualTokens(b)),
        }
        kvInputIds.push(id)
      }
    }
    // --- Out/LSE 输出块（按官方交错：out = base + i*2, lse = base + i*2+1）---
    const outIds = []
    const lseIds = []
    for (let hb = 0; hb < nHeadBlocksQ; hb++) {
      for (let b = 0; b < nBlocks; b++) {
        const outId = cumOutput + hb * nBlocks * 2 + b * 2
        const lseId = cumOutput + hb * nBlocks * 2 + b * 2 + 1
        outputs[outId] = {
          type: 'Out', seqId, headBlockId: hb, blockId: b,
          nTokens: actualTokens(b), blockSize, bytes: outBytes(actualTokens(b)),
        }
        outputs[lseId] = {
          type: 'LSE', seqId, headBlockId: hb, blockId: b,
          nTokens: actualTokens(b), blockSize, bytes: lseBytes(actualTokens(b)),
        }
        outIds.push(outId)
        lseIds.push(lseId)
      }
    }

    // --- 计算块（work）---
    const ranges = getRanges(seqId)
    for (let k = 0; k < nHeadBlocksQ; k++) {
      for (let i = 0; i < nBlocks; i++) {
        const qStart = i * blockSize
        const qEnd = Math.min(qStart + blockSize, nTokens)
        const outId = cumOutput + k * nBlocks * 2 + i * 2
        const lseId = outId + 1
        for (let j = 0; j < nBlocks; j++) {
          const kvStart = j * blockSize
          const kvEnd = Math.min(kvStart + blockSize, nTokens)
          if (!blockPairActive(ranges, qStart, qEnd, kvStart, kvEnd)) continue
          const kvHeadGroup = isGqa
            ? Math.floor((k * headBlockSize) / headsPerQueryGroup)
            : k
          const inputQId = cumInput + k * nBlocks + i
          const inputKVId = cumInput + nHeadBlocksQ * nBlocks + kvHeadGroup * nBlocks + j
          // 成本：q/k 长度用 padded blockSize，mask 用逐行 range。
          // 官方对每个 work unit 传总 head 数（绝对成本被放大 nHeads/headBlockSize 倍，仅影响
          // 调度相对排序）；这里使用该 head 块的真实 head/kv-group 数，得到绝对耗时一致的 roofline 估计。
          const headsPerWorkUnit = headBlockSize
          const kvGroupsPerWorkUnit = Math.max(1, Math.ceil(nQueryGroups * headBlockSize / nHeads))
          const { flopsTimeMs, memTimeMs, sparsity } = costModel.getCost({
            seqlensQ: [blockSize],
            seqlensK: [blockSize],
            rangesForTokens: ranges,
            qStart, qEnd,
            kvStart, kvEnd,
            paddedQ: blockSize,
            paddedK: blockSize,
            nHeads: headsPerWorkUnit,
            nQueryGroups: kvGroupsPerWorkUnit,
            headDim,
            isForward: true,
          })
          const costMs = Math.max(flopsTimeMs, memTimeMs)
          const wid = work.length
          work[wid] = {
            seqId, headBlockId: k, qBlockId: i, kvBlockId: j,
            qInputId: inputQId, kvInputId: inputKVId,
            outId, lseId,
            costMs, flopsMs: flopsTimeMs, memMs: memTimeMs, sparsity,
          }
          workUnitInputMap.push([inputQId, inputKVId])
          workUnitOutputMap.push([outId, lseId])
        }
      }
    }

    // --- colocation 约束：每个 seq-block 的 Q+KV 输入同机、Out+LSE 输出同机 ---
    for (let b = 0; b < nBlocks; b++) {
      const inputGroup = []
      const outputGroup = []
      for (let hb = 0; hb < nHeadBlocksQ; hb++) inputGroup.push(cumInput + hb * nBlocks + b)
      for (let hb = 0; hb < nHeadBlocksKV; hb++) inputGroup.push(cumInput + nHeadBlocksQ * nBlocks + hb * nBlocks + b)
      for (let hb = 0; hb < nHeadBlocksQ; hb++) {
        outputGroup.push(cumOutput + hb * nBlocks * 2 + b * 2)
        outputGroup.push(cumOutput + hb * nBlocks * 2 + b * 2 + 1)
      }
      colocationConstraints.push({ inputIds: inputGroup, outputIds: outputGroup })
    }

    cumInput += nBlocks * (nHeadBlocksQ + nHeadBlocksKV)
    cumOutput += nBlocks * nHeadBlocksQ * 2
  }

  return {
    seqlens,
    blockSize,
    headBlockSize,
    nHeads,
    nQueryGroups,
    headDim,
    dtype,
    maskSpec,
    maskLabel: maskLabel(maskSpec),
    isGqa,
    nHeadBlocksQ,
    nHeadBlocksKV,
    paddedSeqlens,
    seqBlockCounts,
    nSequences: nSeqs,
    inputs,       // id -> meta
    outputs,      // id -> meta
    work,         // id -> meta
    workUnitInputMap,
    workUnitOutputMap,
    colocationConstraints,
    nInputs: inputs.length,
    nOutputs: outputs.length,
    nWork: work.length,
    nTokensTotal: seqlens.reduce((a, b) => a + b, 0),
    nSeqBlocksTotal: seqBlockCounts.reduce((a, b) => a + b, 0),
    rangesPerSeq: rangesCache,
  }
}

function bytesPerElem(dtype) {
  return dtype
}