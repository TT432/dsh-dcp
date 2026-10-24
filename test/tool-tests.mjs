// dcp_compress tool wiring test: registration, route resolution, execution,
// and rendering against the real @deepseek-ai/dsh-tools + dsh-llm packages.
//   node --test test/tool-tests.mjs
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import { registerCompressTool } from '../lib/tools/compress.js'

function makeLlm({ provider = 'fake-provider', model = 'fake-model', text } = {}) {
  const calls = []
  return {
    calls,
    async *stream(options) {
      calls.push(options)
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text }
      yield { type: 'block-end', index: 0, block: { type: 'text', text } }
      yield { type: 'usage', usage: { inputTokens: 20, outputTokens: 10 } }
      yield { type: 'finish', reason: { kind: 'stop' } }
    },
  }
}

function register(llm) {
  const tools = []
  const ctx = {
    llm,
    tools: { register: (tool) => tools.push(tool) },
  }
  registerCompressTool(ctx)
  return tools
}

test('registerCompressTool：注册 dcp_compress 且 schema/render 完整', () => {
  const tools = register(makeLlm({ text: 'x' }))
  assert.equal(tools.length, 1)
  const tool = tools[0]
  assert.equal(tool.name, 'dcp_compress')
  assert.equal(tool.parameters.type, 'object')
  assert.ok(tool.parameters.properties.text)
  assert.deepEqual(tool.parameters.required, ['text'])
  assert.ok(tool.output.schema)
  assert.equal(typeof tool.output.render, 'function')
  assert.equal(typeof tool.execute, 'function')
})

test('dcp_compress：从会话 requestContext 解析路由并执行', async () => {
  const original = 'Long prompt body for compression. '.repeat(10)
  const compressed = original.slice(0, Math.round(original.length * 0.5))
  const llm = makeLlm({ text: compressed })
  const tool = register(llm)[0]
  const controller = new AbortController()
  const exec = {
    signal: controller.signal,
    agent: { session: { requestContext: () => ({ provider: 'fake-provider', model: 'fake-model' }) } },
  }
  const value = await tool.execute({ text: original, targetRatio: 0.5, steps: 1 }, exec)
  assert.equal(value.text, compressed.trim())
  assert.equal(value.stepsUsed, 1)
  // 路由来自 exec.agent.session.requestContext()
  assert.equal(llm.calls.length, 1)
  assert.equal(llm.calls[0].provider, 'fake-provider')
  assert.equal(llm.calls[0].model, 'fake-model')
  assert.equal(llm.calls[0].signal, controller.signal)
  const rendered = tool.output.render({}, value)
  assert.ok(rendered[0].text.includes('LLM-DCP 压缩结果'))
  assert.ok(rendered[0].text.includes(compressed.trim()))
})

test('dcp_compress：显式 provider/model 覆盖会话路由', async () => {
  const llm = makeLlm({ text: 'short' })
  const tool = register(llm)[0]
  const exec = {
    signal: new AbortController().signal,
    agent: { session: { requestContext: () => ({ provider: 'session-provider', model: 'session-model' }) } },
  }
  await tool.execute({ text: 'a much longer text body '.repeat(5), targetRatio: 0.5, steps: 1, provider: 'explicit-provider', model: 'explicit-model' }, exec)
  assert.equal(llm.calls[0].provider, 'explicit-provider')
  assert.equal(llm.calls[0].model, 'explicit-model')
})

test('dcp_compress：无路由时明确报错（不静默失败）', async () => {
  const tool = register(makeLlm({ text: 'x' }))[0]
  await assert.rejects(
    tool.execute({ text: 'text' }, { signal: new AbortController().signal }),
    /无法确定 LLM 路由/,
  )
})

test('dcp_compress：参数校验由 defineTool 执行', async () => {
  const tool = register(makeLlm({ text: 'x' }))[0]
  await assert.rejects(
    tool.execute({}, { signal: new AbortController().signal }),
    /text|required/i,
  )
})
