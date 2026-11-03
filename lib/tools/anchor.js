/**
 * The three model-facing tools of anchor-based context compression.
 *
 *   add_anchor — places a named anchor at the call site. The call itself is the
 *                anchor: its provider-assigned tool-call id is the durable
 *                identity. Names are unique per session and `session_start` is
 *                reserved.
 *   compress   — replaces the span from a start anchor up to this compress call
 *                itself with the model-authored summary. The end is implicit:
 *                this call's position on the surface. Validation and the
 *                replacement live in ../anchor/compress.js; failed validation
 *                throws before any session append.
 *   expand     — returns the full text of a compression block currently shown
 *                only as an index line under a newer summary, plus its own
 *                index so the model can walk further back.
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { compressFromAnchor, recordAnchor } from '../anchor/compress.js'
import { expandBlock } from '../anchor/blocks.js'
import {
  ADD_ANCHOR_DESCRIPTION,
  COMPRESS_DESCRIPTION,
  COMPRESS_START_PARAM_DESCRIPTION,
  COMPRESS_SUMMARY_PARAM_DESCRIPTION,
  EXPAND_BLOCK_PARAM_DESCRIPTION,
  EXPAND_DESCRIPTION,
  formatAnchorPlaced,
} from '../anchor/prompts.js'
import { formatProblems } from '../anchor/summary.js'

/** Text content helper. */
function textBlock(text) {
  return [{ type: 'text', text }]
}

/** Require the calling agent's session. */
function sessionOf(exec) {
  const session = exec?.agent?.session
  if (session === undefined || session === null) {
    throw new Error('dsh-dcp: this tool needs an agent session')
  }
  return session
}

/** Render the compress tool result. */
function renderCompressResult(value) {
  const absorbed = value.absorbed > 0
    ? ` Absorbed ${value.absorbed} prior block(s); the chain reaches back ${value.chain.length}: ` +
      value.chain
        .map((block) => `块 ${block.blockId}「${block.anchor}」${block.gist || '(无摘要)'} — expand({block: ${block.blockId}})`)
        .join('；')
    : ''
  return (
    `Compressed from "${value.anchor}" to this compress call ` +
    `(${value.shadowedMessages} message(s) elided, ~${value.freedTokens} tokens freed) as block ${value.blockId}.${absorbed} ` +
    'The summary now replaces that span in your context, right before this compress call; the call and this ' +
    'confirmation stay visible. The session log keeps the original events.' +
    formatProblems(value.problems)
  )
}

/**
 * Register `add_anchor`, `compress` and `expand`.
 * @param {object} ctx - plugin context carrying `tools`.
 * @param {{ logger: object }} deps - plugin logger.
 */
export function registerAnchorTools(ctx, { logger }) {
  ctx.tools.register(defineTool({
    name: 'add_anchor',
    description: ADD_ANCHOR_DESCRIPTION,
    parameters: {
      anchor: {
        type: 'string',
        required: true,
        description: 'Unique name for this anchor (no whitespace; "session_start" is reserved)',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          anchor: { type: 'string', required: true },
        },
        additionalProperties: false,
      },
      render: (_args, value) => textBlock(formatAnchorPlaced(value.anchor)),
    },
    async execute(args, exec) {
      const result = recordAnchor({
        session: sessionOf(exec),
        anchor: args.anchor,
        callId: exec.callId,
      })
      logger.info('Anchor placed', { anchor: result.anchor, callId: exec.callId })
      return result
    },
  }))

  ctx.tools.register(defineTool({
    name: 'compress',
    description: COMPRESS_DESCRIPTION,
    parameters: {
      startAnchor: {
        type: 'string',
        required: true,
        description: COMPRESS_START_PARAM_DESCRIPTION,
      },
      summary: {
        type: 'string',
        required: true,
        description: COMPRESS_SUMMARY_PARAM_DESCRIPTION,
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          anchor: { type: 'string', required: true },
          blockId: { type: 'integer', required: true },
          shadowedMessages: { type: 'integer', required: true },
          absorbed: { type: 'integer', required: true },
          absorbedBlockIds: { type: 'array', items: { type: 'integer' }, required: true },
          chain: {
            type: 'array',
            items: {
              type: 'object',
              properties: {
                blockId: { type: 'integer', required: true },
                anchor: { type: 'string', required: true },
                gist: { type: 'string', required: true },
              },
              additionalProperties: false,
            },
            required: true,
          },
          shadowedTokens: { type: 'integer', required: true },
          freedTokens: { type: 'integer', required: true },
          problems: { type: 'array', items: { type: 'string' }, required: true },
        },
        additionalProperties: false,
      },
      render: (_args, value) => textBlock(renderCompressResult(value)),
    },
    async execute(args, exec) {
      const session = sessionOf(exec)
      const tokenMeter = typeof ctx.get === 'function' ? ctx.get('tokenMeter') : undefined
      const result = compressFromAnchor({
        session,
        tokenMeter,
        startAnchor: args.startAnchor,
        summary: args.summary,
        callId: exec.callId,
        logger,
      })
      try {
        logger.saveContext(session.id, session.deriveMessages())
      } catch {
        // debug snapshots must never fail the call
      }
      return {
        anchor: result.anchor,
        blockId: result.blockId,
        shadowedMessages: result.shadowedMessages,
        absorbed: result.absorbed,
        absorbedBlockIds: result.absorbedBlockIds,
        chain: result.chain,
        shadowedTokens: result.shadowedTokens,
        freedTokens: result.freedTokens,
        problems: result.problems,
      }
    },
  }))

  ctx.tools.register(defineTool({
    name: 'expand',
    description: EXPAND_DESCRIPTION,
    parameters: {
      block: {
        type: 'integer',
        required: true,
        description: EXPAND_BLOCK_PARAM_DESCRIPTION,
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          block: { type: 'integer', required: true },
          content: { type: 'string', required: true },
        },
        additionalProperties: false,
      },
      render: (_args, value) => textBlock(value.content),
    },
    async execute(args, exec) {
      const result = expandBlock({ session: sessionOf(exec), block: args.block })
      logger.info('Expanded compression block', { blockId: result.block, callId: exec.callId })
      return result
    },
  }))
}
