/**
 * The two model-facing tools of anchor-based context compression.
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
 */

import { defineTool } from '@deepseek-ai/dsh-tools'
import { compressFromAnchor, recordAnchor } from '../anchor/compress.js'
import {
  ADD_ANCHOR_DESCRIPTION,
  COMPRESS_DESCRIPTION,
  COMPRESS_SUMMARY_PARAM_DESCRIPTION,
  formatAnchorPlaced,
} from '../anchor/prompts.js'

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
  const absorbed = value.absorbed > 0 ? ` Absorbed ${value.absorbed} prior compression(s).` : ''
  return (
    `Compressed from "${value.anchor}" to this compress call ` +
    `(${value.shadowedMessages} message(s) elided, ~${value.freedTokens} tokens freed).${absorbed} ` +
    'The summary now replaces that span in your context, right before this compress call; the call and this ' +
    'confirmation stay visible. The session log keeps the original events.'
  )
}

/**
 * Register `add_anchor` and `compress`.
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
        description: 'Anchor where compression starts (consumed); "session_start" compresses from the original prompt',
      },
      summary: {
        type: 'string',
        required: true,
        description: COMPRESS_SUMMARY_PARAM_DESCRIPTION,
      },
      force: {
        type: 'boolean',
        description: 'Required (=true) when startAnchor is "session_start": full-session compression while work is in progress causes amnesia. Confirm only when the current work unit is complete, and make the summary carry 当前任务/刚完成的动作/下一步 (current task, last action, next step).',
      },
    },
    output: {
      schema: {
        type: 'object',
        properties: {
          anchor: { type: 'string', required: true },
          shadowedMessages: { type: 'integer', required: true },
          absorbed: { type: 'integer', required: true },
          shadowedTokens: { type: 'integer', required: true },
          freedTokens: { type: 'integer', required: true },
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
        force: args.force,
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
        shadowedMessages: result.shadowedMessages,
        absorbed: result.absorbed,
        shadowedTokens: result.shadowedTokens,
        freedTokens: result.freedTokens,
      }
    },
  }))
}
