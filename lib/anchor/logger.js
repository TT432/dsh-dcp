/**
 * Debug logger for dsh-dcp.
 *
 * `DCP_DEBUG` (anything except empty/0/false/off/no) enables per-session
 * context snapshots and structured lines under `$DSH_HOME/logs/dcp/`
 * (default `~/.dsh/logs/dcp`). Otherwise file logging is a no-op shell so call
 * sites stay unconditional. Cordis logging is attempted regardless; this dsh
 * build has no readable sink for it, so the files are the practical record.
 */

import { appendFileSync, existsSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'

/**
 * Interpret a truthy debug flag.
 * @param {unknown} raw - raw environment value.
 * @returns {boolean} whether debug logging is enabled.
 */
export function isDebugEnabled(raw) {
  const value = String(raw ?? '').trim().toLowerCase()
  return value !== '' && value !== '0' && value !== 'false' && value !== 'off' && value !== 'no'
}

/** Resolve `$DSH_HOME` the way dsh does (environment first, then `~/.dsh`). */
function resolveDshHome() {
  const fromEnv = typeof process.env.DSH_HOME === 'string' ? process.env.DSH_HOME.trim() : ''
  return fromEnv.length > 0 ? fromEnv : join(homedir(), '.dsh')
}

/** JSON-safe rendering for log payloads. */
function safeJson(value) {
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export class Logger {
  /**
   * @param {{ debug?: boolean, ctx?: object }} [options] - debug flag and plugin context.
   */
  constructor({ debug = false, ctx = undefined } = {}) {
    this.debugEnabled = debug === true
    this.dir = join(resolveDshHome(), 'logs', 'dcp')
    this.cordis = undefined
    if (ctx !== undefined) {
      try {
        const logger = ctx.logger?.('dsh-dcp')
        if (logger !== undefined && logger !== null) this.cordis = logger
      } catch {
        // logging must never break the plugin
      }
    }
    if (this.debugEnabled) {
      try {
        if (!existsSync(this.dir)) mkdirSync(this.dir, { recursive: true })
      } catch {
        // best effort
      }
    }
  }

  #write(level, message, extra) {
    if (!this.debugEnabled) return
    const line = extra !== undefined
      ? `[${new Date().toISOString()}] ${level} ${message} ${safeJson(extra)}`
      : `[${new Date().toISOString()}] ${level} ${message}`
    try {
      appendFileSync(join(this.dir, 'dcp.log'), line + '\n')
    } catch {
      // best effort
    }
  }

  debug(message, extra) {
    if (!this.debugEnabled) return
    this.#write('DEBUG', message, extra)
  }

  info(message, extra) {
    try {
      this.cordis?.info?.(message, ...(extra === undefined ? [] : [extra]))
    } catch {
      // best effort
    }
    this.#write('INFO', message, extra)
  }

  warn(message, extra) {
    try {
      this.cordis?.warn?.(message, ...(extra === undefined ? [] : [extra]))
    } catch {
      // best effort
    }
    this.#write('WARN', message, extra)
  }

  error(message, extra) {
    try {
      this.cordis?.error?.(message, ...(extra === undefined ? [] : [extra]))
    } catch {
      // best effort
    }
    this.#write('ERROR', message, extra)
  }

  /**
   * Persist a snapshot of the derived messages after a compression.
   * @param {string} sessionId - session identity.
   * @param {readonly object[]} messages - derived model-visible messages.
   */
  saveContext(sessionId, messages) {
    if (!this.debugEnabled) return
    try {
      const safe = String(sessionId).replace(/[^a-zA-Z0-9_-]/gu, '_').slice(0, 64) || 'session'
      appendFileSync(join(this.dir, `context-${safe}.jsonl`), safeJson(messages) + '\n')
    } catch {
      // best effort
    }
  }
}
