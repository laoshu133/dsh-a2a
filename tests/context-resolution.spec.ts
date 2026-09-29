/**
 * What one inbound message's `contextId` actually resolves to.
 *
 * The interesting cases are exactly the ones that miss the resident registry: a
 * peer that presents no id, a peer presenting an id this process has forgotten,
 * and a peer presenting someone else's. All three answer with a usable task, so
 * the response alone cannot tell an operator which one happened — which is why
 * the router reports an audit row, and why these tests read it.
 *
 * The router is driven directly against stubbed deps. It is deliberately
 * Cordis-free, so pinning the policy that decides whose conversation a message
 * lands in needs no harness, no agent, and no listening socket.
 */

import type { IncomingMessage, ServerResponse } from 'node:http'
import { describe, expect, it } from 'vitest'
import { A2AContextId, type A2ATask, type A2ATaskId } from '../src/protocol/index.ts'
import { ContextRegistry, type Activation, type ContextAuditRow } from '../src/contexts.ts'
import { RateLimiter, TurnTracker } from '../src/security.ts'
import { createRouter, type Router, type RouterDeps } from '../src/router.ts'
import type { A2AServerConfig } from '../src/config.ts'
import type { TaskSlot } from '../src/tasks.ts'

const ALICE_TOKEN = 'tok-alice'
const BOB_TOKEN = 'tok-bob'

/** The one thing a test cares about: what the peer was answered. */
interface Answer {
  status: number
  body: Record<string, unknown>
}

/** Build one `SendMessage` request as `serveRpc` will read it. */
function request(token: string, message: Record<string, unknown>): {
  req: IncomingMessage
  res: ServerResponse
  answer: Answer
} {
  const answer: Answer = { status: 0, body: {} }
  const payload = JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'SendMessage',
    params: { message },
  })
  const req = {
    method: 'POST',
    headers: { host: 'localhost:9900', authorization: `Bearer ${token}` },
    async *[Symbol.asyncIterator](): AsyncGenerator<Buffer> {
      yield Buffer.from(payload)
    },
  }
  const res = {
    writeHead(status: number): void {
      answer.status = status
    },
    end(text: string): void {
      answer.body = JSON.parse(text) as Record<string, unknown>
    },
  }
  return {
    req: req as unknown as IncomingMessage,
    res: res as unknown as ServerResponse,
    answer,
  }
}

/** The contextId the peer was told its message runs in. */
function resolvedContextId(body: Record<string, unknown>): string {
  const result = body['result'] as { task: { contextId: string } }
  return result.task.contextId
}

interface Harness {
  router: Router
  contexts: ContextRegistry
  /** Every audit row the router reported, in order. */
  audit: ContextAuditRow[]
  /** Contexts the stubbed activation factory was asked to create. */
  created: string[]
  /** ContextIds the stubbed durable store was asked to resume, in order. */
  askedToResume: string[]
  warns: string[]
}

/**
 * Build the router under test.
 * @param resumable - contextIds the stubbed durable store can resume, mapped to
 *   the peer its log records. The store refuses any other peer, exactly as the
 *   real one does when ownership cannot be proven.
 */
function harness(resumable: ReadonlyMap<string, string> = new Map()): Harness {
  const audit: ContextAuditRow[] = []
  const created: string[] = []
  const askedToResume: string[] = []
  const warns: string[] = []
  const contexts = new ContextRegistry(64, 1_800_000)

  const activationOf = (contextId: string, peer: string): Activation => ({
    contextId: A2AContextId(contextId),
    agent: {} as Activation['agent'],
    handle: { dispose: async () => {} } as unknown as Activation['handle'],
    peer,
    cwd: '/tmp/workspace',
    slots: new Map(),
    lastTouchedAt: Date.now(),
  })

  const config: A2AServerConfig = {
    basePath: '/a2a',
    card: { name: 'test', description: '', public: true, skills: [], extendedSkills: [] },
    peers: { alice: { tokenEnv: 'A2A_PEER_ALICE' }, bob: { tokenEnv: 'A2A_PEER_BOB' } },
    rateLimitPerMinute: 600,
    maxContextTurns: 5,
    // Immediate, so the send path never awaits a settlement this stub has none of.
    sendMode: 'immediate',
    blockTimeoutMs: 1_000,
    contextIdleTtlMs: 1_800_000,
    maxResidentContexts: 64,
    contextAuditPath: '',
    isolation: { workspaceMode: 'per-peer', workspaceRoot: '/tmp/workspaces', peerWorkspaces: {} },
    push: { enabled: false },
  }

  const secrets = new Map([['alice', ALICE_TOKEN], ['bob', BOB_TOKEN]])
  const deps: RouterDeps = {
    config,
    contexts,
    turns: new TurnTracker(),
    rateLimiter: new RateLimiter(600),
    logger: {
      warn: (message: string) => void warns.push(message),
      error: (message: string) => void warns.push(message),
    },
    assertOpen: () => {},
    resolvePeerSecrets: async () => secrets,
    identify: (token, known) => token === undefined
      ? undefined
      : [...known.entries()].find(([, secret]) => secret === token)?.[0],
    cardFor: () => ({ name: 'test' }) as ReturnType<RouterDeps['cardFor']>,
    extendedCardFor: () => undefined,
    createActivation: async (peer) => {
      const contextId = `created-${created.length + 1}`
      created.push(contextId)
      const activation = activationOf(contextId, peer)
      contexts.add(activation)
      return activation
    },
    resumeActivation: async (rawContextId, peer) => {
      askedToResume.push(rawContextId)
      if (resumable.get(rawContextId) !== peer) return undefined
      const activation = activationOf(rawContextId, peer)
      contexts.add(activation)
      return activation
    },
    audit: (row) => void audit.push(row),
    submit: async (activation) => ({
      taskId: `task-of-${activation.contextId}` as A2ATaskId,
      settled: Promise.resolve(),
    }) as unknown as TaskSlot,
    cancel: () => {},
    taskSnapshot: (activation, _slot, state) => ({
      id: `task-of-${activation.contextId}`,
      contextId: activation.contextId,
      status: { state },
    }) as unknown as A2ATask,
  }

  return { router: createRouter(deps), contexts, audit, created, askedToResume, warns }
}

/** Send one message and hand back the peer's answer. */
async function send(
  h: Harness,
  token: string,
  message: Record<string, unknown>,
): Promise<Answer> {
  const { req, res, answer } = request(token, message)
  await h.router.serveRpc(req, res)
  return answer
}

/** The A2A v1.0 user message every test sends unless it says otherwise. */
function message(text: string, contextId?: string): Record<string, unknown> {
  return {
    messageId: `m-${text}`,
    role: 'ROLE_USER',
    parts: [{ text, mediaType: 'text/plain' }],
    ...contextId === undefined ? {} : { contextId },
  }
}

describe('context resolution', () => {
  it('starts a context when the peer presents none, and says so', async () => {
    const h = harness()

    const answer = await send(h, ALICE_TOKEN, message('hello'))

    expect(resolvedContextId(answer.body)).toBe('created-1')
    expect(h.created).toEqual(['created-1'])
    expect(h.askedToResume).toEqual([])
    expect(h.audit).toEqual([
      { peer: 'alice', presented: false, outcome: 'created', newContextId: 'created-1' },
    ])
  })

  it('continues a resident context without creating anything', async () => {
    const h = harness()
    h.contexts.add({
      contextId: A2AContextId('resident-1'),
      agent: {} as Activation['agent'],
      handle: {} as Activation['handle'],
      peer: 'alice',
      cwd: '/tmp/workspace',
      slots: new Map(),
      lastTouchedAt: Date.now(),
    })

    const answer = await send(h, ALICE_TOKEN, message('again', 'resident-1'))

    expect(resolvedContextId(answer.body)).toBe('resident-1')
    expect(h.created).toEqual([])
    // A resident hit never consults the durable store: nothing was forgotten.
    expect(h.askedToResume).toEqual([])
    expect(h.audit).toEqual([
      { peer: 'alice', presented: true, contextId: 'resident-1', outcome: 'resident' },
    ])
  })

  it('resumes the durable Session behind a forgotten contextId', async () => {
    const h = harness(new Map([['forgotten-1', 'alice']]))

    const answer = await send(h, ALICE_TOKEN, message('still me', 'forgotten-1'))

    // The whole point: the peer keeps the conversation it already had, rather
    // than being quietly handed a new one that merely looks the same.
    expect(resolvedContextId(answer.body)).toBe('forgotten-1')
    expect(h.askedToResume).toEqual(['forgotten-1'])
    expect(h.created).toEqual([])
    expect(h.audit).toEqual([
      { peer: 'alice', presented: true, contextId: 'forgotten-1', outcome: 'resumed' },
    ])
  })

  it('starts a fresh context when the conversation is genuinely gone', async () => {
    const h = harness()

    const answer = await send(h, ALICE_TOKEN, message('anyone there', 'vanished-1'))

    expect(resolvedContextId(answer.body)).toBe('created-1')
    expect(h.askedToResume).toEqual(['vanished-1'])
    expect(h.audit).toEqual([{
      peer: 'alice',
      presented: true,
      contextId: 'vanished-1',
      outcome: 'created-unresumable',
      newContextId: 'created-1',
    }])
    // Loud, because this is the outcome an operator must be able to see: the
    // peer cannot tell the replacement from a continuation.
    expect(h.warns).toHaveLength(1)
    expect(h.warns[0]).toContain('vanished-1')
  })

  it("answers another peer's contextId as absent and never resumes it", async () => {
    const h = harness(new Map([['alices-context', 'alice']]))
    h.contexts.add({
      contextId: A2AContextId('alices-context'),
      agent: {} as Activation['agent'],
      handle: {} as Activation['handle'],
      peer: 'alice',
      cwd: '/tmp/workspace',
      slots: new Map(),
      lastTouchedAt: Date.now(),
    })

    const answer = await send(h, BOB_TOKEN, message('let me in', 'alices-context'))

    expect(resolvedContextId(answer.body)).toBe('created-1')
    expect(h.askedToResume).toEqual([])
    expect(h.audit).toEqual([{
      peer: 'bob',
      presented: true,
      contextId: 'alices-context',
      outcome: 'created-foreign',
      newContextId: 'created-1',
    }])
  })

  it('falls back when the durable log does not prove this peer owns the context', async () => {
    // The registry is process-local, so another peer's context is usually NOT
    // resident when it is presented: the lookup cannot tell "never existed" from
    // "belongs to someone else", and the store is asked. Ownership is settled
    // there — from the context's OWN log — and a refusal must land on the same
    // fresh-context path as a context that is genuinely gone, never on the
    // presented id.
    const h = harness(new Map([['alices-elsewhere', 'alice']]))

    const answer = await send(h, BOB_TOKEN, message('let me in', 'alices-elsewhere'))

    expect(resolvedContextId(answer.body)).toBe('created-1')
    expect(h.askedToResume).toEqual(['alices-elsewhere'])
    expect(h.audit).toEqual([{
      peer: 'bob',
      presented: true,
      contextId: 'alices-elsewhere',
      outcome: 'created-unresumable',
      newContextId: 'created-1',
    }])
  })
})
