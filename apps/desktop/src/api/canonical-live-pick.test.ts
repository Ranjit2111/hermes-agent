import { JsonRpcGatewayClient, JsonRpcGatewayError } from '@hermes/shared'
import { afterEach, expect, test, vi } from 'vitest'

import { HermesGateway } from './client'

afterEach(() => { vi.restoreAllMocks() })

async function attached(mutate: (params: Record<string, unknown>) => unknown) {
  vi.spyOn(JsonRpcGatewayClient.prototype, 'connect').mockResolvedValue()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  vi.spyOn(JsonRpcGatewayClient.prototype, 'request').mockImplementation(async (method, params) => {
    calls.push({ method, params: params as Record<string, unknown> })

    if (method === 'session.mutate') { return mutate(params as Record<string, unknown>) as never }

    return { session_id: 's', revision: 4, execution_generation: 2, messages: [], info: {} } as never
  })
  const client = new HermesGateway()
  await client.connect('ws://localhost/api/ws?native_dial=fixture&ticket=fixture')
  await client.request('session.resume', { session_id: 's' })
  calls.length = 0

  return { calls, client }
}

const pick = { session_id: 's', key: 'model', value: 'picked --provider custom' }

test('a composer pick is the session-scoped mutate; a busy session answers deferred for the next turn', async () => {
  const { calls, client } = await attached(params => ({
    session_id: 's', operation: 'model', status: 'deferred', model: (params.payload as { model: string }).model, revision: 4
  }))

  await expect(client.request('config.set', pick)).resolves.toMatchObject({ deferred: true, ok: true })
  expect(calls).toEqual([{ method: 'session.mutate', params: expect.objectContaining({
    operation: 'model', payload: { model: 'picked', provider: 'custom' }, expected_revision: 4, expected_generation: 2 }) }])
})

test('a fence refusal naming fresh fences is retried once at them, never looped', async () => {
  const stale = () => { throw new JsonRpcGatewayError('stale_generation', { code: 4001, data: { reason: 'stale_generation', revision: 6, execution_generation: 3 } }) }
  let attempts = 0

  const { calls, client } = await attached(params => {
    attempts += 1

    return attempts === 1 ? stale() : { session_id: 's', operation: 'model', model: 'picked', revision: (params.expected_revision as number) + 1 }
  })

  await expect(client.request('config.set', pick)).resolves.toMatchObject({ model: 'picked' })
  expect(calls.map(call => [call.params.expected_revision, call.params.expected_generation])).toEqual([[4, 2], [6, 3]])
  expect(calls[1].params.request_id).not.toBe(calls[0].params.request_id)

  attempts = 0
  calls.length = 0
  const { calls: looping, client: refused } = await attached(stale)
  await expect(refused.request('config.set', pick)).rejects.toThrow('stale_generation')
  expect(looping).toHaveLength(2)
})
