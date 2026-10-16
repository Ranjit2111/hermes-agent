import { afterEach, describe, expect, it, vi } from 'vitest'

import { $currentModel, $currentProvider, setCurrentModelTransient, setCurrentProviderTransient } from '@/store/session'

import { handleDeferredPickEvent } from './deferred-pick'
import type { GatewayEventContext } from './types'

const notifyError = vi.hoisted(() => vi.fn())
const surfaceModelSwitchConfirm = vi.hoisted(() => vi.fn(async () => true))

vi.mock('@/store/notifications', async importOriginal => ({ ...(await importOriginal<object>()), notifyError }))
vi.mock('@/lib/guarded-model-switch', () => ({ surfaceModelSwitchConfirm }))

function context(payload: Record<string, unknown>) {
  const states: Record<string, Record<string, unknown>> = { s: { model: 'picked', provider: 'custom' } }

  const ctx = {
    deps: {
      activeGatewayProfile: 'default',
      queryClient: { invalidateQueries: vi.fn(async () => undefined) },
      updateSessionState: (sid: string, updater: (state: never) => never) => (states[sid] = updater(states[sid] as never))
    },
    event: { payload, session_id: 's', type: 'session.updated' },
    explicitSid: 's',
    fromActiveSource: () => true,
    isActiveEvent: true,
    occurredAt: 0,
    payload,
    scheduleConfigRefresh: vi.fn(),
    sessionId: 's'
  } as unknown as GatewayEventContext

  return { ctx, states }
}

describe('handleDeferredPickEvent', () => {
  afterEach(() => { vi.clearAllMocks() })

  it('shows a refused next-turn pick by name and repaints the committed model', () => {
    setCurrentModelTransient('picked')
    setCurrentProviderTransient('custom')

    const { ctx, states } = context({ model: 'frozen', operation: 'model', provider: 'openrouter',
      reason: 'model_resolution_failed', session_id: 's', status: 'refused', target_model: 'picked' })

    expect(handleDeferredPickEvent(ctx)).toBe(true)
    expect(notifyError).toHaveBeenCalledWith(expect.objectContaining({ message: 'model_resolution_failed' }), 'Model switch failed')
    expect(states.s).toMatchObject({ model: 'frozen', provider: 'openrouter' })
    expect([$currentModel.get(), $currentProvider.get()]).toEqual(['frozen', 'openrouter'])
  })

  it('asks for a guarded next-turn pick instead of dropping it', () => {
    const { ctx } = context({ confirm: 'token', confirm_message: 'Costly', operation: 'model',
      payload: { model: 'guarded', provider: 'custom' }, session_id: 's', status: 'confirmation_required', target_model: 'guarded' })

    expect(handleDeferredPickEvent(ctx)).toBe(true)
    expect(surfaceModelSwitchConfirm).toHaveBeenCalledWith(expect.objectContaining({ confirmMessage: 'Costly', model: 'guarded' }))
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('leaves other session updates to their handlers', () => {
    expect(handleDeferredPickEvent(context({ operation: 'rename', session_id: 's', title: 'x' }).ctx)).toBe(false)
    expect(handleDeferredPickEvent(context({ model: 'm', operation: 'model', revision: 3, session_id: 's' }).ctx)).toBe(false)
  })
})
