import { renderHook } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useWebSocket } from '../hooks/useWebSocket'

// Mock WebSocket globally
const mockWs = {
  close: vi.fn(),
  send: vi.fn(),
  onopen: null as ((ev: Event) => void) | null,
  onclose: null as ((ev: CloseEvent) => void) | null,
  onmessage: null as ((ev: MessageEvent) => void) | null,
  onerror: null as ((ev: Event) => void) | null,
  readyState: 0,
}

const MockWebSocket = vi.fn(() => mockWs)
// Attach OPEN constant so hooks can reference WebSocket.OPEN
Object.defineProperty(MockWebSocket, 'OPEN', { value: 1 })

beforeEach(() => {
  vi.stubGlobal('WebSocket', MockWebSocket)
  mockWs.close.mockClear()
  mockWs.send.mockClear()
  MockWebSocket.mockClear()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('useWebSocket', () => {
  it('returns initial disconnected state when no analysisId provided', () => {
    const { result } = renderHook(() => useWebSocket({}))
    expect(result.current.isConnected).toBe(false)
    expect(result.current.lastMessage).toBeNull()
  })

  it('does not create WebSocket when analysisId is undefined', () => {
    renderHook(() => useWebSocket({}))
    expect(MockWebSocket).not.toHaveBeenCalled()
  })

  it('does not create WebSocket when enabled is false', () => {
    renderHook(() => useWebSocket({ analysisId: 42, enabled: false, wsToken: 'tok' }))
    expect(MockWebSocket).not.toHaveBeenCalled()
  })

  it('creates WebSocket connection when analysisId and wsToken are provided', () => {
    renderHook(() => useWebSocket({ analysisId: 42, wsToken: 'test-token' }))
    expect(MockWebSocket).toHaveBeenCalled()
  })

  it('exposes unsubscribe function', () => {
    const { result } = renderHook(() => useWebSocket({}))
    expect(typeof result.current.unsubscribe).toBe('function')
  })
})
