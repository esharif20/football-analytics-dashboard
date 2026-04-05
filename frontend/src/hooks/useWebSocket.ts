/**
 * WebSocket hook for real-time analysis progress updates
 * Replaces polling with instant push notifications
 */
import { useEffect, useRef, useState, useCallback } from 'react'

export interface WSMessage {
  type: 'progress' | 'status' | 'complete' | 'error' | 'connected'
  analysisId?: number
  data?: {
    status?: string
    progress?: number
    currentStage?: string
    eta?: number
    error?: string
    result?: unknown
  }
}

export interface UseWebSocketOptions {
  analysisId?: number
  onProgress?: (data: WSMessage['data']) => void
  onComplete?: (data: WSMessage['data']) => void
  onError?: (error: string) => void
  enabled?: boolean
  wsToken?: string
}

export function useWebSocket(options: UseWebSocketOptions) {
  const { analysisId, onProgress, onComplete, onError, enabled = true, wsToken } = options

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const terminalRef = useRef(false)
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)

  const connect = useCallback(() => {
    const canConnect = enabled && !!analysisId && !!wsToken
    if (!canConnect || terminalRef.current || wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    // Cap reconnect attempts to prevent runaway retries
    if (reconnectAttemptsRef.current >= 10) {
      return
    }

    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const query = wsToken ? `?token=${encodeURIComponent(wsToken)}` : ''
    const wsUrl = `${protocol}//${window.location.host}/ws/${analysisId}${query}`

    try {
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        terminalRef.current = false

        // Subscribe to analysis updates if analysisId provided
        if (analysisId) {
          ws.send(
            JSON.stringify({
              type: 'subscribe',
              analysisId,
            })
          )
        }
      }

      ws.onmessage = (event) => {
        try {
          const message: WSMessage = JSON.parse(event.data)
          setLastMessage(message)

          // Only process messages for our analysis
          if (message.analysisId && message.analysisId !== analysisId) {
            return
          }

          switch (message.type) {
            case 'progress':
              onProgress?.(message.data)
              break
            case 'complete':
              terminalRef.current = true
              onComplete?.(message.data)
              break
            case 'error':
              terminalRef.current = true
              onError?.(message.data?.error || 'Unknown error')
              break
          }
        } catch (e) {
          console.error('WebSocket message parse error:', e)
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        wsRef.current = null

        // Attempt to reconnect after 3 seconds (unless terminal or max attempts)
        if (enabled && !terminalRef.current) {
          reconnectAttemptsRef.current += 1
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, 3000)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }
    } catch (e) {
      console.error('WebSocket connection error:', e)
    }
  }, [enabled, analysisId, onProgress, onComplete, onError, wsToken])

  // Connect on mount and when enabled changes
  useEffect(() => {
    if (enabled && analysisId && wsToken) {
      connect()
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [enabled, connect, analysisId, wsToken])

  // Subscribe to new analysis when analysisId changes
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && analysisId) {
      wsRef.current.send(
        JSON.stringify({
          type: 'subscribe',
          analysisId,
        })
      )
    }
  }, [analysisId])

  const unsubscribe = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && analysisId) {
      wsRef.current.send(
        JSON.stringify({
          type: 'unsubscribe',
          analysisId,
        })
      )
    }
  }, [analysisId])

  return {
    isConnected,
    lastMessage,
    unsubscribe,
  }
}
