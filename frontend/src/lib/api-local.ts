/**
 * API Client for FastAPI Backend
 * All endpoints are relative — the Vite dev proxy forwards /api to the FastAPI server.
 */

const API_BASE = '/api'

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  body?: any
  headers?: Record<string, string>
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {} } = options

  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    credentials: 'include',
  }

  if (body) {
    config.body = JSON.stringify(body)
  }

  const response = await fetch(`${API_BASE}${endpoint}`, config)

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// Auth API
export const authApi = {
  me: () => request<any>('/auth/me'),
  login: (email: string, password?: string) =>
    request<any>('/auth/login', { method: 'POST', body: { email, password } }),
  logout: () => request<any>('/auth/logout', { method: 'POST' }),
  autoLogin: () => request<any>('/auth/auto-login', { method: 'POST' }),
}

// Videos API
export const videosApi = {
  list: () => request<any[]>('/videos'),
  get: (id: number) => request<any>(`/videos/${id}`),
  delete: (id: number) => request<any>(`/videos/${id}`, { method: 'DELETE' }),
}

// Analysis API
export const analysisApi = {
  list: () => request<any[]>('/analysis'),
  get: (id: number) => request<any>(`/analysis/${id}`),
  byVideo: (videoId: number) => request<any[]>(`/analysis/by-video/${videoId}`),
  modes: () => request<any[]>('/analysis/modes'),
  stages: () => request<any[]>('/analysis/stages'),
  create: (data: {
    videoId: number
    mode: string
    fresh?: boolean
    cameraType?: string | null
    useCustomModels?: boolean | null
  }) => request<any>('/analysis', { method: 'POST', body: data }),
  updateStatus: (
    id: number,
    data: { status: string; progress: number; currentStage?: string; errorMessage?: string }
  ) => request<any>(`/analysis/${id}/status`, { method: 'PUT', body: data }),
  updateResults: (id: number, data: any) =>
    request<any>(`/analysis/${id}/results`, { method: 'PUT', body: data }),
  terminate: (id: number) => request<any>(`/analysis/${id}/terminate`, { method: 'POST' }),
  eta: (id: number) => request<any>(`/analysis/${id}/eta`),
}

// Events API
export const eventsApi = {
  list: (analysisId: number) => request<any[]>(`/events/${analysisId}`),
  byType: (analysisId: number, type: string) =>
    request<any[]>(`/events/${analysisId}/by-type/${type}`),
  create: (data: { analysisId: number; events: any[] }) =>
    request<any>('/events', { method: 'POST', body: data }),
}

// Tracks API
export const tracksApi = {
  list: (
    analysisId: number,
    params?: { offset?: number; limit?: number; frame_start?: number; frame_end?: number }
  ) => {
    const qs = new URLSearchParams()
    if (params?.offset != null) qs.set('offset', String(params.offset))
    if (params?.limit != null) qs.set('limit', String(params.limit))
    if (params?.frame_start != null) qs.set('frame_start', String(params.frame_start))
    if (params?.frame_end != null) qs.set('frame_end', String(params.frame_end))
    const q = qs.toString()
    return request<any[]>(`/tracks/${analysisId}${q ? '?' + q : ''}`)
  },
  atFrame: (analysisId: number, frame: number) =>
    request<any>(`/tracks/${analysisId}/frame/${frame}`),
  create: (data: { analysisId: number; tracks: any[] }) =>
    request<any>('/tracks', { method: 'POST', body: data }),
}

// Statistics API
export const statisticsApi = {
  get: (analysisId: number) => request<any>(`/statistics/${analysisId}`),
  create: (data: any) => request<any>('/statistics', { method: 'POST', body: data }),
}

// Commentary API
export const commentaryApi = {
  list: (analysisId: number) => request<any[]>(`/commentary/${analysisId}`),
  generate: (analysisId: number, data: { type: string; context?: any }) =>
    request<any>(`/commentary/${analysisId}`, { method: 'POST', body: data }),
  generateStream: async (
    analysisId: number,
    data: { type: string; context?: any },
    onChunk: (text: string) => void,
    onDone: (id: number | null, type: string) => void,
    onError: (message: string) => void
  ): Promise<void> => {
    const response = await fetch(`${API_BASE}/commentary/${analysisId}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify(data),
    })
    if (!response.ok || !response.body) {
      const err = await response.json().catch(() => ({ detail: 'Stream failed' }))
      onError(err.detail || `HTTP ${response.status}`)
      return
    }
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        try {
          const payload = JSON.parse(line.slice(6))
          if (payload.error) {
            onError(payload.error)
            return
          }
          if (payload.done) {
            onDone(payload.id ?? null, payload.type ?? data.type)
            return
          }
          if (payload.text) {
            onChunk(payload.text)
          }
        } catch {
          // malformed SSE line — skip
        }
      }
    }
  },
}

// Chat API
export const chatApi = {
  send: (analysisId: number, messages: Array<{ role: string; content: string }>) =>
    request<{ role: string; content: string }>(`/chat/${analysisId}`, {
      method: 'POST',
      body: { messages },
    }),
}

// WebSocket for real-time updates
export function createWebSocket(analysisId: number, onMessage: (data: any) => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/${analysisId}`
  const ws = new WebSocket(wsUrl)

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (e) {
      console.error('WebSocket message parse error:', e)
    }
  }

  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }

  return ws
}
