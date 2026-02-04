/**
 * API Client for Local FastAPI Backend
 * This replaces tRPC when running locally with FastAPI
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  body?: any;
  headers?: Record<string, string>;
}

async function request<T>(endpoint: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, headers = {} } = options;
  
  const config: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      ...headers,
    },
    credentials: 'include', // Include cookies for session
  };
  
  if (body) {
    config.body = JSON.stringify(body);
  }
  
  const response = await fetch(`${API_BASE}${endpoint}`, config);
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

// Auth API
export const authApi = {
  me: () => request<any>('/auth/me'),
  login: (email: string, password?: string) => 
    request<any>('/auth/login', { method: 'POST', body: { email, password } }),
  logout: () => request<any>('/auth/logout', { method: 'POST' }),
  autoLogin: () => request<any>('/auth/auto-login', { method: 'POST' }),
};

// Videos API
export const videosApi = {
  list: () => request<any[]>('/videos'),
  get: (id: number) => request<any>(`/videos/${id}`),
  upload: (data: { title: string; description?: string; fileName: string; fileBase64: string; fileSize: number; mimeType: string }) =>
    request<any>('/videos/upload-base64', { method: 'POST', body: data }),
  delete: (id: number) => request<any>(`/videos/${id}`, { method: 'DELETE' }),
};

// Analysis API
export const analysisApi = {
  list: () => request<any[]>('/analysis'),
  get: (id: number) => request<any>(`/analysis/${id}`),
  byVideo: (videoId: number) => request<any[]>(`/analysis/by-video/${videoId}`),
  modes: () => request<any[]>('/analysis/modes'),
  stages: () => request<any[]>('/analysis/stages'),
  create: (data: { videoId: number; mode: string }) =>
    request<any>('/analysis', { method: 'POST', body: data }),
  updateStatus: (id: number, data: { status: string; progress: number; currentStage?: string; errorMessage?: string }) =>
    request<any>(`/analysis/${id}/status`, { method: 'PUT', body: data }),
  updateResults: (id: number, data: any) =>
    request<any>(`/analysis/${id}/results`, { method: 'PUT', body: data }),
  terminate: (id: number) =>
    request<any>(`/analysis/${id}/terminate`, { method: 'POST' }),
  eta: (id: number) => request<any>(`/analysis/${id}/eta`),
};

// Events API
export const eventsApi = {
  list: (analysisId: number) => request<any[]>(`/events/${analysisId}`),
  byType: (analysisId: number, type: string) => request<any[]>(`/events/${analysisId}/by-type/${type}`),
  create: (data: { analysisId: number; events: any[] }) =>
    request<any>('/events', { method: 'POST', body: data }),
};

// Tracks API
export const tracksApi = {
  list: (analysisId: number) => request<any[]>(`/tracks/${analysisId}`),
  atFrame: (analysisId: number, frame: number) => request<any>(`/tracks/${analysisId}/frame/${frame}`),
  create: (data: { analysisId: number; tracks: any[] }) =>
    request<any>('/tracks', { method: 'POST', body: data }),
};

// Statistics API
export const statisticsApi = {
  get: (analysisId: number) => request<any>(`/statistics/${analysisId}`),
  create: (data: any) => request<any>('/statistics', { method: 'POST', body: data }),
};

// WebSocket for real-time updates
export function createWebSocket(analysisId: number, onMessage: (data: any) => void): WebSocket {
  const wsUrl = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + `/ws/${analysisId}`;
  const ws = new WebSocket(wsUrl);
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WebSocket message parse error:', e);
    }
  };
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
  
  return ws;
}
