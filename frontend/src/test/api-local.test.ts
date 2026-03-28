import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock fetch before importing the module under test
const mockFetch = vi.fn()
beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch)
  mockFetch.mockResolvedValue({
    ok: true,
    json: async () => [],
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.clearAllMocks()
  vi.resetModules()
})

describe('analysisApi', () => {
  it('list() calls GET /api/analysis', async () => {
    const { analysisApi } = await import('../lib/api-local')
    await analysisApi.list()
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/analysis'),
      expect.objectContaining({ method: 'GET' })
    )
  })

  it('get(id) calls GET /api/analysis/{id}', async () => {
    const { analysisApi } = await import('../lib/api-local')
    await analysisApi.get(42)
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/analysis/42'),
      expect.anything()
    )
  })

  it('create() calls POST /api/analysis', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({ id: 99 }),
    })
    const { analysisApi } = await import('../lib/api-local')
    const result = await analysisApi.create({ videoId: 1, mode: 'all' })
    expect(result).toHaveProperty('id')
    const [url, opts] = mockFetch.mock.calls[0]
    expect(url).toContain('/api/analysis')
    expect(opts.method).toBe('POST')
  })
})

describe('videosApi', () => {
  it('list() calls GET /api/videos', async () => {
    const { videosApi } = await import('../lib/api-local')
    await videosApi.list()
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/videos'),
      expect.objectContaining({ method: 'GET' })
    )
  })
})
