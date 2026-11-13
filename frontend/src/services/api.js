
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')

// Prefix backend routes when the React app is deployed separately from Flask.
// In local development, the empty base URL keeps using the Vite proxy.
export function apiUrl(path) {
  return `${API_BASE_URL}${path}`
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    credentials: 'include',
    ...options,
  })

  const contentType = response.headers.get('content-type') || ''
  const payload = contentType.includes('application/json')
    ? await response.json()
    : null

  if (!response.ok) {
    throw new Error(payload?.error || `Request failed (${response.status})`)
  }

  return payload
}

export const api = {
  session: () => request('/api/session'),
  health: () => request('/api/health'),

  upload: (file) => {
    const body = new FormData()
    body.append('file', file)
    return request('/api/upload', { method: 'POST', body })
  },

  cleanProfile: () => request('/api/clean/profile'),

  clean: (options) => request('/api/clean', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  }),

  eda: async () => {
    const result = await request('/api/eda')

    // Convert backend chart paths into absolute backend URLs.
    if (result?.charts) {
      result.charts = Object.fromEntries(
        Object.entries(result.charts).map(([key, value]) => [
          key,
          typeof value === 'string' && value.startsWith('/')
            ? apiUrl(value)
            : value,
        ])
      )
    }

    return result
  },

  mlOptions: () => request('/api/ml/options'),

  train: (options) => request('/api/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(options),
  }),

  results: () => request('/api/results'),
  progress: () => request('/api/progress'),
  reset: () => request('/api/new-analysis', { method: 'POST' }),
}
