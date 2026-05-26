import type { AppSettings, PixelInfo, PointsData, ProjectInfo, RecentProject } from '../types'

const BASE = ''  // relative — proxied in dev, same-origin in prod

async function _get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`)
  if (!r.ok) {
    const msg = await r.text().catch(() => r.statusText)
    throw new Error(`GET ${path} → ${r.status}: ${msg}`)
  }
  return r.json() as Promise<T>
}

async function _post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const msg = await r.text().catch(() => r.statusText)
    throw new Error(`POST ${path} → ${r.status}: ${msg}`)
  }
  return r.json() as Promise<T>
}

export const api = {
  openProject: (path: string) => _post<ProjectInfo>('/api/project/open', { path }),
  projectInfo: () => _get<ProjectInfo>('/api/project/info'),
  closeProject: () => _post<{ status: string }>('/api/project/close', {}),

  pixel: (lat: number, lon: number) => _get<PixelInfo>(`/api/pixel?lat=${lat}&lon=${lon}`),
  points: () => _get<PointsData>('/api/points'),

  overlayUrl: (key: string, dateIndex: number) => `/overlay/${encodeURIComponent(key)}/${dateIndex}.png`,

  metadata: () => _get<Record<string, unknown>>('/api/metadata'),

  getSettings: () => _get<AppSettings>('/api/settings'),
  putSettings: (s: AppSettings) => _post<AppSettings>('/api/settings', s),

  recent: () => _get<RecentProject[]>('/api/recent'),
}
