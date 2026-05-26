import { useState, useEffect } from 'react'
import { api } from '../../api'
import { useProjectStore } from '../../stores/projectStore'
import type { RecentProject } from '../../types'

declare global {
  interface Window {
    pywebview?: {
      api?: {
        pick_folder?: () => Promise<string | null>
      }
    }
  }
}

export function ProjectPicker() {
  const { setProject, setLoading, setError, isLoading, error } = useProjectStore()
  const [recent, setRecent] = useState<RecentProject[]>([])
  const [manualPath, setManualPath] = useState('')

  useEffect(() => {
    api.recent().then(setRecent).catch(() => {})
  }, [])

  async function pickFolder() {
    let path: string | null = null

    if (window.pywebview?.api?.pick_folder) {
      path = await window.pywebview.api.pick_folder()
    } else {
      // Browser dev mode — use text input fallback
      path = window.prompt('Enter project folder path:')
    }

    if (!path) return
    await loadProject(path)
  }

  async function loadProject(path: string) {
    setLoading(true)
    setError(null)
    try {
      const info = await api.openProject(path)
      setProject(info)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="flex flex-col items-center justify-center h-full gap-8 px-8">
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight" style={{ color: 'var(--text)' }}>
          InSAR Deformation Viewer
        </h1>
        <p className="mt-2 text-sm" style={{ color: 'var(--text2)' }}>
          Open a project folder to begin exploring SBAS results
        </p>
      </div>

      <div className="flex flex-col items-center gap-3 w-full max-w-md">
        <button
          onClick={pickFolder}
          disabled={isLoading}
          className="w-full py-3 px-6 rounded-lg font-semibold text-sm transition-all"
          style={{
            background: 'var(--accent)',
            color: '#0b1825',
            opacity: isLoading ? 0.6 : 1,
            cursor: isLoading ? 'not-allowed' : 'pointer',
          }}
        >
          {isLoading ? 'Loading…' : 'Open Project Folder'}
        </button>

        {/* Dev-mode path input (hidden in pywebview) */}
        {!window.pywebview && (
          <div className="flex gap-2 w-full">
            <input
              type="text"
              value={manualPath}
              onChange={(e) => setManualPath(e.target.value)}
              placeholder="Or paste folder path here…"
              className="flex-1 px-3 py-2 rounded-lg text-sm"
              style={{
                background: 'var(--panel2)',
                border: '1px solid var(--border2)',
                color: 'var(--text)',
              }}
              onKeyDown={(e) => e.key === 'Enter' && manualPath && loadProject(manualPath)}
            />
            <button
              onClick={() => manualPath && loadProject(manualPath)}
              disabled={!manualPath || isLoading}
              className="px-4 py-2 rounded-lg text-sm font-semibold transition-all"
              style={{
                border: '1px solid var(--border2)',
                color: 'var(--text2)',
                background: 'transparent',
                cursor: !manualPath || isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              Load
            </button>
          </div>
        )}

        {error && (
          <div
            className="w-full px-4 py-3 rounded-lg text-sm"
            style={{
              background: 'rgba(239,83,80,0.1)',
              border: '1px solid rgba(239,83,80,0.3)',
              color: 'var(--danger)',
            }}
          >
            {error}
          </div>
        )}
      </div>

      {recent.length > 0 && (
        <div className="w-full max-w-md">
          <p className="text-xs font-bold uppercase tracking-wider mb-2" style={{ color: 'var(--text2)' }}>
            Recent Projects
          </p>
          <div
            className="rounded-lg overflow-hidden"
            style={{ border: '1px solid var(--border2)', background: 'var(--panel)' }}
          >
            {recent.map((r, i) => (
              <button
                key={r.path}
                onClick={() => loadProject(r.path)}
                disabled={isLoading}
                className="w-full text-left px-4 py-3 flex items-center justify-between transition-all hover:bg-blue-900/20"
                style={{
                  borderTop: i > 0 ? '1px solid var(--border)' : undefined,
                  cursor: isLoading ? 'not-allowed' : 'pointer',
                }}
              >
                <div>
                  <p className="font-semibold text-sm" style={{ color: 'var(--text)' }}>
                    {r.name}
                  </p>
                  <p className="text-xs mt-0.5 font-mono truncate max-w-80" style={{ color: 'var(--text2)' }}>
                    {r.path}
                  </p>
                </div>
                <span className="text-xs ml-4 shrink-0" style={{ color: 'var(--text2)' }}>
                  {new Date(r.last_opened).toLocaleDateString()}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
