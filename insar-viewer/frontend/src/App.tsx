import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useCallback, useState, useRef } from 'react'
import { useProjectStore } from './stores/projectStore'
import { useViewerStore } from './stores/viewerStore'
import { api } from './api'
import { ProjectPicker } from './components/ProjectPicker/ProjectPicker'
import { ProjectSidebar } from './components/Layout/ProjectSidebar'
import { TopBar } from './components/Layout/TopBar'
import { SettingsModal } from './components/Layout/SettingsModal'
import { MapView } from './components/Map/MapView'
import { LayerPanel } from './components/Controls/LayerPanel'
import { DateSlider } from './components/Controls/DateSlider'
import { TimeSeriesPanel } from './components/TimeSeries/TimeSeriesPanel'

const queryClient = new QueryClient()

function AppShell() {
  const { project } = useProjectStore()
  const { resetLayers, setPointsData, setSelectedPixel, setPixelLoading, setCohThreshold, setDateIndex } = useViewerStore()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null)
  const mapWrapperRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!project) { setPointsData(null); setSelectedPixel(null); return }
    const defaults: Record<string, { enabled: boolean; opacity: number }> = {}
    for (const l of project.baseLayers) defaults[l.key] = { enabled: l.defaultEnabled, opacity: l.defaultOpacity }
    for (const l of project.dataLayers) defaults[l.key] = { enabled: l.defaultEnabled, opacity: l.defaultOpacity }
    resetLayers(defaults)
    setCohThreshold(project.cohThresholdDefault)
    setDateIndex(project.defaultDateIndex)
    setSelectedPixel(null)
    api.points().then(setPointsData).catch(console.error)
  }, [project]) // eslint-disable-line react-hooks/exhaustive-deps

  const handlePixelClick = useCallback(async (lat: number, lon: number) => {
    setPixelLoading(true)
    setSelectedPixel(null)
    try {
      const info = await api.pixel(lat, lon)
      setSelectedPixel(info)
    } catch (e) {
      console.error('Pixel fetch failed:', e)
    } finally {
      setPixelLoading(false)
    }
  }, [setSelectedPixel, setPixelLoading])

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setCtxMenu({ x: e.clientX, y: e.clientY })
  }, [])

  useEffect(() => {
    if (!ctxMenu) return
    const close = () => setCtxMenu(null)
    window.addEventListener('click', close, { once: true })
    return () => window.removeEventListener('click', close)
  }, [ctxMenu])

  const handleSaveImage = useCallback(async () => {
    setCtxMenu(null)
    if (!mapWrapperRef.current) return
    // Let context menu disappear from DOM before capture
    await new Promise(r => setTimeout(r, 60))
    const html2canvas = (await import('html2canvas')).default
    const canvas = await html2canvas(mapWrapperRef.current, {
      useCORS: true,
      allowTaint: false,
      backgroundColor: '#0d1a27',
      scale: window.devicePixelRatio || 1,
      logging: false,
      ignoreElements: (el) => el.classList.contains('leaflet-control-container'),
    })
    canvas.toBlob((blob) => {
      if (!blob) return
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `insar_${new Date().toISOString().slice(0, 10)}.png`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }, 'image/png')
  }, [])

  if (!project) {
    return (
      <div style={{ height: '100%', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
        <ProjectPicker />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', overflow: 'hidden' }}>
      <TopBar projectName={project.projectName} ncFile={project.ncFile} onOpenSettings={() => setSettingsOpen(true)} />

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <ProjectSidebar project={project} />

        {/* Map area — right-click to save PNG */}
        <div
          ref={mapWrapperRef}
          style={{ flex: 1, position: 'relative', overflow: 'hidden' }}
          onContextMenu={handleContextMenu}
        >
          <MapView project={project} onPixelClick={handlePixelClick} />
          <TimeSeriesPanel />
        </div>

        <LayerPanel project={project} />
      </div>

      <DateSlider dates={project.dates} />

      {/* Right-click context menu */}
      {ctxMenu && (
        <div style={{
          position: 'fixed', left: ctxMenu.x, top: ctxMenu.y,
          background: 'var(--panel)', border: '1px solid var(--border2)',
          borderRadius: 6, boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          zIndex: 9999, overflow: 'hidden', minWidth: 160,
        }}>
          <button
            onClick={handleSaveImage}
            style={{ width: '100%', padding: '8px 14px', background: 'transparent', border: 'none', color: 'var(--text)', cursor: 'pointer', fontSize: 12, textAlign: 'left', display: 'flex', alignItems: 'center', gap: 8 }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--panel2)' }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
          >
            &#128247; Save map as PNG
          </button>
        </div>
      )}

      {settingsOpen && <SettingsModal onClose={() => setSettingsOpen(false)} />}
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  )
}
