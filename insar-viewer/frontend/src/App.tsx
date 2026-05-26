import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useEffect, useCallback } from 'react'
import { useProjectStore } from './stores/projectStore'
import { useViewerStore } from './stores/viewerStore'
import { api } from './api'
import { ProjectPicker } from './components/ProjectPicker/ProjectPicker'
import { ProjectSidebar } from './components/Layout/ProjectSidebar'
import { TopBar } from './components/Layout/TopBar'
import { MapView } from './components/Map/MapView'
import { LayerPanel } from './components/Controls/LayerPanel'
import { DateSlider } from './components/Controls/DateSlider'
import { TimeSeriesPanel } from './components/TimeSeries/TimeSeriesPanel'

const queryClient = new QueryClient()

function AppShell() {
  const { project } = useProjectStore()
  const { resetLayers, setPointsData, setSelectedPixel, setCohThreshold, setDateIndex } = useViewerStore()

  // Initialise per-project state when a project opens (or closes)
  useEffect(() => {
    if (!project) {
      setPointsData(null)
      setSelectedPixel(null)
      return
    }

    // Build layer defaults from the project manifest
    const defaults: Record<string, { enabled: boolean; opacity: number }> = {}
    for (const l of project.baseLayers) defaults[l.key] = { enabled: l.defaultEnabled, opacity: l.defaultOpacity }
    for (const l of project.dataLayers) defaults[l.key] = { enabled: l.defaultEnabled, opacity: l.defaultOpacity }
    resetLayers(defaults)
    setCohThreshold(project.cohThresholdDefault)
    setDateIndex(project.defaultDateIndex)
    setSelectedPixel(null)

    // Load points payload for DataCanvas
    api.points().then(setPointsData).catch(console.error)
  }, [project]) // eslint-disable-line react-hooks/exhaustive-deps

  const handlePixelClick = useCallback(async (lat: number, lon: number) => {
    try {
      const info = await api.pixel(lat, lon)
      setSelectedPixel(info)
    } catch (e) {
      console.error('Pixel fetch failed:', e)
    }
  }, [setSelectedPixel])

  if (!project) {
    return (
      <div style={{ height: '100%', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg)' }}>
        <ProjectPicker />
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', overflow: 'hidden' }}>
      <TopBar projectName={project.projectName} ncFile={project.ncFile} />

      {/* Content row: sidebar + map + layer panel */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <ProjectSidebar project={project} />

        {/* Map area — position:relative so TimeSeriesPanel can overlay it */}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <MapView project={project} onPixelClick={handlePixelClick} />
          <TimeSeriesPanel />
        </div>

        {/* Right panel: layer list + coherence filter (rendered inside LayerPanel) */}
        <LayerPanel project={project} />
      </div>

      <DateSlider dates={project.dates} />
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
