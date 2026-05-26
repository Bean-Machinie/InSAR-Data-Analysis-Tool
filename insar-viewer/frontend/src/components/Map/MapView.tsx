import { useEffect, useRef, useCallback } from 'react'
import {
  MapContainer,
  TileLayer,
  Polygon,
  Marker,
  Popup,
  ImageOverlay,
  ScaleControl,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import L from 'leaflet'
import type { ProjectInfo } from '../../types'
import { useViewerStore } from '../../stores/viewerStore'
import { api } from '../../api'
import { DataCanvas } from './DataCanvas'

// Fix Leaflet default icon paths broken by Vite bundling
import iconUrl from 'leaflet/dist/images/marker-icon.png'
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png'
import shadowUrl from 'leaflet/dist/images/marker-shadow.png'
L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl })

const POI_ICON = L.divIcon({
  className: '',
  html: '<div style="width:10px;height:10px;border-radius:50%;background:#00c896;border:2px solid rgba(0,200,150,0.5);box-shadow:0 0 4px rgba(0,200,150,0.6)"></div>',
  iconSize: [10, 10],
  iconAnchor: [5, 5],
})

// ── Click handler ─────────────────────────────────────────────────────────────
function ClickHandler({ onMapClick }: { onMapClick: (lat: number, lng: number) => void }) {
  useMapEvents({ click: (e) => onMapClick(e.latlng.lat, e.latlng.lng) })
  return null
}


// ── DataCanvas layer ──────────────────────────────────────────────────────────
// Separate component so it has access to useMap()
function DataCanvasLayer() {
  const map = useMap()
  const dcRef = useRef<DataCanvas | null>(null)
  const { layers, dateIndex, mode, cohThreshold, cohFilterEnabled, pointsData } = useViewerStore()

  // Create canvas once on mount
  useEffect(() => {
    if (!map.getPane('dataPane')) {
      map.createPane('dataPane')
      const dp = map.getPane('dataPane')!
      dp.style.zIndex = '420'
    }
    dcRef.current = new DataCanvas(map)
    return () => { dcRef.current = null }
  }, [map])

  // Feed new points data
  useEffect(() => {
    if (dcRef.current && pointsData) dcRef.current.setData(pointsData)
  }, [pointsData])

  // Re-render when display params change
  useEffect(() => {
    const dc = dcRef.current
    if (!dc || !pointsData) return

    const velMasked = layers['sbas_velocity_masked']?.enabled ?? false
    const velRaw    = layers['sbas_velocity_raw']?.enabled ?? false
    const disp      = layers['sbas_displacement_masked']?.enabled ?? false

    if (mode === 'date' && disp) {
      dc.render('disp', cohThreshold, dateIndex, layers['sbas_displacement_masked']?.opacity ?? 0.86)
    } else if (cohFilterEnabled) {
      dc.render('vel_coh', cohThreshold, 0, 0.87)
    } else if (velMasked) {
      dc.render('vel_masked', cohThreshold, 0, layers['sbas_velocity_masked']?.opacity ?? 0.87)
    } else if (velRaw) {
      dc.render('vel_raw', cohThreshold, 0, layers['sbas_velocity_raw']?.opacity ?? 0.82)
    } else {
      dc.clear()
    }
  }, [layers, dateIndex, mode, cohThreshold, cohFilterEnabled, pointsData])

  return null
}

// ── Selection ring ────────────────────────────────────────────────────────────
function SelectedPixelRing() {
  const map = useMap()
  const ringRef = useRef<L.Circle | null>(null)
  const { selectedPixel } = useViewerStore()

  useEffect(() => {
    if (!map.getPane('selPane')) {
      map.createPane('selPane')
      map.getPane('selPane')!.style.zIndex = '460'
    }
  }, [map])

  useEffect(() => {
    if (ringRef.current) { map.removeLayer(ringRef.current); ringRef.current = null }
    if (!selectedPixel || !selectedPixel.cellBounds) return

    const [[lat0, lon0], [lat1, lon1]] = selectedPixel.cellBounds
    const cellLatM = Math.abs(lat1 - lat0) * 111320
    const cellLonM = Math.abs(lon1 - lon0) * 111320 * Math.cos(selectedPixel.lat * Math.PI / 180)
    const r = Math.min(cellLatM, cellLonM) * 0.65

    ringRef.current = L.circle([selectedPixel.lat, selectedPixel.lon], {
      pane: 'selPane', radius: r,
      color: '#ffd740', weight: 2.5, opacity: 1, fill: false, interactive: false,
    }).addTo(map)
  }, [selectedPixel, map])

  return null
}

// ── PNG overlay layers ────────────────────────────────────────────────────────
function PngOverlays({ project }: { project: ProjectInfo }) {
  const { layers, dateIndex } = useViewerStore()

  return (
    <>
      {project.dataLayers
        .filter((l) => l.kind === 'png' && (layers[l.key]?.enabled ?? l.defaultEnabled))
        .map((layer) => {
          const idx = layer.temporal ? dateIndex : 0
          const url = api.overlayUrl(layer.key, idx)
          const opacity = layers[layer.key]?.opacity ?? layer.defaultOpacity
          return (
            <ImageOverlay
              key={`${layer.key}-${idx}`}
              url={url}
              bounds={project.bounds}
              opacity={opacity}
              pane="overlayPane"
            />
          )
        })}
    </>
  )
}

// ── Main MapView ──────────────────────────────────────────────────────────────
interface Props {
  project: ProjectInfo
  onPixelClick: (lat: number, lon: number) => void
}

export function MapView({ project, onPixelClick }: Props) {
  const { layers } = useViewerStore()

  const handleClick = useCallback(
    (lat: number, lng: number) => onPixelClick(lat, lng),
    [onPixelClick]
  )

  const aoiEnabled  = layers['aoi_original']?.enabled ?? true
  const aoiPositions = project.aoi as [number, number][] | null

  return (
    <MapContainer
      center={project.center}
      zoom={13}
      style={{ height: '100%', width: '100%', background: '#0d1a27' }}
      zoomControl
      preferCanvas
    >
      <ClickHandler onMapClick={handleClick} />
<DataCanvasLayer />
      <SelectedPixelRing />
      <ScaleControl imperial={false} />

      {/* Basemaps */}
      {project.baseLayers.map((bl) => {
        const enabled = layers[bl.key]?.enabled ?? bl.defaultEnabled
        const opacity = layers[bl.key]?.opacity ?? bl.defaultOpacity
        return enabled ? (
          <TileLayer
            key={bl.key}
            url={bl.url}
            attribution={bl.attribution}
            maxZoom={bl.maxZoom}
            maxNativeZoom={bl.maxZoom}
            opacity={opacity}
          />
        ) : null
      })}

      <PngOverlays project={project} />

      {/* AOI boundary */}
      {aoiEnabled && aoiPositions && (
        <Polygon
          positions={aoiPositions}
          pathOptions={{ color: '#29b6f6', weight: 2, opacity: 0.8, fillColor: '#29b6f6', fillOpacity: 0.04 }}
          interactive={false}
        />
      )}

      {/* POI markers */}
      {project.pois.map((poi) => (
        <Marker key={poi.name} position={[poi.lat, poi.lon]} icon={POI_ICON}>
          <Popup closeButton={false}>
            <div style={{ fontSize: 12, fontFamily: 'inherit' }}>
              <strong>{poi.name}</strong>
              <div style={{ fontSize: 10, marginTop: 2, color: 'var(--text2)' }}>
                {poi.lat.toFixed(5)}, {poi.lon.toFixed(5)}
              </div>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  )
}
