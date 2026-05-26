export interface PoiEntry {
  name: string
  lon: number
  lat: number
}

export interface DataLayerInfo {
  key: string
  label: string
  kind: 'canvas' | 'png' | 'aoi'
  units: string
  temporal: boolean
  symmetric: boolean
  defaultEnabled: boolean
  defaultOpacity: number
  valueRange: [number, number] | null
  colormap: string
}

export interface BasemapInfo {
  key: string
  label: string
  url: string
  attribution: string
  defaultEnabled: boolean
  defaultOpacity: number
  maxZoom: number
}

export interface ProjectInfo {
  projectName: string
  orbit: string
  dateRange: { start: string; end: string }
  sceneCount: number
  dates: string[]
  defaultDateIndex: number
  pois: PoiEntry[]
  center: [number, number]
  bounds: [[number, number], [number, number]]
  aoi: [number, number][] | null
  baseLayers: BasemapInfo[]
  dataLayers: DataLayerInfo[]
  cohThresholdDefault: number
  cohSliderStep: number
  ncFile: string
}

export interface PixelSeries {
  raw: (number | null)[]
  segmented: (number | null)[]
  segment_id: number[]
  valid_time_mask: number[]
  coh_per_date: (number | null)[]
}

export interface PixelInfo {
  found: boolean
  reason?: string
  lat: number
  lon: number
  cellBounds?: [[number, number], [number, number]]
  dates: string[]
  velocity_mm_yr: number | null
  coherence_median: number | null
  valid_epoch_count: number
  total_epoch_count: number
  segment_count: number
  has_gap: boolean
  below_static_mask: boolean
  series: PixelSeries
}

export interface PointsData {
  count: number
  lats: number[]
  lons: number[]
  vel_raw: (number | null)[] | null
  vel_masked: (number | null)[] | null
  coherence: (number | null)[] | null
  disp: (number | null)[][] | null
  cellLat: number
  cellLon: number
}

export interface AppSettings {
  theme: string
  default_basemap: string
  default_colormap_velocity: string
  default_colormap_displacement: string
  units: string
}

export interface RecentProject {
  path: string
  name: string
  last_opened: string
}
