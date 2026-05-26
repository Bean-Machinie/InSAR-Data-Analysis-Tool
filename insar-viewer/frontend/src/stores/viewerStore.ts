import { create } from 'zustand'
import type { PixelInfo, PointsData } from '../types'

interface LayerState {
  enabled: boolean
  opacity: number
}

interface ViewerStore {
  dateIndex: number
  mode: 'velocity' | 'date'
  cohThreshold: number
  cohFilterEnabled: boolean
  layers: Record<string, LayerState>
  selectedPixel: PixelInfo | null
  pixelLoading: boolean
  pointsData: PointsData | null
  setDateIndex: (i: number) => void
  setMode: (m: 'velocity' | 'date') => void
  setCohThreshold: (v: number) => void
  setCohFilterEnabled: (v: boolean) => void
  setLayerEnabled: (key: string, v: boolean) => void
  setLayerOpacity: (key: string, v: number) => void
  setSelectedPixel: (p: PixelInfo | null) => void
  setPixelLoading: (v: boolean) => void
  setPointsData: (d: PointsData | null) => void
  resetLayers: (defaults: Record<string, LayerState>) => void
}

export const useViewerStore = create<ViewerStore>((set) => ({
  dateIndex: 0,
  mode: 'velocity',
  cohThreshold: 0.3,
  cohFilterEnabled: false,
  layers: {},
  selectedPixel: null,
  pixelLoading: false,
  pointsData: null,
  setDateIndex: (i) => set({ dateIndex: i }),
  setMode: (m) => set({ mode: m }),
  setCohThreshold: (v) => set({ cohThreshold: v }),
  setCohFilterEnabled: (v) => set({ cohFilterEnabled: v }),
  setLayerEnabled: (key, v) =>
    set((s) => ({ layers: { ...s.layers, [key]: { ...s.layers[key], enabled: v } } })),
  setLayerOpacity: (key, v) =>
    set((s) => ({ layers: { ...s.layers, [key]: { ...s.layers[key], opacity: v } } })),
  setSelectedPixel: (p) => set({ selectedPixel: p }),
  setPixelLoading: (v) => set({ pixelLoading: v }),
  setPointsData: (d) => set({ pointsData: d }),
  resetLayers: (defaults) => set({ layers: defaults }),
}))
