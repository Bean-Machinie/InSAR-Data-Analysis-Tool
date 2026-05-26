import L from 'leaflet'
import type { PointsData } from '../../types'
import { valueToColor, computeRange } from '../../lib/colormap'

type CanvasMode = 'vel_masked' | 'vel_raw' | 'vel_coh' | 'disp'

/**
 * Custom Leaflet canvas layer that renders ~200k pixel circles.
 * Ported from the original Flask viewer's DataCanvas JS class.
 */
export class DataCanvas {
  private _map: L.Map
  private _el: HTMLCanvasElement
  private _ctx: CanvasRenderingContext2D
  private _pts: PointsData | null = null
  private _mode: CanvasMode | null = null
  private _thr = 0.30
  private _didx = 0
  private _opa = 0.87
  private _vmin = -10
  private _vmax = 10
  private _cellLat = 0.001
  private _cellLon = 0.001
  private _rafPending = false

  constructor(map: L.Map) {
    this._map = map
    this._el = document.createElement('canvas')
    this._el.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;'

    const pane = map.getPane('dataPane')
    if (pane) pane.appendChild(this._el)

    const ctx = this._el.getContext('2d')
    if (!ctx) throw new Error('Could not get 2D canvas context')
    this._ctx = ctx

    // Throttle re-draws through rAF so we never draw faster than display refresh
    const sched = () => {
      if (this._rafPending) return
      this._rafPending = true
      requestAnimationFrame(() => {
        this._rafPending = false
        this._draw()
      })
    }
    map.on('move zoom zoomend resize', sched)
  }

  setData(pts: PointsData): void {
    this._pts = pts
    this._cellLat = pts.cellLat
    this._cellLon = pts.cellLon
    if (this._mode) this._recompute()
  }

  render(mode: CanvasMode, thr: number, didx: number, opa: number): void {
    this._mode = mode
    this._thr = thr
    this._didx = didx
    this._opa = opa
    this._recompute()
  }

  clear(): void {
    this._mode = null
    this._clearCanvas()
  }

  getRange(): [number, number] {
    return [this._vmin, this._vmax]
  }

  getCount(): { count: number; total: number; pct: number } {
    const vals = this._vals()
    const coh = this._pts?.coherence
    if (!vals) return { count: 0, total: 0, pct: 0 }
    let count = 0, total = 0
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i]
      if (v === null || !isFinite(v)) continue
      total++
      if (this._mode === 'vel_coh' && coh && (coh[i] === null || (coh[i] as number) < this._thr)) continue
      count++
    }
    return { count, total, pct: total ? Math.round(count / total * 100) : 0 }
  }

  private _vals(): (number | null)[] | null {
    const d = this._pts
    if (!d) return null
    if (this._mode === 'vel_masked') return d.vel_masked
    if (this._mode === 'vel_raw')    return d.vel_raw
    if (this._mode === 'vel_coh')    return d.vel_raw
    if (this._mode === 'disp')       return d.disp ? d.disp[this._didx] : null
    return null
  }

  private _recompute(): void {
    const vals = this._vals()
    const coh = this._pts?.coherence
    if (!vals) { this._clearCanvas(); return }

    const visible: number[] = []
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i]
      if (v === null || !isFinite(v)) continue
      if (this._mode === 'vel_coh' && coh && (coh[i] === null || (coh[i] as number) < this._thr)) continue
      visible.push(v)
    }
    if (visible.length) {
      [this._vmin, this._vmax] = computeRange(visible)
    }
    this._draw()
  }

  private _clearCanvas(): void {
    const sz = this._map.getSize()
    this._el.width = sz.x
    this._el.height = sz.y
    this._ctx.clearRect(0, 0, sz.x, sz.y)
  }

  private _draw(): void {
    const sz = this._map.getSize()
    this._el.width = sz.x
    this._el.height = sz.y
    const ctx = this._ctx
    ctx.clearRect(0, 0, sz.x, sz.y)

    if (!this._mode || !this._pts) return
    const d = this._pts
    const vals = this._vals()
    if (!vals) return

    const coh = d.coherence
    const doCoh = this._mode === 'vel_coh'

    // The dataPane is CSS-translated during pan; subtract offset so canvas
    // coordinates stay locked to geography regardless of pan distance.
    const offset = (this._map as unknown as { _getMapPanePos(): L.Point })._getMapPanePos()

    // Radius: 48% of one cell in screen pixels
    const ctr = this._map.getCenter()
    const p0 = this._map.latLngToContainerPoint(L.latLng(ctr.lat, ctr.lng))
    const p1 = this._map.latLngToContainerPoint(
      L.latLng(ctr.lat + this._cellLat, ctr.lng + this._cellLon)
    )
    const cellPxH = Math.abs(p1.y - p0.y)
    const cellPxW = Math.abs(p1.x - p0.x)
    const r = Math.max(1.5, Math.min(cellPxH, cellPxW) * 0.48)

    ctx.globalAlpha = this._opa
    const vmin = this._vmin
    const vmax = this._vmax

    for (let i = 0; i < d.lats.length; i++) {
      const v = vals[i]
      if (v === null || !isFinite(v)) continue
      if (doCoh && (!coh || coh[i] === null || (coh[i] as number) < this._thr)) continue

      const color = valueToColor(v, vmin, vmax)
      if (!color) continue

      const pt = this._map.latLngToContainerPoint(L.latLng(d.lats[i], d.lons[i]))
      ctx.beginPath()
      ctx.arc(pt.x - offset.x, pt.y - offset.y, r, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    }
    ctx.globalAlpha = 1
  }
}
