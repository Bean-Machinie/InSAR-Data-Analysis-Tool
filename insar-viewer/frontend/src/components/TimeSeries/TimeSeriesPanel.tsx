import { useMemo } from 'react'
import _Plot from 'react-plotly.js'
import type { PixelInfo } from '../../types'

// react-plotly.js is CommonJS; Vite interop may wrap it — unwrap if needed
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Plot = (_Plot as any).default ?? _Plot
import { useViewerStore } from '../../stores/viewerStore'

const SEG_COLORS = ['#29b6f6', '#00c896', '#ffb74d', '#ef5350', '#ab47bc', '#26c6da', '#d4e157']

export function TimeSeriesPanel() {
  const { selectedPixel, setSelectedPixel } = useViewerStore()
  if (!selectedPixel) return null

  return (
    <div style={{
      position: 'absolute', bottom: 0, left: 0, right: 0, height: 340,
      background: 'var(--panel)',
      borderTop: '1px solid var(--border2)',
      display: 'flex', flexDirection: 'column',
      zIndex: 600,
    }}>
      <PanelHeader pixel={selectedPixel} onClose={() => setSelectedPixel(null)} />
      {selectedPixel.found
        ? <PanelBody pixel={selectedPixel} />
        : <NotFound pixel={selectedPixel} />}
    </div>
  )
}

function PanelHeader({ pixel, onClose }: { pixel: PixelInfo; onClose: () => void }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', padding: '5px 12px',
      borderBottom: '1px solid var(--border)', flexShrink: 0, gap: 10,
    }}>
      <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text2)', flex: 1 }}>
        Pixel Time Series
      </span>
      {pixel.has_gap && (
        <span style={{ fontSize: 10, color: 'var(--warn)' }}>
          &#9888; Gap detected in segmentation
        </span>
      )}
      <button
        onClick={onClose}
        style={{
          background: 'transparent', border: 'none', color: 'var(--text2)',
          cursor: 'pointer', fontSize: 18, lineHeight: 1, padding: '0 2px',
          display: 'flex', alignItems: 'center',
        }}
        aria-label="Close panel"
      >
        &#xd7;
      </button>
    </div>
  )
}

function NotFound({ pixel }: { pixel: PixelInfo }) {
  return (
    <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontSize: 12 }}>
      {pixel.below_static_mask
        ? 'Pixel is outside the static mask (low coherence region).'
        : (pixel.reason ?? 'No data at this location.')}
    </div>
  )
}

function PanelBody({ pixel }: { pixel: PixelInfo }) {
  return (
    <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
      <StatCards pixel={pixel} />
      <TimeSeriesChart pixel={pixel} />
    </div>
  )
}

function StatCards({ pixel }: { pixel: PixelInfo }) {
  const vel = pixel.velocity_mm_yr
  const coh = pixel.coherence_median
  const velColor = vel == null ? 'var(--text)' : vel > 2 ? '#ef9a9a' : vel < -2 ? '#90caf9' : 'var(--accent2)'

  const rows = [
    { label: 'Lat', value: pixel.lat.toFixed(5) },
    { label: 'Lon', value: pixel.lon.toFixed(5) },
    { label: 'Velocity', value: vel != null ? `${vel.toFixed(1)} mm/yr` : '—', color: velColor },
    { label: 'Coherence', value: coh != null ? coh.toFixed(3) : '—' },
    { label: 'Valid epochs', value: `${pixel.valid_epoch_count} / ${pixel.total_epoch_count}` },
    { label: 'Segments', value: `${pixel.segment_count}` },
  ]

  return (
    <div style={{
      width: 136, flexShrink: 0, padding: '10px 12px',
      borderRight: '1px solid var(--border)', overflowY: 'auto',
    }}>
      {rows.map((r) => (
        <div key={r.label} style={{ marginBottom: 11 }}>
          <div style={{ fontSize: 9, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text2)' }}>
            {r.label}
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, color: r.color ?? 'var(--text)', marginTop: 1, fontVariantNumeric: 'tabular-nums' }}>
            {r.value}
          </div>
        </div>
      ))}
    </div>
  )
}

function TimeSeriesChart({ pixel }: { pixel: PixelInfo }) {
  const { dates, series } = pixel

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data = useMemo<any[]>(() => {
    const traces = []

    // Coherence bar (secondary y-axis, behind other traces)
    traces.push({
      type: 'bar',
      x: dates,
      y: series.coh_per_date,
      yaxis: 'y2',
      name: 'Coherence',
      marker: { color: 'rgba(106,145,174,0.22)' },
      hovertemplate: '%{x}: %{y:.3f}<extra>Coherence</extra>',
      showlegend: true,
    })

    // Raw displacement (all epochs, grey dotted)
    traces.push({
      type: 'scatter',
      mode: 'lines+markers',
      x: dates,
      y: series.raw,
      name: 'Raw',
      line: { color: 'rgba(180,210,240,0.28)', width: 1, dash: 'dot' },
      marker: { color: 'rgba(180,210,240,0.28)', size: 3 },
      hovertemplate: '%{x}: %{y:.1f} mm<extra>Raw</extra>',
    })

    // Dropped epochs (red ×)
    const dropX: string[] = []
    const dropY: (number | null)[] = []
    for (let i = 0; i < dates.length; i++) {
      if (series.valid_time_mask[i] === 0) {
        dropX.push(dates[i])
        dropY.push(series.raw[i])
      }
    }
    if (dropX.length) {
      traces.push({
        type: 'scatter',
        mode: 'markers',
        x: dropX,
        y: dropY,
        name: 'Dropped',
        marker: { symbol: 'x', color: '#ef5350', size: 8, line: { color: '#ef5350', width: 2 } },
        hovertemplate: '%{x}<extra>Dropped</extra>',
      })
    }

    // Segmented traces, one per unique segment_id
    const uniqueSegs = [...new Set(series.segment_id)]
    for (const segId of uniqueSegs) {
      const color = SEG_COLORS[segId % SEG_COLORS.length]
      const segX: string[] = []
      const segY: (number | null)[] = []
      for (let i = 0; i < dates.length; i++) {
        if (series.segment_id[i] === segId && series.valid_time_mask[i] === 1 && series.segmented[i] != null) {
          segX.push(dates[i])
          segY.push(series.segmented[i])
        }
      }
      if (segX.length) {
        traces.push({
          type: 'scatter',
          mode: 'lines+markers',
          x: segX,
          y: segY,
          name: uniqueSegs.length > 1 ? `Seg ${segId}` : 'Segmented',
          line: { color, width: 2 },
          marker: { color, size: 5 },
          hovertemplate: `%{x}: %{y:.1f} mm<extra>${uniqueSegs.length > 1 ? `Seg ${segId}` : 'Segmented'}</extra>`,
        })
      }
    }

    return traces
  }, [pixel]) // eslint-disable-line react-hooks/exhaustive-deps

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const layout = useMemo<any>(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#cce0f0', family: 'Inter,Segoe UI,system-ui,sans-serif', size: 11 },
    margin: { l: 52, r: 52, t: 6, b: 46 },
    xaxis: {
      showgrid: true,
      gridcolor: 'rgba(255,255,255,0.05)',
      tickfont: { size: 9 },
      tickangle: -40,
      zeroline: false,
      linecolor: 'rgba(255,255,255,0.1)',
    },
    yaxis: {
      title: { text: 'Displacement (mm)', font: { size: 9 }, standoff: 4 },
      showgrid: true,
      gridcolor: 'rgba(255,255,255,0.07)',
      zeroline: true,
      zerolinecolor: 'rgba(255,255,255,0.18)',
      zerolinewidth: 1,
      tickfont: { size: 9 },
    },
    yaxis2: {
      title: { text: 'Coh', font: { size: 9 }, standoff: 2 },
      overlaying: 'y',
      side: 'right',
      range: [0, 1.05],
      showgrid: false,
      tickfont: { size: 9 },
    },
    legend: {
      bgcolor: 'rgba(0,0,0,0)',
      font: { size: 9 },
      x: 0,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      orientation: 'h',
    },
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#0f2033',
      bordercolor: 'rgba(255,255,255,0.15)',
      font: { size: 10, color: '#cce0f0' },
    },
  }), [])

  return (
    <Plot
      data={data}
      layout={layout}
      style={{ flex: 1, width: '100%', height: '100%' }}
      useResizeHandler
      config={{ displayModeBar: false, scrollZoom: false, responsive: true }}
    />
  )
}
