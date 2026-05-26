import { useViewerStore } from '../../stores/viewerStore'
import type { PointsData } from '../../types'

interface Props {
  cohDefault: number
  cohStep: number
}

export function CoherenceSlider({ cohDefault: _cohDefault, cohStep }: Props) {
  const { cohThreshold, cohFilterEnabled, setCohThreshold, setCohFilterEnabled, pointsData } = useViewerStore()

  const stats = pointsData ? computeStats(pointsData, cohThreshold) : null

  return (
    <div style={{ borderTop: '1px solid var(--border2)', padding: 12, flexShrink: 0 }}>
      <p style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text2)', marginBottom: 10 }}>
        Coherence Filter
      </p>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <span style={{ fontSize: 12, color: 'var(--text)' }}>Apply threshold filter</span>
        <ToggleSwitch checked={cohFilterEnabled} onChange={setCohFilterEnabled} />
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text2)', marginBottom: 4 }}>
        <span>0.10</span><span>0.90</span>
      </div>
      <input
        type="range" min={10} max={90} step={Math.round(cohStep * 100)}
        value={Math.round(cohThreshold * 100)}
        onChange={(e) => setCohThreshold(Number(e.target.value) / 100)}
        style={{ width: '100%', accentColor: 'var(--accent)', marginBottom: 6 }}
      />
      <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--accent)', textAlign: 'center', lineHeight: 1, marginBottom: 4 }}>
        {cohThreshold.toFixed(2)}
      </div>
      <div style={{ fontSize: 11, color: 'var(--text2)', textAlign: 'center', minHeight: 14 }}>
        {stats
          ? `${stats.count.toLocaleString()} px visible (${stats.pct}% of total)`
          : <Spinner />}
      </div>
    </div>
  )
}

function computeStats(pts: PointsData, thr: number) {
  const vals = pts.vel_masked ?? pts.vel_raw
  const coh = pts.coherence
  if (!vals) return null
  let count = 0, total = 0
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i]
    if (v === null || !isFinite(v)) continue
    total++
    if (!coh || coh[i] === null || (coh[i] as number) >= thr) count++
  }
  return { count, total, pct: total ? Math.round(count / total * 100) : 0 }
}

function ToggleSwitch({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label style={{ position: 'relative', display: 'inline-block', width: 34, height: 18, cursor: 'pointer' }}>
      <input
        type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)}
        style={{ opacity: 0, width: 0, height: 0 }}
      />
      <span style={{
        position: 'absolute', inset: 0,
        background: checked ? 'rgba(41,182,246,0.2)' : 'var(--panel2)',
        border: `1px solid ${checked ? 'var(--accent)' : 'var(--border2)'}`,
        borderRadius: 18,
        transition: 'background 0.15s, border-color 0.15s',
      }}>
        <span style={{
          position: 'absolute', left: checked ? 16 : 2, top: 2,
          width: 12, height: 12, borderRadius: '50%',
          background: checked ? 'var(--accent)' : 'var(--text2)',
          transition: 'left 0.15s, background 0.15s',
        }} />
      </span>
    </label>
  )
}

function Spinner() {
  return (
    <span style={{
      display: 'inline-block', width: 12, height: 12,
      border: '2px solid var(--border2)', borderTopColor: 'var(--accent)',
      borderRadius: '50%', animation: 'spin 0.7s linear infinite',
    }} />
  )
}
