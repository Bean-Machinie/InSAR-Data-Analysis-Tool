import { useState } from 'react'
import type { ProjectInfo } from '../../types'
import { useViewerStore } from '../../stores/viewerStore'
import { CoherenceSlider } from './CoherenceSlider'

interface Props {
  project: ProjectInfo
}

export function LayerPanel({ project }: Props) {
  const { layers, setLayerEnabled, setLayerOpacity } = useViewerStore()
  const [baseOpen, setBaseOpen] = useState(true)
  const [dataOpen, setDataOpen] = useState(true)

  return (
    <aside
      style={{
        width: 272,
        flexShrink: 0,
        background: 'var(--panel)',
        borderLeft: '1px solid var(--border2)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div style={{ padding: '9px 12px', borderBottom: '1px solid var(--border)', flexShrink: 0 }}>
        <span style={{ fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.6px', color: 'var(--text2)' }}>
          Layers
        </span>
      </div>

      <div style={{ flex: 1, overflowY: 'auto' }}>
        {/* Basemaps section */}
        <Section title="Basemaps" open={baseOpen} onToggle={() => setBaseOpen(o => !o)}>
          {project.baseLayers.map((bl) => (
            <LayerRow
              key={bl.key}
              label={bl.label}
              enabled={layers[bl.key]?.enabled ?? bl.defaultEnabled}
              opacity={layers[bl.key]?.opacity ?? bl.defaultOpacity}
              onToggle={(v) => setLayerEnabled(bl.key, v)}
              onOpacity={(v) => setLayerOpacity(bl.key, v)}
            />
          ))}
        </Section>

        {/* Data overlays section */}
        <Section title="Data Overlays" open={dataOpen} onToggle={() => setDataOpen(o => !o)}>
          {project.dataLayers.map((dl) => (
            <LayerRow
              key={dl.key}
              label={dl.label}
              enabled={layers[dl.key]?.enabled ?? dl.defaultEnabled}
              opacity={layers[dl.key]?.opacity ?? dl.defaultOpacity}
              onToggle={(v) => setLayerEnabled(dl.key, v)}
              onOpacity={(v) => setLayerOpacity(dl.key, v)}
              valueRange={dl.valueRange}
              units={dl.units}
              colormap={dl.colormap}
            />
          ))}
        </Section>
      </div>

      <CoherenceSlider
        cohDefault={project.cohThresholdDefault}
        cohStep={project.cohSliderStep}
      />
    </aside>
  )
}

function Section({ title, open, onToggle, children }: {
  title: string
  open: boolean
  onToggle: () => void
  children: React.ReactNode
}) {
  return (
    <div style={{ borderBottom: '1px solid var(--border)' }}>
      <button
        onClick={onToggle}
        style={{
          width: '100%', padding: '7px 12px', background: 'transparent',
          border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: 10, fontWeight: 700, textTransform: 'uppercase',
          letterSpacing: '0.6px', color: 'var(--text2)',
        }}
      >
        {title}
        <span style={{ transform: open ? undefined : 'rotate(-90deg)', transition: 'transform 0.15s' }}>▾</span>
      </button>
      {open && children}
    </div>
  )
}

function LayerRow({ label, enabled, opacity, onToggle, onOpacity, valueRange, units, colormap }: {
  label: string
  enabled: boolean
  opacity: number
  onToggle: (v: boolean) => void
  onOpacity: (v: number) => void
  valueRange?: [number, number] | null
  units?: string
  colormap?: string
}) {
  const [opOpen, setOpOpen] = useState(false)

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px',
          transition: 'background 0.15s',
        }}
      >
        <label style={{ display: 'flex', alignItems: 'center', gap: 7, flex: 1, cursor: 'pointer', minWidth: 0 }}>
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
            style={{ accentColor: 'var(--accent)', width: 13, height: 13, flexShrink: 0, cursor: 'pointer' }}
          />
          <span style={{ fontSize: 12, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {label}
          </span>
        </label>
        <button
          onClick={() => setOpOpen(o => !o)}
          title="Opacity"
          style={{
            flexShrink: 0, width: 22, height: 22, border: '1px solid transparent',
            borderRadius: 5, background: 'transparent', color: 'var(--text2)',
            cursor: 'pointer', fontSize: 12, display: 'grid', placeItems: 'center',
          }}
        >
          ◑
        </button>
      </div>
      {opOpen && (
        <div style={{ padding: '4px 12px 8px' }}>
          <input
            type="range" min={0} max={100} step={5}
            value={Math.round(opacity * 100)}
            onChange={(e) => onOpacity(Number(e.target.value) / 100)}
            style={{ width: '100%', accentColor: 'var(--accent)' }}
          />
          {valueRange && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, fontSize: 10, color: 'var(--text2)' }}>
              <span>{round2(valueRange[0])}</span>
              <div style={{ flex: 1, height: 8, borderRadius: 2, background: gradientFor(colormap ?? 'RdBu_r'), border: '1px solid rgba(255,255,255,0.1)' }} />
              <span>{round2(valueRange[1])}</span>
              {units && <span style={{ marginLeft: 2 }}>{units}</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function round2(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2)
}

function gradientFor(colormap: string): string {
  const maps: Record<string, string> = {
    RdBu_r: 'linear-gradient(to right,#053061,#f7f7f7,#67000d)',
    viridis: 'linear-gradient(to right,#440154,#31688e,#35b779,#fde725)',
    terrain: 'linear-gradient(to right,#333399,#006600,#c8a450,#ffffff)',
    hot_r: 'linear-gradient(to right,#ffffff,#ffaa00,#ff2200,#000000)',
  }
  return maps[colormap] ?? maps['RdBu_r']
}
