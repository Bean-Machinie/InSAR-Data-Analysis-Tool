import { useViewerStore } from '../../stores/viewerStore'

interface Props {
  dates: string[]
}

export function DateSlider({ dates }: Props) {
  const { dateIndex, mode, setDateIndex, setMode } = useViewerStore()

  if (!dates.length) return null

  return (
    <div
      style={{
        height: 48,
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        padding: '0 16px',
        borderTop: '1px solid var(--border2)',
        background: 'rgba(11,24,37,0.97)',
        backdropFilter: 'blur(10px)',
        flexShrink: 0,
      }}
    >
      <button
        onClick={() => setMode('velocity')}
        style={{
          flexShrink: 0, height: 28, padding: '0 12px', borderRadius: 6,
          border: `1px solid ${mode === 'velocity' ? 'var(--accent2)' : 'var(--border2)'}`,
          background: 'transparent',
          color: mode === 'velocity' ? 'var(--accent2)' : 'var(--text2)',
          cursor: 'pointer', fontSize: 11, fontWeight: 600,
          transition: 'all 0.15s',
        }}
      >
        Velocity
      </button>

      <span style={{ fontSize: 10, color: 'var(--text2)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
        {dates[0]}
      </span>

      <input
        type="range"
        min={0}
        max={dates.length - 1}
        step={1}
        value={dateIndex}
        onChange={(e) => {
          setDateIndex(Number(e.target.value))
          setMode('date')
        }}
        style={{ flex: 1, accentColor: 'var(--accent)' }}
      />

      <span style={{ fontSize: 10, color: 'var(--text2)', flexShrink: 0, fontVariantNumeric: 'tabular-nums' }}>
        {dates[dates.length - 1]}
      </span>

      <span
        style={{
          flexShrink: 0, fontSize: 11, fontWeight: 700,
          color: 'var(--accent)', minWidth: 82, textAlign: 'right',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {mode === 'date' ? dates[dateIndex] : '— velocity —'}
      </span>
    </div>
  )
}
