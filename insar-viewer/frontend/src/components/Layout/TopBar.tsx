import { useProjectStore } from '../../stores/projectStore'
import { useViewerStore } from '../../stores/viewerStore'
import { api } from '../../api'

interface Props {
  projectName: string
  ncFile: string
  onOpenSettings: () => void
}

export function TopBar({ projectName, ncFile, onOpenSettings }: Props) {
  const { setProject } = useProjectStore()
  const { setPointsData, setSelectedPixel, resetLayers } = useViewerStore()

  const handleClose = async () => {
    try { await api.closeProject() } catch { /* ignore */ }
    setSelectedPixel(null)
    setPointsData(null)
    resetLayers({})
    setProject(null)
  }

  const shortFile = ncFile.split(/[\\/]/).pop() ?? ncFile

  return (
    <div style={{
      height: 40, flexShrink: 0,
      background: 'rgba(11,24,37,0.97)',
      borderBottom: '1px solid var(--border2)',
      display: 'flex', alignItems: 'center',
      padding: '0 12px', gap: 8, zIndex: 700,
    }}>
      <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)', letterSpacing: 0.3, flexShrink: 0 }}>
        InSAR Viewer
      </span>
      <span style={{ fontSize: 11, color: 'var(--border2)', userSelect: 'none', flexShrink: 0 }}>|</span>
      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, minWidth: 0 }}>
        {projectName}
      </span>
      <span style={{ fontSize: 10, color: 'var(--text2)', fontFamily: '"Cascadia Code","Fira Code",monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 280, flexShrink: 0 }}>
        {shortFile}
      </span>
      <button
        onClick={onOpenSettings}
        title="Settings"
        style={{ flexShrink: 0, width: 28, height: 26, background: 'transparent', border: '1px solid var(--border2)', borderRadius: 5, color: 'var(--text2)', cursor: 'pointer', fontSize: 14, display: 'grid', placeItems: 'center', transition: 'border-color 0.15s, color 0.15s' }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)' }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text2)' }}
      >⚙</button>
      <button
        onClick={handleClose}
        style={{ flexShrink: 0, height: 26, padding: '0 12px', background: 'transparent', border: '1px solid var(--border2)', borderRadius: 5, color: 'var(--text2)', cursor: 'pointer', fontSize: 11, transition: 'border-color 0.15s, color 0.15s' }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--danger)'; e.currentTarget.style.color = 'var(--danger)' }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--text2)' }}
      >Close</button>
    </div>
  )
}
