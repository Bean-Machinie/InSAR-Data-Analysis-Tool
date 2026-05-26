import type { ProjectInfo } from '../../types'
import { api } from '../../api'
import { useProjectStore } from '../../stores/projectStore'
import { useViewerStore } from '../../stores/viewerStore'

interface Props {
  project: ProjectInfo
}

export function ProjectSidebar({ project }: Props) {
  const { setProject } = useProjectStore()
  const { setPointsData } = useViewerStore()

  async function handleClose() {
    await api.closeProject().catch(() => {})
    setProject(null)
    setPointsData(null)
  }

  return (
    <aside
      className="flex flex-col h-full overflow-hidden"
      style={{ background: 'var(--panel)', borderRight: '1px solid var(--border2)', width: 224 }}
    >
      <div
        className="flex items-center justify-between px-3 py-2 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <span
          className="text-xs font-bold uppercase tracking-wider"
          style={{ color: 'var(--text2)' }}
        >
          Project
        </span>
        <button
          onClick={handleClose}
          className="text-xs px-2 py-1 rounded transition-all"
          style={{ color: 'var(--text2)', border: '1px solid var(--border2)' }}
          title="Close project"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
        {/* Project name */}
        <div>
          <p className="text-xs" style={{ color: 'var(--text2)' }}>Project</p>
          <p className="font-semibold text-sm mt-0.5" style={{ color: 'var(--text)' }}>
            {project.projectName}
          </p>
          <p className="text-xs" style={{ color: 'var(--accent)' }}>
            Orbit {project.orbit}
          </p>
        </div>

        {/* Date range */}
        <div
          className="rounded-lg p-2"
          style={{ background: 'var(--panel2)', border: '1px solid var(--border)' }}
        >
          <p className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--text2)' }}>
            Time window
          </p>
          <p className="text-xs font-mono" style={{ color: 'var(--text)' }}>
            {project.dateRange.start} →
          </p>
          <p className="text-xs font-mono" style={{ color: 'var(--text)' }}>
            {project.dateRange.end}
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2">
          <StatCard label="Scenes" value={String(project.sceneCount)} />
          <StatCard label="Layers" value={String(project.dataLayers.filter(l => l.kind !== 'aoi').length)} />
        </div>

        {/* Dataset file */}
        <div>
          <p className="text-xs" style={{ color: 'var(--text2)' }}>Dataset</p>
          <p className="text-xs font-mono mt-0.5 break-all" style={{ color: 'var(--accent2)' }}>
            {project.ncFile}
          </p>
        </div>

        {/* POIs */}
        {project.pois.length > 0 && (
          <div>
            <p className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--text2)' }}>
              Points of Interest ({project.pois.length})
            </p>
            <div className="flex flex-col gap-1">
              {project.pois.map((poi) => (
                <div
                  key={poi.name}
                  className="rounded px-2 py-1.5"
                  style={{ background: 'var(--panel2)', border: '1px solid var(--border)' }}
                >
                  <p className="text-xs font-semibold" style={{ color: 'var(--text)' }}>
                    {poi.name}
                  </p>
                  <p className="text-xs font-mono" style={{ color: 'var(--text2)' }}>
                    {poi.lat.toFixed(5)}, {poi.lon.toFixed(5)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Scene dates */}
        <div>
          <p className="text-xs font-bold uppercase tracking-wider mb-1" style={{ color: 'var(--text2)' }}>
            Acquisition dates
          </p>
          <div className="flex flex-wrap gap-1">
            {project.dates.map((d) => (
              <span
                key={d}
                className="text-xs px-1.5 py-0.5 rounded font-mono"
                style={{ background: 'var(--panel2)', color: 'var(--text2)', border: '1px solid var(--border)' }}
              >
                {d}
              </span>
            ))}
          </div>
        </div>
      </div>
    </aside>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="rounded-lg p-2"
      style={{ background: 'var(--panel2)', border: '1px solid var(--border)' }}
    >
      <p className="text-xs" style={{ color: 'var(--text2)' }}>{label}</p>
      <p className="text-lg font-bold" style={{ color: 'var(--text)' }}>{value}</p>
    </div>
  )
}
