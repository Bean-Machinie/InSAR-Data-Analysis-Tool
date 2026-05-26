import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useProjectStore } from './stores/projectStore'
import { ProjectPicker } from './components/ProjectPicker/ProjectPicker'
import { ProjectSidebar } from './components/Layout/ProjectSidebar'

const queryClient = new QueryClient()

function AppShell() {
  const { project } = useProjectStore()

  if (!project) {
    return (
      <div className="h-full w-full flex items-center justify-center" style={{ background: 'var(--bg)' }}>
        <ProjectPicker />
      </div>
    )
  }

  return (
    <div className="flex h-full w-full" style={{ background: 'var(--bg)' }}>
      <ProjectSidebar project={project} />
      {/* Map view will be added in Phase 2 */}
      <div className="flex-1 flex items-center justify-center">
        <div style={{ color: 'var(--text2)', textAlign: 'center' }}>
          <p className="text-lg font-semibold" style={{ color: 'var(--text)' }}>
            {project.projectName} — {project.orbit} orbit
          </p>
          <p className="text-sm mt-1">
            {project.sceneCount} scenes · {project.dates[0]} → {project.dates[project.dates.length - 1]}
          </p>
          <p className="text-xs mt-2" style={{ color: 'var(--accent2)' }}>
            Map view coming in Phase 2
          </p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppShell />
    </QueryClientProvider>
  )
}
