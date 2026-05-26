import { create } from 'zustand'
import type { ProjectInfo } from '../types'

interface ProjectStore {
  project: ProjectInfo | null
  isLoading: boolean
  error: string | null
  setProject: (p: ProjectInfo | null) => void
  setLoading: (v: boolean) => void
  setError: (e: string | null) => void
}

export const useProjectStore = create<ProjectStore>((set) => ({
  project: null,
  isLoading: false,
  error: null,
  setProject: (p) => set({ project: p, error: null }),
  setLoading: (v) => set({ isLoading: v }),
  setError: (e) => set({ error: e, isLoading: false }),
}))
