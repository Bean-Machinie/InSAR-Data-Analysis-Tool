// RdBu_r colormap — matches matplotlib exactly
const RDBU_R: [number, [number, number, number]][] = [
  [0.000, [5,   48,  97 ]],
  [0.125, [33,  102, 172]],
  [0.250, [67,  147, 195]],
  [0.375, [146, 197, 222]],
  [0.500, [247, 247, 247]],
  [0.625, [244, 165, 130]],
  [0.750, [214, 96,  77 ]],
  [0.875, [178, 24,  43 ]],
  [1.000, [103, 0,   31 ]],
]

function lerpStops(stops: typeof RDBU_R, t: number): string {
  t = Math.max(0, Math.min(1, t))
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i]
    const [t1, c1] = stops[i + 1]
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0)
      return `rgb(${Math.round(c0[0] + f * (c1[0] - c0[0]))},${Math.round(c0[1] + f * (c1[1] - c0[1]))},${Math.round(c0[2] + f * (c1[2] - c0[2]))})`
    }
  }
  return 'rgb(103,0,31)'
}

/**
 * Map a value to an RdBu_r color using TwoSlopeNorm (symmetric around 0).
 * Returns null if value is null or non-finite.
 */
export function valueToColor(v: number | null, vmin: number, vmax: number): string | null {
  if (v === null || !isFinite(v)) return null
  // TwoSlopeNorm: [vmin, 0, vmax] → [0, 0.5, 1]
  const t = v <= 0
    ? 0.5 * (v - vmin) / (0 - vmin)
    : 0.5 + 0.5 * v / vmax
  return lerpStops(RDBU_R, t)
}

export function computeRange(values: (number | null)[]): [number, number] {
  const finite = values.filter((v): v is number => v !== null && isFinite(v))
  if (finite.length === 0) return [-10, 10]
  finite.sort((a, b) => a - b)
  const p2  = finite[Math.max(0, Math.floor(finite.length * 0.02))]
  const p98 = finite[Math.min(finite.length - 1, Math.floor(finite.length * 0.98))]
  const vr  = Math.max(Math.abs(p2), Math.abs(p98)) || 10
  return [-vr, vr]
}
