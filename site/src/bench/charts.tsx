/**
 * Bench charts, drawn at build time as inline SVG: no script, no chart library. Colour carries one
 * thing, the Pareto frontier (Ollaya blue) against every other model (muted ink); every point links
 * to its model and names itself on hover (<title>), and the tables next to the charts carry every number.
 */
import { tickLabel } from '../components/charts'
import { modelUrl, type Device, fmtMs, frontier, pointsOn, TASKS } from './data'

const MONO = 'ui-monospace, SFMono-Regular, Menlo, monospace'

function niceRange(lo: number, hi: number, step: number): number[] {
  const a = Math.floor(lo / step) * step
  const b = Math.ceil(hi / step) * step
  const out = []
  for (let v = a; v <= b + 1e-9; v += step) out.push(Math.round(v))
  return out
}

function logTicks(lo: number, hi: number): number[] {
  const all = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
  const a = all.filter((v) => v <= lo).at(-1) ?? all[0]!
  const b = all.find((v) => v >= hi) ?? all.at(-1)!
  return all.filter((v) => v >= a && v <= b)
}

/** Score against median latency on one device, log x, with the frontier drawn as a staircase. */
export function ParetoChart({ device, compact }: { device: Device; compact?: boolean }) {
  const pts = pointsOn(device)
  if (!pts.length) return null
  const front = frontier(device)
  const W = compact ? 400 : 1000
  const H = compact ? 380 : 440
  const m = { l: compact ? 34 : 48, r: compact ? 10 : 20, t: 14, b: compact ? 40 : 46 }
  const lat = (name: string) => pts.find((p) => p.name === name)!.speed.get(device.slug)!.p50_ms!
  const xs = pts.map((p) => p.speed.get(device.slug)!.p50_ms!)
  const ys = pts.map((p) => p.score!.score)
  const xTicks = logTicks(Math.min(...xs), Math.max(...xs))
  const yTicks = niceRange(Math.min(0, ...ys), Math.max(...ys), 10)
  const [x0, x1] = [Math.log(xTicks[0]!), Math.log(xTicks.at(-1)!)]
  const x = (v: number) => m.l + ((Math.log(v) - x0) / (x1 - x0)) * (W - m.l - m.r)
  const y = (v: number) => H - m.b - ((v - yTicks[0]!) / (yTicks.at(-1)! - yTicks[0]!)) * (H - m.t - m.b)
  const fr = pts.filter((p) => front.has(p.name)).sort((a, b) => lat(a.name) - lat(b.name))
  let stair = ''
  fr.forEach((p, i) => {
    const px = x(lat(p.name))
    const py = y(p.score!.score)
    stair += i === 0 ? `M${px} ${py}` : ` H${px} V${py}`
  })
  const fs = compact ? 10 : 11.5
  // Frontier labels: the first free spot around each point, never over another label or a point and
  // inside the plot. The stair runs right from each point, so above-left comes first; the first
  // point (nothing on its left) tries below-right first and the last (no stair after it) the right.
  const charW = fs * 0.61
  const pointBoxes = pts.map((p) => ({ x: x(lat(p.name)) - 6, y: y(p.score!.score) - 6, w: 12, h: 12 }))
  const placed: { x: number; y: number; w: number; h: number }[] = []
  const hit = (a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) =>
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h
  const labels = fr.map((p, i) => {
    const px = x(lat(p.name))
    const py = y(p.score!.score)
    const w = p.name.length * charW
    type Spot = { x: number; y: number; anchor: 'start' | 'end' }
    const aboveLeft: Spot = { x: px - 8, y: py - 9, anchor: 'end' }
    const aboveRight: Spot = { x: px + 8, y: py - 9, anchor: 'start' }
    const belowRight: Spot = { x: px + 8, y: py + 17, anchor: 'start' }
    const belowLeft: Spot = { x: px - 8, y: py + 17, anchor: 'end' }
    const right: Spot = { x: px + 10, y: py + 4, anchor: 'start' }
    const left: Spot = { x: px - 10, y: py + 4, anchor: 'end' }
    const lift = (sp: Spot, dy: number): Spot => ({ ...sp, y: sp.y + dy })
    const near =
      i === fr.length - 1
        ? [right, aboveLeft, belowLeft, aboveRight]
        : i === 0
          ? [belowRight, right, aboveRight, aboveLeft]
          : [aboveLeft, belowRight, aboveRight, belowLeft, left, right]
    // Then further out, a line at a time, before giving up (the point keeps its hover title).
    const order = [...near, ...[14, 28].flatMap((d) => [lift(aboveLeft, -d), lift(aboveRight, -d), lift(belowRight, d), lift(belowLeft, d)])]
    const box = (s: Spot) => ({ x: s.anchor === 'end' ? s.x - w : s.x, y: s.y - fs, w, h: fs + 3 })
    const fits = (s: Spot) => {
      const b = box(s)
      return b.x >= m.l + 2 && b.x + b.w <= W - 2 && b.y >= 0 && b.y + b.h <= H - m.b && !placed.some((o) => hit(b, o)) && !pointBoxes.some((o) => hit(b, o))
    }
    const at = order.find(fits)
    if (!at) return null
    placed.push(box(at))
    return { name: p.name, ...at }
  }).filter((l) => l !== null)
  const label = `Score against median latency on the ${device.name}: ${pts.length} models, ${fr.length} on the Pareto frontier`
  return (
    <svg viewBox={`0 0 ${W} ${H}`} class="h-auto w-full" role="img" aria-label={label}>
      <title>{label}</title>
      {yTicks.map((t) => (
        <g>
          <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} class="stroke-line" stroke-width="1" />
          <text x={m.l - 8} y={y(t) + 4} text-anchor="end" class="fill-muted" font-size={fs - 0.5}>
            {t}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <g>
          <line x1={x(t)} x2={x(t)} y1={H - m.b} y2={H - m.b + 4} class="stroke-line-strong" stroke-width="1" />
          <text x={x(t)} y={H - m.b + 17} text-anchor="middle" class="fill-muted" font-size={fs - 0.5}>
            {tickLabel(t)}
          </text>
        </g>
      ))}
      <text x={W - m.r} y={H - 6} text-anchor="end" class="fill-muted" font-size={fs}>
        Median latency, five questions (ms, log scale)
      </text>
      <text x={12} y={m.t} text-anchor="end" class="fill-muted" font-size={fs} transform={`rotate(-90 12 ${m.t})`}>
        Score
      </text>
      <path d={stair} fill="none" style="stroke:var(--color-us)" stroke-width="2" stroke-linejoin="round" opacity="0.45" />
      {[...pts]
        .sort((a, b) => Number(front.has(a.name)) - Number(front.has(b.name)))
        .map((p) => {
          const on = front.has(p.name)
          const px = x(lat(p.name))
          const py = y(p.score!.score)
          return (
            <a href={modelUrl(p.name)}>
              <g>
                <title>{`${p.name}: score ${p.score!.score.toFixed(1)}, ${fmtMs(lat(p.name))}`}</title>
                <circle cx={px} cy={py} r={12} fill="transparent" />
                <circle
                  cx={px}
                  cy={py}
                  r={on ? 5.5 : 4.5}
                  style={on ? 'fill:var(--color-us)' : 'fill:var(--color-faint)'}
                  class="stroke-canvas"
                  stroke-width="2"
                />
              </g>
            </a>
          )
        })}
      {labels.map((l) => (
        <text x={l.x} y={l.y} text-anchor={l.anchor} font-size={fs} class="fill-fg" font-family={MONO}>
          {l.name}
        </text>
      ))}
    </svg>
  )
}

/** Chance-corrected accuracy on each task, as horizontal bars from zero (negative ones go left). */
export function TaskBars({ tasks, trainedOn }: { tasks: Record<string, number>; trainedOn: string[] }) {
  const vals = TASKS.map((t) => tasks[t.id]).filter((v): v is number => v !== undefined)
  const lo = Math.min(0, ...vals)
  const hi = Math.max(100, ...vals)
  const span = hi - lo
  const zero = ((0 - lo) / span) * 100
  let area = ''
  return (
    <div class="grid grid-cols-[minmax(0,9rem)_1fr_3.5rem] items-center gap-x-3 gap-y-1.5 text-[13px]">
      {TASKS.map((t) => {
        const v = tasks[t.id]
        const head = t.area !== area
        area = t.area
        const left = v === undefined ? zero : ((Math.min(0, v) - lo) / span) * 100
        const width = v === undefined ? 0 : (Math.abs(v) / span) * 100
        return (
          <>
            {head ? <div class="col-span-3 mt-3 text-[11px] font-medium tracking-wide text-muted uppercase first:mt-0">{t.area}</div> : null}
            <div class="truncate text-body">
              {t.label}
              {trainedOn.includes(t.id) ? <span title="Trained on this task's train split"> *</span> : null}
            </div>
            <div class="relative h-2.5 rounded-sm bg-fill" aria-hidden="true">
              <span class="absolute inset-y-0 w-px bg-line-strong" style={`left:${zero.toFixed(2)}%`}></span>
              <span class="absolute inset-y-0 rounded-sm bg-bar" style={`left:${left.toFixed(2)}%;width:${width.toFixed(2)}%`}></span>
            </div>
            <div class="text-right font-medium text-fg tabular-nums">{v === undefined ? '–' : v.toFixed(1)}</div>
          </>
        )
      })}
    </div>
  )
}
