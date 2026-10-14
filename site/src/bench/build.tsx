/**
 * Static generation for Ollaya Bench (BENCH_ORIGIN). scripts/build-bench.mjs bundles this file, calls
 * renderBench() and writes the files into dist-bench/. Like the main site, the deployed site is plain
 * files: HTML, CSS, one small script, JSON and SVG.
 *
 * Model pages keep the colon in their URL (/models/winnow:e4b): each is written to
 * models/winnow:e4b.html for plain static hosts and to models/winnow/e4b.html, which the bench's
 * _redirects serves for the colon URL on Cloudflare (whose asset server would percent-encode it).
 */
import type { Child } from 'hono/jsx'
import { setAssetVersions } from '../lib/assets'
import { runs } from '../data/results'
import { allEditions, badgeUrl, devices, edition, memoryOn, modelFiles, modelUrl, rows, scoredRows, type Row } from './data'
import { BenchNotFound, DataPage, LeaderboardPage, MethodologyPage, ModelPage } from './pages'
import { BenchLayout, type BenchMeta } from './ui'

export interface BenchOptions {
  /** The bench's public origin (BENCH_ORIGIN). */
  origin: string
  /** The main site's origin (SITE_ORIGIN), for links back to it. */
  main: string
  assetVersions: Record<string, string>
}

export interface OutputFile {
  path: string
  body: string
}

interface Page {
  url: string
  files: string[]
  meta: BenchMeta
  render: () => Child
}

async function toHtml(node: Child): Promise<string> {
  return `<!DOCTYPE html>${await (await (node as Promise<string> | string)).toString()}`
}

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

/** A shields-style badge: "ollaya bench | score 52.3 · #4". Text widths are estimated (Verdana-ish, 11px). */
export function badgeSvg(r: Row): string {
  const left = 'ollaya bench'
  const right = `score ${r.score!.score.toFixed(1)} · #${r.score!.rank}`
  const w = (t: string) => Math.round(t.length * 6.6 + 12)
  const lw = w(left)
  const rw = w(right)
  const W = lw + rw
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="20" role="img" aria-label="${esc(`${left}: ${right}`)}">
<title>${esc(`${r.name} on Ollaya Bench ${edition?.id ?? ''}: ${right}`)}</title>
<clipPath id="r"><rect width="${W}" height="20" rx="3" fill="#fff"/></clipPath>
<g clip-path="url(#r)"><rect width="${lw}" height="20" fill="#171717"/><rect x="${lw}" width="${rw}" height="20" fill="#0969da"/></g>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
<text x="${lw / 2}" y="14">${esc(left)}</text><text x="${lw + rw / 2}" y="14">${esc(right)}</text></g>
</svg>
`
}

function leaderboardJson(origin: string) {
  return {
    edition: edition?.id,
    date: edition?.date,
    ollaya: edition?.ollaya,
    score: edition?.score.definition,
    devices: devices.map((d) => ({ id: d.slug, name: d.name, kind: d.kind, machine: d.machine, device: d.device })),
    models: rows.map((r) => ({
      model: r.name,
      url: `${origin}${modelUrl(r.name)}`,
      badge: r.score ? `${origin}${badgeUrl(r.name)}` : null,
      publisher: r.publisher,
      params: r.params,
      runtime: r.runtime,
      license: r.license,
      digest: r.digest,
      score: r.score?.score ?? null,
      score_ci: r.score?.score_ci ?? null,
      rank: r.score?.rank ?? null,
      trained_on: r.score?.trained_on ?? [],
      public_macro_accuracy: r.public?.macro_accuracy ?? null,
      public_pooled_ece: r.public?.pooled_ece ?? null,
      typed_accuracy: r.typed?.accuracy ?? null,
      typed_ece: r.typed?.ece ?? null,
      speed: Object.fromEntries(
        devices.map((d) => {
          const s = r.speed.get(d.slug)
          return [d.slug, s && !s.error ? { p50_ms: s.p50_ms, p90_ms: s.p90_ms, load_ms: s.load_ms, memory_mb: memoryOn(d, s) ?? null, precision: s.precision } : null]
        }),
      ),
    })),
  }
}

export async function renderBench({ origin, main, assetVersions }: BenchOptions): Promise<OutputFile[]> {
  setAssetVersions(assetVersions)
  const pages: Page[] = []
  devices.forEach((d, i) => {
    pages.push({
      url: i === 0 ? '/' : `/hardware/${d.slug}`,
      files: [i === 0 ? 'index.html' : `hardware/${d.slug}.html`],
      meta: i === 0 ? { nav: 'leaderboard' } : { title: `Leaderboard on the ${d.name}`, description: `Every open decision model on the ${d.name} (${d.kind}): score, latency and memory.`, nav: 'leaderboard' },
      render: () => <LeaderboardPage device={d} />,
    })
  })
  pages.push(
    {
      url: '/methodology',
      files: ['methodology.html'],
      meta: { title: 'Methodology', description: 'How Ollaya Bench measures decision models: the tasks, the score, calibration, speed, memory and the rules we hold ourselves to.', nav: 'methodology' },
      render: () => <MethodologyPage />,
    },
    {
      url: '/data',
      files: ['data.html'],
      meta: { title: 'Data', description: 'Every Ollaya Bench result file, as the tools wrote it, and the static JSON API.', nav: 'data' },
      render: () => <DataPage />,
    },
  )
  for (const r of rows) {
    const s = r.score ? `score ${r.score.score.toFixed(1)}, rank ${r.score.rank} of ${scoredRows.length}` : 'speed and memory'
    pages.push({
      url: modelUrl(r.name),
      files: modelFiles(r.name),
      meta: { title: r.name, description: `${r.name} on Ollaya Bench ${edition?.id}: ${s}, measured on ${devices.map((d) => d.name).join(' and ')}.`, nav: 'leaderboard' },
      render: () => <ModelPage r={r} origin={origin} main={main} />,
    })
  }

  const out: OutputFile[] = []
  for (const p of pages) {
    const html = await toHtml(
      <BenchLayout origin={origin} main={main} path={p.url} meta={p.meta}>
        {p.render()}
      </BenchLayout>,
    )
    for (const f of p.files) out.push({ path: f, body: html })
  }
  out.push({
    path: '404.html',
    body: await toHtml(
      <BenchLayout origin={origin} main={main} path="/404" meta={{ title: 'Page not found', noindex: true }}>
        <BenchNotFound />
      </BenchLayout>,
    ),
  })
  for (const r of scoredRows) out.push({ path: badgeUrl(r.name).slice(1), body: badgeSvg(r) })
  out.push({ path: 'api/v1/leaderboard.json', body: `${JSON.stringify(leaderboardJson(origin), null, 1)}\n` })
  for (const e of allEditions) out.push({ path: `data/editions/${e.id}.json`, body: `${JSON.stringify(e, null, 1)}\n` })
  for (const { file, ...run } of runs) {
    if (edition?.runs.includes(file)) out.push({ path: `data/runs/${file}`, body: `${JSON.stringify(run, null, 1)}\n` })
  }
  out.push({ path: 'robots.txt', body: `User-agent: *\nAllow: /\n\nSitemap: ${origin}/sitemap.xml\n` })
  out.push({
    path: 'sitemap.xml',
    body:
      `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
      pages.map((p) => `  <url><loc>${esc(`${origin}${p.url}`)}</loc></url>\n`).join('') +
      `</urlset>\n`,
  })
  const seen = new Set<string>()
  for (const f of out) {
    if (seen.has(f.path)) throw new Error(`${f.path}: generated twice`)
    seen.add(f.path)
  }
  return out
}
