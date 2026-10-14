// Social preview cards (1200×630 PNG) for Ollaya Bench, made from the built edition
// (dist-bench/api/v1/leaderboard.json) into bench-public/static/og/:
//
//   og/hardware/<device>.png        a device's leaderboard (the first device's is also the home page's)
//   og/models/<model>/<tag>.png     a model's page: rank, score, accuracy, speed and memory
//
// They change with every edition, so they are not committed (.gitignore): `npm run deploy:bench`
// builds, runs this, and builds again so the pages point at the cards. Pages without a card use the
// committed og-bench.png. Needs a Chromium and the network, like og-images.mjs.
//
//   node scripts/og-bench.mjs

import { readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { esc, inlineFonts, page, renderCards, top } from './og-lib.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const out = join(root, 'bench-public', 'static')
const board = JSON.parse(await readFile(join(root, 'dist-bench', 'api', 'v1', 'leaderboard.json'), 'utf8').catch(() => {
  throw new Error('dist-bench/api/v1/leaderboard.json is missing: run `npm run build:bench` first')
}))
const host = new URL(process.env.BENCH_ORIGIN?.trim() || 'https://bench.ollaya.dev').host
const fonts = await inlineFonts()
const brand = 'ollaya <span style="color:#8a8a8a">bench</span>'

const CSS = `
  .by { margin-top: 14px; font-size: 26px; color: #a3a3a3; }
  .stats { position: relative; margin-top: 40px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 28px;
    border-top: 1px solid #262626; padding-top: 28px; }
  .stats span { display: block; font-size: 19px; color: #8a8a8a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .stats b { display: block; margin-top: 6px; font-size: 46px; font-weight: 500; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
  .stats i { display: block; margin-top: 4px; font-style: normal; font-size: 19px; color: #8a8a8a; white-space: nowrap; }
  .rows { position: relative; margin-top: 26px; display: grid; grid-template-columns: 44px 1fr auto auto; column-gap: 28px; row-gap: 12px;
    font-size: 30px; align-items: baseline; font-variant-numeric: tabular-nums; }
  .rows .n { color: #8a8a8a; }
  .rows .m { font-family: 'Geist Mono', ui-monospace, monospace; }
  .rows .s { text-align: right; } .rows .l { text-align: right; color: #a3a3a3; }
  h1.board { font-size: 64px; margin-top: 20px; white-space: nowrap; }
  .rows .h { font-size: 18px; color: #8a8a8a; letter-spacing: .06em; text-transform: uppercase; }
`

const fmtMs = (ms) => (ms == null ? '–' : ms >= 10000 ? `${(ms / 1000).toFixed(1)} s` : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : ms >= 100 ? `${Math.round(ms)} ms` : `${ms.toFixed(1)} ms`)
const fmtMb = (mb) => (mb == null ? '–' : mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`)
const pct = (v) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`)
const scored = board.models.filter((m) => m.score !== null)
const runtimes = { 'ONNX Runtime': 'ONNX Runtime', 'llama.cpp': 'llama.cpp', router: 'router' }

function modelCard(m) {
  const size = Math.min(112, Math.floor(1040 / (m.model.length * 0.6)))
  const facts = [m.publisher, m.params, runtimes[m.runtime], m.license.split(' (')[0]].filter(Boolean).map(esc).join(' · ')
  const speed = board.devices.map((d) => {
    const s = m.speed[d.id]
    const mem = d.device === 'cpu' ? 'RAM' : 'VRAM'
    return `<div><span>${esc(d.name)}</span><b>${fmtMs(s?.p50_ms)}</b><i>${s?.memory_mb != null ? `${fmtMb(s.memory_mb)} ${mem}` : 'five questions'}</i></div>`
  })
  const cells =
    m.score !== null
      ? [
          `<div><span>Score</span><b>${m.score.toFixed(1)}</b><i>${m.score_ci[0].toFixed(1)}–${m.score_ci[1].toFixed(1)}, 95%</i></div>`,
          `<div><span>Public benchmark</span><b>${pct(m.public_macro_accuracy)}</b><i>typed-decisions ${pct(m.typed_accuracy)}</i></div>`,
          ...speed,
        ]
      : speed
  const badge = m.score !== null ? `Edition ${board.edition} · Rank ${m.rank} of ${scored.length}` : `Edition ${board.edition} · speed and memory`
  return page(
    fonts,
    `${top(`${host}/models/${m.model}`, brand)}
    <div class="main"><span class="badge">${esc(badge)}</span>
      <h1 class="model" style="font-size:${size}px">${esc(m.model)}</h1><p class="by">${facts}</p></div>
    <div class="stats" style="grid-template-columns:repeat(${cells.length},1fr)">${cells.join('')}</div>`,
    CSS,
  )
}

function deviceCard(d) {
  const top5 = scored.slice().sort((a, b) => b.score - a.score).slice(0, 5)
  const rows = top5
    .map((m, i) => `<span class="n">${i + 1}</span><span class="m">${esc(m.model)}</span><span class="s">${m.score.toFixed(1)}</span><span class="l">${fmtMs(m.speed[d.id]?.p50_ms)}</span>`)
    .join('')
  return page(
    fonts,
    `${top(`${host}${d === board.devices[0] ? '' : `/hardware/${d.id}`}`, brand)}
    <div class="main"><span class="badge">${esc(`Edition ${board.edition} · ${board.models.length} models · ${d.kind}`)}</span>
      <h1 class="board" style="font-size:${Math.min(64, Math.floor(1050 / (`Leaderboard on the ${d.name}`.length * 0.52)))}px">Leaderboard on the ${esc(d.name)}</h1>
      <div class="rows"><span></span><span></span><span class="h s">Score</span><span class="h l">Latency</span>${rows}</div></div>`,
    CSS,
  )
}

await rm(join(out, 'og'), { recursive: true, force: true })
const cards = [
  ...board.devices.map((d) => ({ file: `og/hardware/${d.id}.png`, html: deviceCard(d) })),
  ...board.models.map((m) => ({ file: `og/models/${m.model.replace(':', '/')}.png`, html: modelCard(m) })),
]
await renderCards(cards, out)
console.log(`og-bench: ${cards.length} cards for edition ${board.edition}`)
