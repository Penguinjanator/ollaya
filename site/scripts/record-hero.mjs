// Records the home page's terminal example, src/data/hero.json: runs its command through the
// installed CLI against the local server, keeps the output as printed, and times it (the median
// `total_duration` of ten warm runs, which the caption quotes). The site draws the terminal from
// this file, so no number in it is typed in by hand.
//
// To change the example, edit `model`, `preset` or `message` in hero.json and run this; set `device`
// to the GPU it ran on. Needs `ollaya` on PATH (or OLLAYA_BIN) and the model pulled.
//
//   node scripts/record-hero.mjs

import { spawnSync } from 'node:child_process'
import { readFile, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const file = fileURLToPath(new URL('../src/data/hero.json', import.meta.url))
const bin = process.env.OLLAYA_BIN || 'ollaya'
const hero = JSON.parse(await readFile(file, 'utf8'))

function ollaya(...args) {
  const r = spawnSync(bin, args, { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`${bin} ${args[0]} failed:\n${r.stderr || r.error}`)
  return r.stdout
}

const decide = () => JSON.parse(ollaya('run', hero.model, '--preset', hero.preset, '--format', 'json', '--verbose', hero.message))

decide() // warm-up: loads the model
const runs = Array.from({ length: 10 }, decide)
const answers = JSON.stringify(runs[0].answers)
if (runs.some((r) => JSON.stringify(r.answers) !== answers)) throw new Error('the answers differ between runs')
const ms = runs.map((r) => r.total_duration / 1e6).sort((a, b) => a - b)

hero.recorded = {
  date: new Date().toLocaleDateString('sv'),
  ollaya: ollaya('--version').trim().split(/\s+/).at(-1),
}
hero.latency_ms = Math.round((ms[4] + ms[5]) / 2)
hero.output = runs[0]
await writeFile(file, `${JSON.stringify(hero, null, 2)}\n`)
console.log(`hero: ${hero.model}, ${Object.keys(runs[0].answers).length} questions, ${hero.latency_ms} ms (median of ${ms.length})`)
