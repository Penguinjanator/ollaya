// Static build of bench.ollaya.dev into dist-bench/, from the same sources and stylesheet as ollaya.dev:
//
//   1. A fresh output directory gets a copy of bench-public/ (favicon, _headers, _redirects, static/bench.js).
//   2. Tailwind compiles src/styles/app.css (it scans src/, so src/bench/ too) → static/app.css.
//   3. Static files are hashed for ?v=<hash> cache busting.
//   4. src/bench/build.tsx is bundled with esbuild and renderBench() writes every page, badge and data file.
//   5. Checks: required files, doctype, and every internal href/src resolves to a file.
//
// BENCH_ORIGIN (env) is the bench's public origin, SITE_ORIGIN ollaya.dev's (for links back).
// Run `npm run gen` first (npm run build does).

import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { cp, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join, relative, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { build } from 'esbuild'

const root = fileURLToPath(new URL('..', import.meta.url))
const out = join(root, 'dist-bench')
const tmp = join(root, '.dist-bench-tmp')
const cacheDir = join(root, 'node_modules', '.cache', 'ollaya-site')
const origin = new URL(process.env.BENCH_ORIGIN?.trim() || 'https://bench.ollaya.dev').origin
const main = new URL(process.env.SITE_ORIGIN?.trim() || 'https://ollaya.dev').origin
const started = Date.now()

async function listFiles(dir) {
  const files = []
  for (const e of await readdir(dir, { withFileTypes: true })) {
    const p = join(dir, e.name)
    if (e.isDirectory()) files.push(...(await listFiles(p)))
    else files.push(p)
  }
  return files
}

await rm(tmp, { recursive: true, force: true })
await cp(join(root, 'bench-public'), tmp, { recursive: true })

const tailwind = spawnSync(
  join(root, 'node_modules', '.bin', 'tailwindcss'),
  ['-i', 'src/styles/app.css', '-o', join(tmp, 'static', 'app.css'), '--minify'],
  { cwd: root, stdio: ['ignore', 'ignore', 'pipe'], encoding: 'utf8' },
)
if (tailwind.status !== 0) {
  process.stderr.write(tailwind.stderr ?? '')
  throw new Error('tailwindcss failed')
}

const staticDir = join(tmp, 'static')
const assetVersions = {}
for (const file of await listFiles(staticDir)) {
  assetVersions[relative(staticDir, file).split(sep).join('/')] = createHash('sha256').update(await readFile(file)).digest('hex').slice(0, 10)
}

const bundle = join(cacheDir, 'build-bench.mjs')
await build({
  entryPoints: [join(root, 'src', 'bench', 'build.tsx')],
  outfile: bundle,
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node22',
  jsx: 'automatic',
  jsxImportSource: 'hono/jsx',
  logLevel: 'warning',
})
const { renderBench } = await import(`${pathToFileURL(bundle).href}?t=${Date.now()}`)
const files = await renderBench({ origin, main, assetVersions })
for (const f of files) {
  const target = join(tmp, f.path)
  if (relative(tmp, target).startsWith('..')) throw new Error(`refusing to write outside dist-bench/: ${f.path}`)
  await mkdir(dirname(target), { recursive: true })
  await writeFile(target, f.body)
}

// Checks ------------------------------------------------------------------------------------------
const errors = []
const all = (await listFiles(tmp)).map((f) => relative(tmp, f).split(sep).join('/'))
const fileSet = new Set(all)
for (const f of ['index.html', '404.html', 'methodology.html', 'data.html', 'robots.txt', 'sitemap.xml', 'favicon.svg', '_headers', '_redirects', 'static/app.css', 'static/bench.js', 'api/v1/leaderboard.json']) {
  if (!fileSet.has(f)) errors.push(`dist-bench/${f} is missing`)
}
const rewrites = (await readFile(join(tmp, '_redirects'), 'utf8'))
  .split('\n')
  .filter((l) => l.trim() && !l.startsWith('#'))
  .map((l) => l.trim().split(/\s+/))
function resolves(url) {
  const p = decodeURIComponent(url)
  if (p === '/') return fileSet.has('index.html')
  const rel = p.slice(1).replace(/\/$/, '')
  if (fileSet.has(rel) || fileSet.has(`${rel}.html`)) return true
  // /models/:model:* → /models/:model/:splat
  const m = /^\/models\/([^/:]+):(.+)$/.exec(p)
  return !!m && rewrites.some(([from]) => from === '/models/:model:*') && fileSet.has(`models/${m[1]}/${m[2]}.html`)
}
let pages = 0
for (const file of all.filter((f) => f.endsWith('.html'))) {
  pages++
  const html = await readFile(join(tmp, file), 'utf8')
  if (!html.startsWith('<!DOCTYPE html>')) errors.push(`dist-bench/${file}: missing doctype`)
  if (/coming soon|pre-release|illustrative output|being built/i.test(html.replace(/<[^>]+>/g, ' '))) errors.push(`dist-bench/${file}: placeholder wording`)
  for (const [, url] of html.matchAll(/(?:href|src)="(\/(?!\/)[^"#?]*)[^"]*"/g)) if (!resolves(url)) errors.push(`dist-bench/${file}: broken internal link ${url}`)
}
if (errors.length) {
  console.error(`build-bench: ${errors.length} problem(s)\n  - ${[...new Set(errors)].slice(0, 40).join('\n  - ')}`)
  process.exit(1)
}

await rm(out, { recursive: true, force: true })
await cp(tmp, out, { recursive: true })
await rm(tmp, { recursive: true, force: true })
console.log(`build-bench: ${pages} pages, ${all.length} files → dist-bench/ for ${origin} (${Date.now() - started} ms)`)
