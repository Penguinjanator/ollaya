/** Ollaya Bench's public origin (BENCH_ORIGIN at build time), set once by the static build before rendering. */
let origin = ''

export function setBenchOrigin(o: string): void {
  origin = o.replace(/\/$/, '')
}

/** A URL on Ollaya Bench, e.g. benchUrl('/methodology'). */
export function benchUrl(path = '/'): string {
  return `${origin}${path}`
}
