// Ollaya Bench: click a column header to sort a leaderboard table. Without this script the tables
// stay in rank order, which is also the page's default.
;(() => {
  const ASC = new Set(['rank', 'model', 'ece', 'latency', 'memory', 'license'])
  document.querySelectorAll('table[data-sortable]').forEach((table) => {
    const body = table.tBodies[0]
    const heads = [...table.tHead.rows[0].cells]
    let current = 'rank'
    let dir = 1
    heads.forEach((th, col) => {
      const button = th.querySelector('button[data-sort]')
      if (!button) return
      const key = button.dataset.sort
      button.addEventListener('click', () => {
        dir = current === key ? -dir : ASC.has(key) ? 1 : -1
        current = key
        const value = (row) => row.cells[col]?.dataset.v ?? ''
        const rows = [...body.rows].sort((a, b) => {
          const x = value(a)
          const y = value(b)
          const nx = Number(x)
          const ny = Number(y)
          const c = x !== '' && y !== '' && !Number.isNaN(nx) && !Number.isNaN(ny) ? nx - ny : x.localeCompare(y)
          return c * dir
        })
        body.append(...rows)
        heads.forEach((h) => h.removeAttribute('aria-sort'))
        th.setAttribute('aria-sort', dir > 0 ? 'ascending' : 'descending')
      })
    })
  })
})()
