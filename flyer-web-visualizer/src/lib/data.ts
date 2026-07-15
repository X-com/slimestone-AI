// Load + type the machine archive (genetic-ml/data/outputs/ga_archive.jsonl,
// served from /public via symlink).

export interface Vec3 {
  x: number
  y: number
  z: number
}

export interface Block extends Vec3 {
  state: number
}

export interface Candidate {
  id: number
  trigger: Vec3
  blocks: Block[]
  name?: string
  path?: string
}

export interface Result {
  id: number
  ok: boolean
  working: boolean
  ticks: number
  start: number
  end: number
  period: number
  shift: Vec3
  elapsedNs: number
  ticksPerSecond: number
}

export interface Machine {
  hash: string // unique id + selection key (archive: structural hash; uploaded: synthetic)
  label?: string // float text override (uploaded uses "#id"); archive falls back to hash[:8]
  source?: string // filename, for machines loaded from an uploaded .data file
  generation: number
  origin: string
  block_count: number
  candidate: Candidate
  result: Result | null // null for uploaded bare candidates (no simulation metadata)
  found_at: string
}

export async function loadMachines(
  url = `${import.meta.env.BASE_URL}ga_archive.jsonl`,
): Promise<Machine[]> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`)
  const text = await res.text()
  return text
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as Machine)
}

// Complex example machines: single candidate-shaped JSON objects under public/machines/,
// listed in public/machines/manifest.json (regenerate with `pnpm sync-machines`).
function complexToMachine(file: string, c: Candidate): Machine {
  return {
    hash: `complex:${file}`,
    label: c.name ?? file,
    source: file,
    generation: 0,
    origin: 'complex',
    block_count: c.blocks.length,
    candidate: c,
    result: null,
    found_at: '',
  }
}

export async function loadComplexMachines(
  base = import.meta.env.BASE_URL,
): Promise<Machine[]> {
  const res = await fetch(`${base}machines/manifest.json`)
  if (!res.ok) return [] // no manifest -> no complex examples
  const files = (await res.json()) as string[]
  return Promise.all(
    files.map(async (f) => {
      const r = await fetch(`${base}machines/${f}`)
      if (!r.ok) throw new Error(`Failed to load ${f}: ${r.status}`)
      return complexToMachine(f, (await r.json()) as Candidate)
    }),
  )
}

// Compact binary .data format (mirrors genetic-ml/genetic_ml/compact_format.py):
// little-endian, records concatenated to EOF.
//   header: int32 id, int32 trigger x/y/z, uint32 block_count   (20 bytes)
//   each block: int32 x, int32 y, int32 z, uint32 state         (16 bytes)
const HEADER_BYTES = 20
const BLOCK_BYTES = 16

export function parseCompactData(name: string, buf: ArrayBuffer): Machine[] {
  const view = new DataView(buf)
  const machines: Machine[] = []
  let off = 0
  let index = 0
  while (off < buf.byteLength) {
    if (off + HEADER_BYTES > buf.byteLength)
      throw new Error(`${name}: truncated record header at byte ${off}`)
    const id = view.getInt32(off, true)
    const tx = view.getInt32(off + 4, true)
    const ty = view.getInt32(off + 8, true)
    const tz = view.getInt32(off + 12, true)
    const blockCount = view.getUint32(off + 16, true)
    off += HEADER_BYTES

    const blocks: Block[] = []
    for (let i = 0; i < blockCount; i++) {
      if (off + BLOCK_BYTES > buf.byteLength)
        throw new Error(`${name}: truncated block data at byte ${off}`)
      blocks.push({
        x: view.getInt32(off, true),
        y: view.getInt32(off + 4, true),
        z: view.getInt32(off + 8, true),
        state: view.getUint32(off + 12, true),
      })
      off += BLOCK_BYTES
    }

    machines.push({
      hash: `uploaded:${name}#${index}`,
      label: `#${id}`,
      source: name,
      generation: 0,
      origin: 'uploaded',
      block_count: blockCount,
      candidate: { id, trigger: { x: tx, y: ty, z: tz }, blocks },
      result: null,
      found_at: '',
    })
    index++
  }
  return machines
}
