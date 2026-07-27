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

// One real, simulator-reported keyframe (tick/order/x/y/z copied verbatim from the .simlog's own
// SimEvent fields by stream_to_visualizer.py's build_animation_record - never synthesized).
// `order` breaks ties between same-tick events; it is NOT a sub-tick timing value (see that
// script's docstring on executedSubtick).
export interface MoveStep {
  tick: number
  order: number
  x: number
  y: number
  z: number
}
export interface BlockMove {
  blockIndex: number // index into candidate.blocks
  steps: MoveStep[]
}

// A piston/sticky_piston's extend/retract timeline - the head isn't a separate block in the
// initial state (it only exists once extended), so it can't be tracked as a "moved" block the
// way BlockMove is; instead each piston gets its own real extended/retracted keyframes, starting
// from its true t=0 state (steps[0].tick is always 0).
export interface ExtensionStep {
  tick: number
  order: number
  extended: boolean
}
export interface BlockExtension {
  blockIndex: number // index into candidate.blocks - always a piston or sticky_piston
  steps: ExtensionStep[]
}

// A rail (golden/activator)/fence gate/trapdoor/redstone lamp's own on-off timeline (powered/open/
// lit) - same shape as BlockExtension, starting from its true t=0 state (steps[0].tick is always
// 0) so an already-on block at load shows correctly instead of only lighting on the first event.
export interface PoweredStep {
  tick: number
  order: number
  on: boolean
}
export interface BlockPowered {
  blockIndex: number // index into candidate.blocks
  steps: PoweredStep[]
}

// One real logged event, in true recorded (tick, order) order - the itemized log the /generator
// page's tick/subtick stepper walks through to verify event ordering. `blockPushed`/`pistonExtend`/
// `pistonRetract` are redundant with moves/extensions (same keyframes, just flattened into one
// timeline); `pistonBlocked`/`observerFired`/`observerOff` are effects moves/extensions can't show
// at all. `scheduledTickDropped` is a diagnostic-only event (see sim_event_log.h's
// ScheduledTickDropped doc) - the simulator's scheduling collision it marks is intentional,
// unchanged behavior, this just makes an otherwise-silent drop visible in the trace.
export type MachineEventKind =
  | 'blockPushed'
  | 'pistonExtend'
  | 'pistonRetract'
  | 'pistonBlocked'
  | 'observerFired'
  | 'observerOff'
  | 'scheduledTickDropped'
  | 'poweredOn'
  | 'poweredOff'
  | 'blockDestroyed'
export interface MachineEvent {
  tick: number
  order: number
  kind: MachineEventKind
  blockIndex: number
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
  // Present only on machines decoded from the /generator page's stream (parseGeneratorRecords) -
  // the real per-tick move/extension timelines read straight from the .simlog, animated by
  // animatedScene.ts.
  moves?: BlockMove[]
  extensions?: BlockExtension[]
  powered?: BlockPowered[]
  events?: MachineEvent[]
  terminationTick?: number
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

// /generator page's wire format (mirrors generator/stream_to_visualizer.py's build_animation_record):
// newline-separated JSON records, {trigger, blocks, moves, extensions, terminationTick, name}.
// JSON (not the binary compact format above) because it also has to carry the per-block move/
// extension timelines, which aren't fixed-size the way a plain block list is.
interface GeneratorRecord {
  name: string
  trigger: Vec3
  blocks: Block[]
  moves: BlockMove[]
  extensions: BlockExtension[]
  powered: BlockPowered[]
  events: MachineEvent[]
  terminationTick: number
}

export function parseGeneratorRecords(frame: string): Machine[] {
  return frame
    .split('\n')
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line) as GeneratorRecord)
    .map((r) => ({
      hash: `gen:${r.name}`,
      label: r.name,
      source: r.name,
      generation: 0,
      origin: 'generator',
      block_count: r.blocks.length,
      candidate: { id: 0, trigger: r.trigger, blocks: r.blocks },
      result: null,
      found_at: '',
      moves: r.moves,
      extensions: r.extensions,
      powered: r.powered,
      events: r.events,
      terminationTick: r.terminationTick,
    }))
}
