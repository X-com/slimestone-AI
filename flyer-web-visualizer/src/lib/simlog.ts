// Reader for the C++ simulator's binary "simulation_data" event log (see
// cpp simulator/src/sim_event_log.h - the authoritative layout, mirrored here byte-for-byte;
// keep both in sync). Same manual DataView-offset style as parseCompactData in data.ts.
//
// Layout: [ events (72B each, grouped per block, in sim order) ][ block index (32B each) ]
//         [ 48B footer at EOF ]. No animation/playback here - this only supports the hover
//         inspector: for a given block key, return its complete ordered event list.
//
// 64-bit fields use bigint (DataView getBigUint64/getBigInt64) - packed position keys can
// exceed Number.MAX_SAFE_INTEGER, and bigint is also what packPos()/unpackPos() below need to
// exactly mirror packed_pos.h's 21-bit-per-axis packing.

export const EVENT_SIZE = 72
export const BLOCK_INDEX_SIZE = 32
export const FOOTER_SIZE = 48

export const enum SimEventKind {
  PistonQueued = 0,
  PistonMoveExecuted = 1,
  BlockPushed = 2,
  ObserverFired = 3,
  ObserverActivated = 4,
  RedstoneBlockAppeared = 5,
  RedstoneBlockRemoved = 6,
  RedstoneActivatedPiston = 7,
  RedstoneDeactivatedPiston = 8,
}

export const SEF_EXTEND = 1 << 0
export const SEF_SUCCESS = 1 << 1
export const SEF_TARGET_PISTON = 1 << 4

export const NO_DIRECTION = 0xff

export interface SimEvent {
  blockKey: bigint
  actorKey: bigint
  targetKey: bigint
  activationTick: bigint
  scheduledTick: bigint
  executedTick: bigint
  activationSubtick: number
  scheduledSubtick: number
  executedSubtick: number
  pushGroupId: number
  kind: SimEventKind
  direction: number
  flags: number
  attemptedAmount: number
  actualAmount: number
}

export interface BlockIndexEntry {
  originalKey: bigint
  currentKey: bigint
  firstEventIdx: number
  eventCount: number
  originalState: number
}

export interface SimLog {
  eventCount: number
  blockCount: number
  eventsByKey: Map<bigint, SimEvent[]>
  indexByKey: Map<bigint, BlockIndexEntry>
}

// Mirrors packed_pos.h exactly: 21 signed bits per axis, x | (y<<21) | (z<<42).
export function packPos(x: number, y: number, z: number): bigint {
  const px = BigInt.asUintN(21, BigInt(x))
  const py = BigInt.asUintN(21, BigInt(y))
  const pz = BigInt.asUintN(21, BigInt(z))
  return px | (py << 21n) | (pz << 42n)
}

export function unpackPos(key: bigint): [number, number, number] {
  const unpack21 = (v: bigint): number => Number(BigInt.asIntN(21, v & 0x1fffffn))
  return [unpack21(key), unpack21(key >> 21n), unpack21(key >> 42n)]
}

function readEvent(view: DataView, off: number): SimEvent {
  return {
    blockKey: view.getBigUint64(off + 0, true),
    actorKey: view.getBigUint64(off + 8, true),
    targetKey: view.getBigUint64(off + 16, true),
    activationTick: view.getBigInt64(off + 24, true),
    scheduledTick: view.getBigInt64(off + 32, true),
    executedTick: view.getBigInt64(off + 40, true),
    activationSubtick: view.getUint32(off + 48, true),
    scheduledSubtick: view.getUint32(off + 52, true),
    executedSubtick: view.getUint32(off + 56, true),
    pushGroupId: view.getUint32(off + 60, true),
    kind: view.getUint8(off + 64),
    direction: view.getUint8(off + 65),
    flags: view.getUint8(off + 66),
    attemptedAmount: view.getUint8(off + 67),
    actualAmount: view.getUint8(off + 68),
  }
}

function readBlockIndexEntry(view: DataView, off: number): BlockIndexEntry {
  return {
    originalKey: view.getBigUint64(off + 0, true),
    currentKey: view.getBigUint64(off + 8, true),
    firstEventIdx: view.getUint32(off + 16, true),
    eventCount: view.getUint32(off + 20, true),
    originalState: view.getUint32(off + 24, true),
  }
}

export function parseSimLog(buf: ArrayBuffer): SimLog {
  const view = new DataView(buf)
  if (buf.byteLength < FOOTER_SIZE) throw new Error('simlog: file too small')
  const footerOff = buf.byteLength - FOOTER_SIZE
  const magic = String.fromCharCode(
    view.getUint8(footerOff), view.getUint8(footerOff + 1),
    view.getUint8(footerOff + 2), view.getUint8(footerOff + 3),
  )
  if (magic !== 'SDL2') throw new Error(`simlog: bad magic ${magic}`)
  // Footer layout (48B, packed): magic[4] version(u32) eventCount(u64) blockIndexOffset(u64)
  // blockCount(u32) eventRecSize(u32) blockRecSize(u32) reserved(u32) reserved2(u64).
  const eventCount = Number(view.getBigUint64(footerOff + 8, true))
  const blockIndexOffset = Number(view.getBigUint64(footerOff + 16, true))
  const blockCount = view.getUint32(footerOff + 24, true)
  const eventRecSize = view.getUint32(footerOff + 28, true)
  if (eventRecSize !== EVENT_SIZE) throw new Error(`simlog: record size mismatch ${eventRecSize}`)

  const indexByKey = new Map<bigint, BlockIndexEntry>()
  for (let i = 0; i < blockCount; i++) {
    const entry = readBlockIndexEntry(view, blockIndexOffset + i * BLOCK_INDEX_SIZE)
    indexByKey.set(entry.originalKey, entry)
  }

  const eventsByKey = new Map<bigint, SimEvent[]>()
  for (const entry of indexByKey.values()) {
    const events: SimEvent[] = []
    for (let i = 0; i < entry.eventCount; i++) {
      events.push(readEvent(view, (entry.firstEventIdx + i) * EVENT_SIZE))
    }
    eventsByKey.set(entry.originalKey, events)
  }

  return { eventCount, blockCount, eventsByKey, indexByKey }
}

const KIND_NAMES: Record<number, string> = {
  0: 'PistonQueued', 1: 'PistonMoveExecuted', 2: 'BlockPushed', 3: 'ObserverFired',
  4: 'ObserverActivated', 5: 'RedstoneBlockAppeared', 6: 'RedstoneBlockRemoved',
  7: 'RedstoneActivatedPiston', 8: 'RedstoneDeactivatedPiston',
}
const CAUSE_NAMES: Record<number, string> = { 0: 'scheduled', 1: 'facing-changed', 2: 'observer-moved' }
const DIR_NAMES: Record<number, string> = {
  0: 'DOWN', 1: 'UP', 2: 'NORTH', 3: 'SOUTH', 4: 'WEST', 5: 'EAST', 255: '-',
}

function posStr(key: bigint): string {
  const [x, y, z] = unpackPos(key)
  return `(${x},${y},${z})`
}

// One human-readable line per event, mirroring verify_simulation_data.py's _fmt_event so the
// hover tooltip and the Python dump describe the same record the same way.
export function formatEvent(ev: SimEvent): string {
  const kind = KIND_NAMES[ev.kind] ?? `?${ev.kind}`
  const parts = [`t=${ev.activationTick} s=${ev.activationSubtick} ${kind}`]
  if (ev.kind === SimEventKind.PistonQueued || ev.kind === SimEventKind.PistonMoveExecuted || ev.kind === SimEventKind.BlockPushed) {
    parts.push(ev.flags & SEF_EXTEND ? 'extend' : 'retract')
    parts.push(`dir=${DIR_NAMES[ev.direction] ?? ev.direction}`)
    if (ev.kind === SimEventKind.PistonMoveExecuted) parts.push(ev.flags & SEF_SUCCESS ? 'moved' : 'BLOCKED')
    if (ev.kind === SimEventKind.BlockPushed) parts.push(`by piston${posStr(ev.actorKey)}->${posStr(ev.targetKey)}`)
    parts.push(`amt ${ev.attemptedAmount}->${ev.actualAmount}`)
    parts.push(`grp=${ev.pushGroupId}`)
    parts.push(`sched(t=${ev.scheduledTick},s=${ev.scheduledSubtick})`)
  } else if (ev.kind === SimEventKind.ObserverFired) {
    parts.push(`cause=${CAUSE_NAMES[(ev.flags >> 2) & 3] ?? '?'}`)
  } else if (ev.kind === SimEventKind.ObserverActivated) {
    parts.push(`-> ${ev.flags & SEF_TARGET_PISTON ? 'piston' : 'observer'}${posStr(ev.targetKey)}`)
  } else if (ev.kind === SimEventKind.RedstoneActivatedPiston || ev.kind === SimEventKind.RedstoneDeactivatedPiston) {
    parts.push(`-> piston${posStr(ev.targetKey)}`)
  }
  return parts.join(' ')
}
