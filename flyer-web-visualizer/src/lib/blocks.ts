// Block palette + state decoding.
// Source of truth: genetic-ml/genetic_ml/blocks.py
//   state = block_id | (meta << 8)
//   block_id = state & 0xFF ; meta = state >> 8
//   facing   = meta & 0b111  -> [down, up, north, south, west, east]
//   meta & 8 -> piston extended / observer powered (runtime state)

export type Facing = 'down' | 'up' | 'north' | 'south' | 'west' | 'east'

export const FACINGS: readonly Facing[] = [
  'down',
  'up',
  'north',
  'south',
  'west',
  'east',
] as const

// Facing unit vectors already in three.js space (Minecraft +z=south mapped to -z).
// MC: down(0,-1,0) up(0,1,0) north(0,0,-1) south(0,0,1) west(-1,0,0) east(1,0,0)
// -> negate z to go right-handed:
export const FACING_VECTORS: Record<Facing, [number, number, number]> = {
  down: [0, -1, 0],
  up: [0, 1, 0],
  north: [0, 0, 1], // MC north = -z, negated -> +z
  south: [0, 0, -1], // MC south = +z, negated -> -z
  west: [-1, 0, 0],
  east: [1, 0, 0],
}

export interface BlockType {
  id: number
  name: string
  color: string // hex, used for the instanced cube
  directional: boolean // shows a facing nub
}

// Only the ~8 ids that actually appear in ga_archive.jsonl (plus air).
export const BLOCK_TYPES: Record<number, BlockType> = {
  0: { id: 0, name: 'air', color: '#000000', directional: false },
  1: { id: 1, name: 'stone', color: '#8a8d91', directional: false },
  20: { id: 20, name: 'glass', color: '#bfe3ef', directional: false },
  29: { id: 29, name: 'sticky_piston', color: '#6f9e4a', directional: true },
  33: { id: 33, name: 'piston', color: '#c2a86b', directional: true },
  34: { id: 34, name: 'piston_head', color: '#a8905a', directional: true },
  152: { id: 152, name: 'redstone_block', color: '#c1332c', directional: false },
  165: { id: 165, name: 'slime', color: '#7bd45a', directional: false },
  218: { id: 218, name: 'observer', color: '#556172', directional: true },
}

const UNKNOWN: BlockType = {
  id: -1,
  name: 'unknown',
  color: '#ff00ff',
  directional: false,
}

export interface DecodedBlock {
  blockId: number
  meta: number
  type: BlockType
  facing: Facing
  facingVec: [number, number, number]
  extended: boolean // meta & 8
}

export function decodeState(state: number): DecodedBlock {
  const blockId = state & 0xff
  const meta = state >> 8
  const facing = FACINGS[meta & 0b111] ?? 'down'
  return {
    blockId,
    meta,
    type: BLOCK_TYPES[blockId] ?? UNKNOWN,
    facing,
    facingVec: FACING_VECTORS[facing],
    extended: (meta & 8) !== 0,
  }
}
