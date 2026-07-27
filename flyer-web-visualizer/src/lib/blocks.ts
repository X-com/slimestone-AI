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

// The ~8 ids the simulator/genetic algorithm actually produce (plus air) get a real color and
// directional flag - these matter for the flat-color fallback path (textures failed to load) and
// the facing nub.
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

  // Every other real Minecraft 1.12 block id, for uploaded/complex example schematics that can
  // contain arbitrary blocks (see schematic_to_stream_json.py) - the simulator/GA themselves never
  // produce these. Generated from the real block registry (PrismarineJS/minecraft-data), not
  // hand-typed. '#999999' is a neutral placeholder color (only used if texture loading fails);
  // the real appearance comes from textures.ts's FACES. None of these are directional - Minecraft's
  // per-block facing storage varies too much block-to-block (rotation vs. half vs. axis, etc.) to
  // fit the single facingVec model FACING_VECTORS already uses for pistons/observers.
  2: { id: 2, name: 'grass', color: '#999999', directional: false }, // Grass Block
  3: { id: 3, name: 'dirt', color: '#999999', directional: false }, // Dirt
  4: { id: 4, name: 'cobblestone', color: '#999999', directional: false }, // Cobblestone
  5: { id: 5, name: 'planks', color: '#999999', directional: false }, // Wood Planks
  7: { id: 7, name: 'bedrock', color: '#999999', directional: false }, // Bedrock
  12: { id: 12, name: 'sand', color: '#999999', directional: false }, // Sand
  13: { id: 13, name: 'gravel', color: '#999999', directional: false }, // Gravel
  14: { id: 14, name: 'gold_ore', color: '#999999', directional: false }, // Gold Ore
  15: { id: 15, name: 'iron_ore', color: '#999999', directional: false }, // Iron Ore
  16: { id: 16, name: 'coal_ore', color: '#999999', directional: false }, // Coal Ore
  17: { id: 17, name: 'log', color: '#999999', directional: false }, // Wood
  18: { id: 18, name: 'leaves', color: '#999999', directional: false }, // Leaves
  19: { id: 19, name: 'sponge', color: '#999999', directional: false }, // Sponge
  21: { id: 21, name: 'lapis_ore', color: '#999999', directional: false }, // Lapis Lazuli Ore
  22: { id: 22, name: 'lapis_block', color: '#999999', directional: false }, // Lapis Lazuli Block
  23: { id: 23, name: 'dispenser', color: '#999999', directional: false }, // Dispenser
  24: { id: 24, name: 'sandstone', color: '#999999', directional: false }, // Sandstone
  25: { id: 25, name: 'noteblock', color: '#999999', directional: false }, // Note Block
  35: { id: 35, name: 'wool', color: '#999999', directional: false }, // Wool
  41: { id: 41, name: 'gold_block', color: '#999999', directional: false }, // Block of Gold
  42: { id: 42, name: 'iron_block', color: '#999999', directional: false }, // Block of Iron
  43: { id: 43, name: 'double_stone_slab', color: '#999999', directional: false }, // Double Stone Slab
  44: { id: 44, name: 'stone_slab', color: '#999999', directional: false }, // Stone Slab
  45: { id: 45, name: 'brick_block', color: '#999999', directional: false }, // Bricks
  46: { id: 46, name: 'tnt', color: '#999999', directional: false }, // TNT
  47: { id: 47, name: 'bookshelf', color: '#999999', directional: false }, // Bookshelf
  48: { id: 48, name: 'mossy_cobblestone', color: '#999999', directional: false }, // Moss Stone
  49: { id: 49, name: 'obsidian', color: '#999999', directional: false }, // Obsidian
  52: { id: 52, name: 'mob_spawner', color: '#999999', directional: false }, // Monster Spawner
  53: { id: 53, name: 'oak_stairs', color: '#999999', directional: false }, // Oak Wood Stairs
  56: { id: 56, name: 'diamond_ore', color: '#999999', directional: false }, // Diamond Ore
  57: { id: 57, name: 'diamond_block', color: '#999999', directional: false }, // Block of Diamond
  58: { id: 58, name: 'crafting_table', color: '#999999', directional: false }, // Crafting Table
  60: { id: 60, name: 'farmland', color: '#999999', directional: false }, // Farmland
  61: { id: 61, name: 'furnace', color: '#999999', directional: false }, // Furnace
  62: { id: 62, name: 'lit_furnace', color: '#999999', directional: false }, // Burning Furnace
  64: { id: 64, name: 'wooden_door', color: '#999999', directional: false }, // Oak Door
  65: { id: 65, name: 'ladder', color: '#999999', directional: false }, // Ladder
  67: { id: 67, name: 'stone_stairs', color: '#999999', directional: false }, // Cobblestone Stairs
  71: { id: 71, name: 'iron_door', color: '#999999', directional: false }, // Iron Door
  73: { id: 73, name: 'redstone_ore', color: '#999999', directional: false }, // Redstone Ore
  74: { id: 74, name: 'lit_redstone_ore', color: '#999999', directional: false }, // Glowing Redstone Ore
  78: { id: 78, name: 'snow_layer', color: '#999999', directional: false }, // Snow
  79: { id: 79, name: 'ice', color: '#999999', directional: false }, // Ice
  80: { id: 80, name: 'snow', color: '#999999', directional: false }, // Snow
  81: { id: 81, name: 'cactus', color: '#999999', directional: false }, // Cactus
  82: { id: 82, name: 'clay', color: '#999999', directional: false }, // Clay
  84: { id: 84, name: 'jukebox', color: '#999999', directional: false }, // Jukebox
  85: { id: 85, name: 'fence', color: '#999999', directional: false }, // Oak Fence
  86: { id: 86, name: 'pumpkin', color: '#999999', directional: false }, // Pumpkin
  87: { id: 87, name: 'netherrack', color: '#999999', directional: false }, // Netherrack
  88: { id: 88, name: 'soul_sand', color: '#999999', directional: false }, // Soul Sand
  89: { id: 89, name: 'glowstone', color: '#999999', directional: false }, // Glowstone
  91: { id: 91, name: 'lit_pumpkin', color: '#999999', directional: false }, // Jack o'Lantern
  92: { id: 92, name: 'cake', color: '#999999', directional: false }, // Cake
  93: { id: 93, name: 'unpowered_repeater', color: '#999999', directional: false }, // Redstone Repeater
  94: { id: 94, name: 'powered_repeater', color: '#999999', directional: false }, // Redstone Repeater
  95: { id: 95, name: 'stained_glass', color: '#999999', directional: false }, // Stained Glass
  96: { id: 96, name: 'trapdoor', color: '#999999', directional: false }, // Wooden Trapdoor
  97: { id: 97, name: 'monster_egg', color: '#999999', directional: false }, // Monster Egg
  98: { id: 98, name: 'stonebrick', color: '#999999', directional: false }, // Stone Bricks
  99: { id: 99, name: 'brown_mushroom_block', color: '#999999', directional: false }, // Mushroom
  100: { id: 100, name: 'red_mushroom_block', color: '#999999', directional: false }, // Mushroom
  101: { id: 101, name: 'iron_bars', color: '#999999', directional: false }, // Iron Bars
  102: { id: 102, name: 'glass_pane', color: '#999999', directional: false }, // Glass Pane
  103: { id: 103, name: 'melon_block', color: '#999999', directional: false }, // Melon
  107: { id: 107, name: 'fence_gate', color: '#999999', directional: false }, // Fence Gate
  108: { id: 108, name: 'brick_stairs', color: '#999999', directional: false }, // Brick Stairs
  109: { id: 109, name: 'stone_brick_stairs', color: '#999999', directional: false }, // Stone Brick Stairs
  110: { id: 110, name: 'mycelium', color: '#999999', directional: false }, // Mycelium
  111: { id: 111, name: 'waterlily', color: '#999999', directional: false }, // Lily Pad
  112: { id: 112, name: 'nether_brick', color: '#999999', directional: false }, // Nether Brick
  113: { id: 113, name: 'nether_brick_fence', color: '#999999', directional: false }, // Nether Brick Fence
  114: { id: 114, name: 'nether_brick_stairs', color: '#999999', directional: false }, // Nether Brick Stairs
  116: { id: 116, name: 'enchanting_table', color: '#999999', directional: false }, // Enchantment Table
  117: { id: 117, name: 'brewing_stand', color: '#999999', directional: false }, // Brewing Stand
  118: { id: 118, name: 'cauldron', color: '#999999', directional: false }, // Cauldron
  120: { id: 120, name: 'end_portal_frame', color: '#999999', directional: false }, // End Portal Frame
  121: { id: 121, name: 'end_stone', color: '#999999', directional: false }, // End Stone
  122: { id: 122, name: 'dragon_egg', color: '#999999', directional: false }, // Dragon Egg
  123: { id: 123, name: 'redstone_lamp', color: '#999999', directional: false }, // Redstone Lamp
  124: { id: 124, name: 'lit_redstone_lamp', color: '#999999', directional: false }, // Redstone Lamp (lit)
  125: { id: 125, name: 'double_wooden_slab', color: '#999999', directional: false }, // Double Wooden Slab
  126: { id: 126, name: 'wooden_slab', color: '#999999', directional: false }, // Wooden Slab
  127: { id: 127, name: 'cocoa', color: '#999999', directional: false }, // Cocoa
  128: { id: 128, name: 'sandstone_stairs', color: '#999999', directional: false }, // Sandstone Stairs
  129: { id: 129, name: 'emerald_ore', color: '#999999', directional: false }, // Emerald Ore
  133: { id: 133, name: 'emerald_block', color: '#999999', directional: false }, // Block of Emerald
  134: { id: 134, name: 'spruce_stairs', color: '#999999', directional: false }, // Spruce Wood Stairs
  135: { id: 135, name: 'birch_stairs', color: '#999999', directional: false }, // Birch Wood Stairs
  136: { id: 136, name: 'jungle_stairs', color: '#999999', directional: false }, // Jungle Wood Stairs
  137: { id: 137, name: 'command_block', color: '#999999', directional: false }, // Command Block
  138: { id: 138, name: 'beacon', color: '#999999', directional: false }, // Beacon
  139: { id: 139, name: 'cobblestone_wall', color: '#999999', directional: false }, // Cobblestone Wall
  140: { id: 140, name: 'flower_pot', color: '#999999', directional: false }, // Flower Pot
  145: { id: 145, name: 'anvil', color: '#999999', directional: false }, // Anvil
  149: { id: 149, name: 'unpowered_comparator', color: '#999999', directional: false }, // Redstone Comparator
  150: { id: 150, name: 'powered_comparator', color: '#999999', directional: false }, // Redstone Comparator (lit)
  151: { id: 151, name: 'daylight_detector', color: '#999999', directional: false }, // Daylight Sensor
  153: { id: 153, name: 'quartz_ore', color: '#999999', directional: false }, // Nether Quartz Ore
  154: { id: 154, name: 'hopper', color: '#999999', directional: false }, // Hopper
  155: { id: 155, name: 'quartz_block', color: '#999999', directional: false }, // Block of Quartz
  156: { id: 156, name: 'quartz_stairs', color: '#999999', directional: false }, // Quartz Stairs
  158: { id: 158, name: 'dropper', color: '#999999', directional: false }, // Dropper
  159: { id: 159, name: 'stained_hardened_clay', color: '#999999', directional: false }, // Stained Clay
  160: { id: 160, name: 'stained_glass_pane', color: '#999999', directional: false }, // Stained Glass Pane
  161: { id: 161, name: 'leaves2', color: '#999999', directional: false }, // Leaves (Acacia/Dark Oak)
  162: { id: 162, name: 'log2', color: '#999999', directional: false }, // Wood (Acacia/Dark Oak)
  163: { id: 163, name: 'acacia_stairs', color: '#999999', directional: false }, // Acacia Wood Stairs
  164: { id: 164, name: 'dark_oak_stairs', color: '#999999', directional: false }, // Dark Oak Wood Stairs
  167: { id: 167, name: 'iron_trapdoor', color: '#999999', directional: false }, // Iron Trapdoor
  168: { id: 168, name: 'prismarine', color: '#999999', directional: false }, // Prismarine
  169: { id: 169, name: 'sea_lantern', color: '#999999', directional: false }, // Sea Lantern
  170: { id: 170, name: 'hay_block', color: '#999999', directional: false }, // Hay Bale
  171: { id: 171, name: 'carpet', color: '#999999', directional: false }, // Carpet
  172: { id: 172, name: 'hardened_clay', color: '#999999', directional: false }, // Hardened Clay
  173: { id: 173, name: 'coal_block', color: '#999999', directional: false }, // Block of Coal
  174: { id: 174, name: 'packed_ice', color: '#999999', directional: false }, // Packed Ice
  178: { id: 178, name: 'daylight_detector_inverted', color: '#999999', directional: false }, // Inverted Daylight Sensor
  179: { id: 179, name: 'red_sandstone', color: '#999999', directional: false }, // Red Sandstone
  180: { id: 180, name: 'red_sandstone_stairs', color: '#999999', directional: false }, // Red Sandstone Stairs
  181: { id: 181, name: 'double_stone_slab2', color: '#999999', directional: false }, // Double Red Sandstone Slab
  182: { id: 182, name: 'stone_slab2', color: '#999999', directional: false }, // Red Sandstone Slab
  183: { id: 183, name: 'spruce_fence_gate', color: '#999999', directional: false }, // Spruce Fence Gate
  184: { id: 184, name: 'birch_fence_gate', color: '#999999', directional: false }, // Birch Fence Gate
  185: { id: 185, name: 'jungle_fence_gate', color: '#999999', directional: false }, // Jungle Fence Gate
  186: { id: 186, name: 'dark_oak_fence_gate', color: '#999999', directional: false }, // Dark Oak Fence Gate
  187: { id: 187, name: 'acacia_fence_gate', color: '#999999', directional: false }, // Acacia Fence Gate
  188: { id: 188, name: 'spruce_fence', color: '#999999', directional: false }, // Spruce Fence
  189: { id: 189, name: 'birch_fence', color: '#999999', directional: false }, // Birch Fence
  190: { id: 190, name: 'jungle_fence', color: '#999999', directional: false }, // Jungle Fence
  191: { id: 191, name: 'dark_oak_fence', color: '#999999', directional: false }, // Dark Oak Fence
  192: { id: 192, name: 'acacia_fence', color: '#999999', directional: false }, // Acacia Fence
  193: { id: 193, name: 'spruce_door', color: '#999999', directional: false }, // Spruce Door
  194: { id: 194, name: 'birch_door', color: '#999999', directional: false }, // Birch Door
  195: { id: 195, name: 'jungle_door', color: '#999999', directional: false }, // Jungle Door
  196: { id: 196, name: 'acacia_door', color: '#999999', directional: false }, // Acacia Door
  197: { id: 197, name: 'dark_oak_door', color: '#999999', directional: false }, // Dark Oak Door
  198: { id: 198, name: 'end_rod', color: '#999999', directional: false }, // End Rod
  199: { id: 199, name: 'chorus_plant', color: '#999999', directional: false }, // Chorus Plant
  200: { id: 200, name: 'chorus_flower', color: '#999999', directional: false }, // Chorus Flower
  201: { id: 201, name: 'purpur_block', color: '#999999', directional: false }, // Purpur Block
  202: { id: 202, name: 'purpur_pillar', color: '#999999', directional: false }, // Purpur Pillar
  203: { id: 203, name: 'purpur_stairs', color: '#999999', directional: false }, // Purpur Stairs
  204: { id: 204, name: 'purpur_double_slab', color: '#999999', directional: false }, // Purpur Double Slab
  205: { id: 205, name: 'purpur_slab', color: '#999999', directional: false }, // Purpur Slab
  206: { id: 206, name: 'end_bricks', color: '#999999', directional: false }, // End Stone Bricks
  208: { id: 208, name: 'grass_path', color: '#999999', directional: false }, // Grass Path
  210: { id: 210, name: 'repeating_command_block', color: '#999999', directional: false }, // Repeating Command Block
  211: { id: 211, name: 'chain_command_block', color: '#999999', directional: false }, // Chain Command Block
  212: { id: 212, name: 'frosted_ice', color: '#999999', directional: false }, // Frosted Ice
  213: { id: 213, name: 'magma', color: '#999999', directional: false }, // Magma Block
  214: { id: 214, name: 'nether_wart_block', color: '#999999', directional: false }, // Nether Wart Block
  215: { id: 215, name: 'red_nether_brick', color: '#999999', directional: false }, // Red Nether Brick
  216: { id: 216, name: 'bone_block', color: '#999999', directional: false }, // Bone Block
  219: { id: 219, name: 'white_shulker_box', color: '#999999', directional: false }, // White Shulker Box
  220: { id: 220, name: 'orange_shulker_box', color: '#999999', directional: false }, // Orange Shulker Box
  221: { id: 221, name: 'magenta_shulker_box', color: '#999999', directional: false }, // Magenta Shulker Box
  222: { id: 222, name: 'light_blue_shulker_box', color: '#999999', directional: false }, // Light Blue Shulker Box
  223: { id: 223, name: 'yellow_shulker_box', color: '#999999', directional: false }, // Yellow Shulker Box
  224: { id: 224, name: 'lime_shulker_box', color: '#999999', directional: false }, // Lime Shulker Box
  225: { id: 225, name: 'pink_shulker_box', color: '#999999', directional: false }, // Pink Shulker Box
  226: { id: 226, name: 'gray_shulker_box', color: '#999999', directional: false }, // Gray Shulker Box
  227: { id: 227, name: 'light_gray_shulker_box', color: '#999999', directional: false }, // Light Gray Shulker Box
  228: { id: 228, name: 'cyan_shulker_box', color: '#999999', directional: false }, // Cyan Shulker Box
  229: { id: 229, name: 'purple_shulker_box', color: '#999999', directional: false }, // Purple Shulker Box
  230: { id: 230, name: 'blue_shulker_box', color: '#999999', directional: false }, // Blue Shulker Box
  231: { id: 231, name: 'brown_shulker_box', color: '#999999', directional: false }, // Brown Shulker Box
  232: { id: 232, name: 'green_shulker_box', color: '#999999', directional: false }, // Green Shulker Box
  233: { id: 233, name: 'red_shulker_box', color: '#999999', directional: false }, // Red Shulker Box
  234: { id: 234, name: 'black_shulker_box', color: '#999999', directional: false }, // Black Shulker Box
  235: { id: 235, name: 'white_glazed_terracotta', color: '#999999', directional: false }, // White Glazed Terracotta
  236: { id: 236, name: 'orange_glazed_terracotta', color: '#999999', directional: false }, // Orange Glazed Terracotta
  237: { id: 237, name: 'magenta_glazed_terracotta', color: '#999999', directional: false }, // Magenta Glazed Terracotta
  238: { id: 238, name: 'light_blue_glazed_terracotta', color: '#999999', directional: false }, // Light Blue Glazed Terracotta
  239: { id: 239, name: 'yellow_glazed_terracotta', color: '#999999', directional: false }, // Yellow Glazed Terracotta
  240: { id: 240, name: 'lime_glazed_terracotta', color: '#999999', directional: false }, // Lime Glazed Terracotta
  241: { id: 241, name: 'pink_glazed_terracotta', color: '#999999', directional: false }, // Pink Glazed Terracotta
  242: { id: 242, name: 'gray_glazed_terracotta', color: '#999999', directional: false }, // Gray Glazed Terracotta
  243: { id: 243, name: 'light_gray_glazed_terracotta', color: '#999999', directional: false }, // Light Gray Glazed Terracotta
  244: { id: 244, name: 'cyan_glazed_terracotta', color: '#999999', directional: false }, // Cyan Glazed Terracotta
  245: { id: 245, name: 'purple_glazed_terracotta', color: '#999999', directional: false }, // Purple Glazed Terracotta
  246: { id: 246, name: 'blue_glazed_terracotta', color: '#999999', directional: false }, // Blue Glazed Terracotta
  247: { id: 247, name: 'brown_glazed_terracotta', color: '#999999', directional: false }, // Brown Glazed Terracotta
  248: { id: 248, name: 'green_glazed_terracotta', color: '#999999', directional: false }, // Green Glazed Terracotta
  249: { id: 249, name: 'red_glazed_terracotta', color: '#999999', directional: false }, // Red Glazed Terracotta
  250: { id: 250, name: 'black_glazed_terracotta', color: '#999999', directional: false }, // Black Glazed Terracotta
  251: { id: 251, name: 'concrete', color: '#999999', directional: false }, // Concrete
  252: { id: 252, name: 'concrete_powder', color: '#999999', directional: false }, // Concrete Powder
  255: { id: 255, name: 'structure_block', color: '#999999', directional: false }, // Structure Block
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
