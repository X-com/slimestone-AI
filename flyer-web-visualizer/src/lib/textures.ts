// Builds a small texture atlas from public/textures/*.png (populated by
// scripts/sync-textures.sh: DABB preferred, vanilla fallback) and hands the scene
// a per-block-type geometry (per-face UVs baked into the atlas) + material.
//
// Faces use BoxGeometry material order [+x, -x, +y, -y, +z, -z]. Directional blocks
// put their "front" texture on +Z; the scene rotates each instance so +Z aligns with
// the block's facing. Glass is alpha-tested (opaque frame / clear pane), slime is
// translucent; everything else is opaque.
import * as THREE from 'three'

const TEX_NAMES = [
  'stone', 'glass', 'slime', 'redstone_block',
  'piston_top', 'piston_top_sticky', 'piston_side', 'piston_side_sticky',
  'piston_bottom', 'piston_bottom_sticky',
  'observer_front', 'observer_back', 'observer_side', 'observer_top',
] as const
type TexName = (typeof TEX_NAMES)[number]

// Face textures in BoxGeometry material order [+x, -x, +y, -y, +z, -z].
// A block's "front" (the head/face) sits on the axis returned by frontAxis(); the scene
// rotates each instance so that axis aligns with the block's facing direction. Pistons put
// their head on +Y so all four *vertical* faces are the "sides" and share the same upward
// texture orientation (avoids the side texture looking rotated/garbled face-to-face).
const FACES: Record<number, [TexName, TexName, TexName, TexName, TexName, TexName]> = {
  1: ['stone', 'stone', 'stone', 'stone', 'stone', 'stone'],
  20: ['glass', 'glass', 'glass', 'glass', 'glass', 'glass'],
  165: ['slime', 'slime', 'slime', 'slime', 'slime', 'slime'],
  152: ['redstone_block', 'redstone_block', 'redstone_block', 'redstone_block', 'redstone_block', 'redstone_block'],
  // pistons: head on +Y, back on -Y, piston_side on the 4 vertical faces
  33: ['piston_side', 'piston_side', 'piston_top', 'piston_bottom', 'piston_side', 'piston_side'],
  29: ['piston_side_sticky', 'piston_side_sticky', 'piston_top_sticky', 'piston_bottom_sticky', 'piston_side_sticky', 'piston_side_sticky'],
  34: ['piston_side', 'piston_side', 'piston_top', 'piston_side', 'piston_side', 'piston_side'],
  // observer: front (eyes) on +Z, back (output) on -Z, arrow textures on the other four
  218: ['observer_side', 'observer_side', 'observer_top', 'observer_top', 'observer_front', 'observer_back'],
}

// Which world axis a block's "front" points to before per-instance facing rotation.
const AXIS_Y = new THREE.Vector3(0, 1, 0)
const AXIS_Z = new THREE.Vector3(0, 0, 1)
export function frontAxis(blockId: number): THREE.Vector3 | null {
  if (blockId === 33 || blockId === 29 || blockId === 34) return AXIS_Y // pistons
  if (blockId === 218) return AXIS_Z // observer
  return null // non-directional: no rotation
}

const CUTOUT = new Set([20]) // glass: opaque frame, clear centre -> alphaTest
const TRANSLUCENT = new Set([165]) // slime: semi-transparent

const N = 4 // atlas grid (4x4 = 16 cells, 14 used)
const CELL = 16

export interface BlockAssets {
  geo(blockId: number): THREE.BufferGeometry
  material(blockId: number): THREE.Material
  dispose(): void
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`failed to load ${src}`))
    img.src = src
  })
}

function cellOf(name: TexName): { col: number; row: number } {
  const i = TEX_NAMES.indexOf(name)
  return { col: i % N, row: Math.floor(i / N) }
}

export async function loadBlockAssets(base: string): Promise<BlockAssets> {
  const imgs = await Promise.all(TEX_NAMES.map((n) => loadImage(`${base}textures/${n}.png`)))

  const canvas = document.createElement('canvas')
  canvas.width = N * CELL
  canvas.height = N * CELL
  const ctx = canvas.getContext('2d')!
  ctx.imageSmoothingEnabled = false
  imgs.forEach((img, i) => {
    const col = i % N
    const row = Math.floor(i / N)
    const frame = Math.min(img.width, img.height) // first frame if animated (tall strip)
    ctx.drawImage(img, 0, 0, img.width, frame, col * CELL, row * CELL, CELL, CELL)
  })

  const atlas = new THREE.CanvasTexture(canvas)
  atlas.magFilter = THREE.NearestFilter
  atlas.minFilter = THREE.NearestFilter
  atlas.generateMipmaps = false
  atlas.colorSpace = THREE.SRGBColorSpace
  atlas.wrapS = atlas.wrapT = THREE.ClampToEdgeWrapping

  const PAD = 0.5 / CELL // half-texel inset so nearest sampling never bleeds into the next cell
  function bake(faces: readonly TexName[]): THREE.BufferGeometry {
    const geo = new THREE.BoxGeometry(1, 1, 1)
    const uv = geo.attributes.uv as THREE.BufferAttribute
    for (let f = 0; f < 6; f++) {
      const { col, row } = cellOf(faces[f])
      for (let k = 0; k < 4; k++) {
        const idx = f * 4 + k
        const fu = PAD + uv.getX(idx) * (1 - 2 * PAD)
        const fv = PAD + uv.getY(idx) * (1 - 2 * PAD)
        uv.setXY(idx, (col + fu) / N, (N - 1 - row + fv) / N)
      }
    }
    uv.needsUpdate = true
    return geo
  }

  const geos = new Map<number, THREE.BufferGeometry>()
  for (const id of Object.keys(FACES)) geos.set(Number(id), bake(FACES[Number(id)]))
  const fallbackGeo = bake(FACES[1]) // unknown ids render as stone

  const matOpaque = new THREE.MeshLambertMaterial({ map: atlas })
  const matCutout = new THREE.MeshLambertMaterial({ map: atlas, alphaTest: 0.5 })
  const matTranslucent = new THREE.MeshLambertMaterial({
    map: atlas,
    transparent: true,
    depthWrite: false, // avoid translucent panels blocking each other's depth
  })

  return {
    geo: (id) => geos.get(id) ?? fallbackGeo,
    material: (id) =>
      CUTOUT.has(id) ? matCutout : TRANSLUCENT.has(id) ? matTranslucent : matOpaque,
    dispose() {
      atlas.dispose()
      for (const g of geos.values()) g.dispose()
      fallbackGeo.dispose()
      matOpaque.dispose()
      matCutout.dispose()
      matTranslucent.dispose()
    },
  }
}
