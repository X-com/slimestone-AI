// Imperative three.js layer: one textured InstancedMesh per block type (facing baked
// into per-instance rotation), CSS2D hash labels, orbit + pan + snap views, click-to-select.
// Kept out of Svelte reactivity on purpose.
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  CSS2DRenderer,
  CSS2DObject,
} from 'three/examples/jsm/renderers/CSS2DRenderer.js'
import { decodeState } from './blocks'
import { BLOCK_TYPES } from './blocks'
import { loadBlockAssets, frontAxis, type BlockAssets } from './textures'
import type { Machine } from './data'

const GAP = 3 // empty cells between machines

export interface SceneHandle {
  // keepCamera: after the first build, leave the camera where it is instead of re-framing
  // (used by the live dashboard so a new batch / page swap doesn't snap the view).
  setMachines(
    machines: Machine[],
    onSelect: (m: Machine | null) => void,
    keepCamera?: boolean,
  ): void
  setSelected(hash: string | null): void
  focusSelected(hash: string): void // recenter + zoom-to-fit a machine (arrow navigation)
  setElevation(elevationDeg: number): void // snap up/down, keeping current azimuth
  resetView(): void // re-frame the whole current set (undo any orbit/zoom)
  dispose(): void
}

interface Placed {
  machine: Machine
  box: THREE.Box3 // world-space bounding box, for the selection outline
  label: CSS2DObject
}

export function createScene(container: HTMLElement): SceneHandle {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color('#0f111a')

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 5000)
  camera.position.set(30, 40, 60)

  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  container.appendChild(renderer.domElement)

  const labelRenderer = new CSS2DRenderer()
  labelRenderer.domElement.style.position = 'absolute'
  labelRenderer.domElement.style.top = '0'
  labelRenderer.domElement.style.left = '0'
  labelRenderer.domElement.style.pointerEvents = 'none' // labels re-enable per element
  container.appendChild(labelRenderer.domElement)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.enablePan = true
  controls.screenSpacePanning = true // pan in the view plane (intuitive right-drag / two-finger)

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.15))
  const dir = new THREE.DirectionalLight(0xffffff, 1.1)
  dir.position.set(1, 2, 1.5)
  scene.add(dir)

  const selectionBox = new THREE.Box3Helper(new THREE.Box3(), 0x22d3ee)
  selectionBox.visible = false
  scene.add(selectionBox)

  // Trigger marker: the GA-specified simulation start block, as a translucent purple glow.
  const triggerGeo = new THREE.BoxGeometry(1, 1, 1)
  const triggerMat = new THREE.MeshBasicMaterial({
    color: 0xb14aff,
    transparent: true,
    opacity: 0.5, // ponytail: 0.1 alpha as requested; additive blend does the "glow"
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })
  let triggerMesh: THREE.InstancedMesh | null = null

  // Block assets (texture atlas + per-type geometry/material). Loads async; until then
  // (or if it fails) blocks fall back to flat colors.
  let assets: BlockAssets | null = null
  const assetsReady = loadBlockAssets(import.meta.env.BASE_URL)
    .then((a) => {
      assets = a
    })
    .catch((err) => console.warn('textures unavailable, using flat colors:', err))
  const plainBox = new THREE.BoxGeometry(1, 1, 1)
  const coloredMats = new Map<number, THREE.MeshLambertMaterial>()
  function coloredMat(id: number): THREE.MeshLambertMaterial {
    let m = coloredMats.get(id)
    if (!m) {
      m = new THREE.MeshLambertMaterial({ color: BLOCK_TYPES[id]?.color ?? '#ff00ff' })
      coloredMats.set(id, m)
    }
    return m
  }

  let blockMeshes: THREE.InstancedMesh[] = []
  let placed: Placed[] = []
  let onSelectCb: (m: Machine | null) => void = () => {}
  let selectedHash: string | null = null
  let hasFramed = false // whether the camera has been auto-framed at least once
  let lastFrame: [cols: number, n: number, cell: number] | null = null // for resetView()
  let buildToken = 0

  const dummy = new THREE.Object3D()
  const quat = new THREE.Quaternion()
  const facing = new THREE.Vector3()
  const raycaster = new THREE.Raycaster()
  const pointer = new THREE.Vector2()
  const _tmp = new THREE.Vector3()
  const _dir = new THREE.Vector3()

  function clearMeshes() {
    for (const p of placed) {
      p.label.element.remove() // CSS2DRenderer won't drop stale DOM on its own
      scene.remove(p.label)
    }
    placed = []
    for (const mesh of blockMeshes) {
      scene.remove(mesh)
      mesh.dispose() // frees instance buffers only; shared geo/material are reused
    }
    blockMeshes = []
    if (triggerMesh) {
      scene.remove(triggerMesh)
      triggerMesh.dispose()
      triggerMesh = null
    }
  }

  async function setMachines(
    machines: Machine[],
    onSelect: (m: Machine | null) => void,
    keepCamera = false,
  ) {
    const token = ++buildToken
    onSelectCb = onSelect
    await assetsReady // first call waits for the atlas; later calls resolve instantly
    if (token !== buildToken) return // superseded by a newer setMachines
    clearMeshes()

    // Uniform grid cell = widest footprint in this set + gap.
    let maxFoot = 1
    for (const m of machines) {
      const b = m.candidate.blocks
      const xs = b.map((v) => v.x)
      const zs = b.map((v) => v.z)
      maxFoot = Math.max(
        maxFoot,
        Math.max(...xs) - Math.min(...xs) + 1,
        Math.max(...zs) - Math.min(...zs) + 1,
      )
    }
    const cell = maxFoot + GAP
    const cols = Math.max(1, Math.ceil(Math.sqrt(machines.length)))

    // One instanced draw per block id; collect placements first, then build meshes.
    const byId = new Map<number, { machine: Machine; matrix: THREE.Matrix4 }[]>()
    const triggerMats: THREE.Matrix4[] = []

    machines.forEach((m, idx) => {
      const blocks = m.candidate.blocks
      const xs = blocks.map((v) => v.x)
      const ys = blocks.map((v) => v.y)
      const zs = blocks.map((v) => v.z)
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2
      const cy = Math.min(...ys)
      const cz = (Math.min(...zs) + Math.max(...zs)) / 2

      const gx = (idx % cols) * cell
      const gz = Math.floor(idx / cols) * cell

      // Trigger block (simulation start), same MC->three transform as blocks.
      const tr = m.candidate.trigger
      triggerMats.push(
        new THREE.Matrix4().makeTranslation(
          gx + (tr.x - cx),
          tr.y - cy + 0.5,
          gz - (tr.z - cz),
        ),
      )

      const mBox = new THREE.Box3()
      for (const blk of blocks) {
        const d = decodeState(blk.state)
        // MC -> three: y up, negate z for right-handed, center within cell.
        const wx = gx + (blk.x - cx)
        const wy = blk.y - cy + 0.5
        const wz = gz - (blk.z - cz)

        const axis = frontAxis(d.blockId)
        if (axis) {
          facing.set(d.facingVec[0], d.facingVec[1], d.facingVec[2])
          quat.setFromUnitVectors(axis, facing) // rotate the "front" face toward facing
        } else {
          quat.identity()
        }
        dummy.position.set(wx, wy, wz)
        dummy.quaternion.copy(quat)
        dummy.scale.set(1, 1, 1)
        dummy.updateMatrix()

        const list = byId.get(d.blockId) ?? []
        list.push({ machine: m, matrix: dummy.matrix.clone() })
        byId.set(d.blockId, list)

        mBox.expandByPoint(new THREE.Vector3(wx - 0.5, wy - 0.5, wz - 0.5))
        mBox.expandByPoint(new THREE.Vector3(wx + 0.5, wy + 0.5, wz + 0.5))
      }

      // Hash / id label above the machine.
      const el = document.createElement('div')
      el.className = 'machine-label'
      el.textContent = m.label ?? m.hash.slice(0, 8) + '…'
      el.title = m.source ? `${m.source}  #${m.candidate.id}` : m.hash
      el.style.pointerEvents = 'auto'
      el.addEventListener('click', (e) => {
        e.stopPropagation()
        onSelectCb(m)
      })
      const label = new CSS2DObject(el)
      label.position.set(gx, mBox.max.y + 1.2, gz)
      scene.add(label)

      placed.push({ machine: m, box: mBox, label })
    })

    for (const [id, list] of byId) {
      const geo = assets ? assets.geo(id) : plainBox
      const mat = assets ? assets.material(id) : coloredMat(id)
      const mesh = new THREE.InstancedMesh(geo, mat, list.length)
      mesh.userData.machines = list.map((e) => e.machine)
      list.forEach((e, i) => mesh.setMatrixAt(i, e.matrix))
      mesh.instanceMatrix.needsUpdate = true
      // A packed per-block-type mesh spans the whole field, so per-object frustum culling just
      // pops the entire layer in/out when its grid-wide bounds leave the view — no draw-call win
      // (packing already gave that). Keep culling off; the pop-out was the visible regression.
      mesh.frustumCulled = false
      scene.add(mesh)
      blockMeshes.push(mesh)
    }

    // Trigger glows (one instance per machine); not raycast so it never steals clicks.
    triggerMesh = new THREE.InstancedMesh(triggerGeo, triggerMat, triggerMats.length)
    triggerMesh.renderOrder = 2 // after opaque blocks
    triggerMats.forEach((mtx, i) => triggerMesh!.setMatrixAt(i, mtx))
    triggerMesh.instanceMatrix.needsUpdate = true
    triggerMesh.frustumCulled = false
    scene.add(triggerMesh)

    lastFrame = [cols, machines.length, cell] // remembered so resetView() can re-frame on demand
    // Frame on the first build; afterwards honour keepCamera so live updates don't snap the view.
    if (!keepCamera || !hasFramed) {
      frameAll(cols, machines.length, cell)
      hasFramed = true
    }
    setSelected(selectedHash) // reattach any current selection to the rebuilt meshes
  }

  function frameAll(cols: number, n: number, cell: number) {
    const rows = Math.max(1, Math.ceil(n / cols))
    const w = cols * cell
    const d = rows * cell
    const cx = w / 2 - cell / 2
    const cz = d / 2 - cell / 2
    const radius = Math.max(w, d)
    // Iso-ish offset direction from the target.
    _dir.set(radius * 0.5, radius * 0.7 + 8, radius * 0.75)
    const baseLen = _dir.length()
    // Aspect-aware fit: the grid's half-extent must fit the *smaller* of the horizontal/vertical
    // FOV so a short/wide viewport (the dashboard's top strip) doesn't clip machines.
    const R = 0.5 * Math.hypot(w, d) + cell
    const vHalf = (camera.fov * Math.PI) / 360
    const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.0001, camera.aspect))
    const fitDist = R / Math.sin(Math.max(0.01, Math.min(vHalf, hHalf)))
    // Frame 20% closer than the base, but never nearer than the fit distance (else it clips).
    const dist = Math.max(baseLen * 0.8, fitDist)
    _dir.setLength(dist)
    controls.target.set(cx, 2, cz)
    camera.position.set(cx + _dir.x, 2 + _dir.y, cz + _dir.z)
    camera.far = dist * 6 + 100
    camera.updateProjectionMatrix()
    controls.update()
  }

  // Re-frame the whole current set (undo any manual orbit/zoom).
  function resetView() {
    if (lastFrame) frameAll(lastFrame[0], lastFrame[1], lastFrame[2])
  }

  // Snap to a target elevation, keeping the current azimuth + zoom distance.
  function setElevation(elevationDeg: number) {
    const off = _dir.copy(camera.position).sub(controls.target)
    const dist = Math.max(1, off.length())
    const az = Math.atan2(off.x, off.z) // preserve current azimuth
    const el = (elevationDeg * Math.PI) / 180
    camera.position.set(
      controls.target.x + Math.cos(el) * Math.sin(az) * dist,
      controls.target.y + Math.sin(el) * dist,
      controls.target.z + Math.cos(el) * Math.cos(az) * dist,
    )
    camera.up.set(0, 1, 0)
    controls.update()
  }

  // Recenter the view on a machine with a gentle pan — same angle AND same zoom distance, so
  // stepping through machines just glides the pivot over instead of snapping the zoom. (arrow nav)
  function focusSelected(hash: string) {
    const hit = placed.find((p) => p.machine.hash === hash)
    if (!hit) return
    const center = hit.box.getCenter(_tmp)
    camera.position.add(_dir.subVectors(center, controls.target))
    controls.target.copy(center)
    controls.update()
  }

  function setSelected(hash: string | null) {
    selectedHash = hash
    const hit = placed.find((p) => p.machine.hash === hash)
    if (hit) {
      selectionBox.box.copy(hit.box)
      selectionBox.visible = true
      // Highlight only — clicking to inspect must not move the camera. Use the arrow keys /
      // ◀ ▶ to recenter on a machine (focusSelected).
    } else {
      selectionBox.visible = false
    }
    for (const p of placed)
      p.label.element.classList.toggle('selected', p.machine.hash === hash)
  }

  // Click-to-pick (ignore orbit/pan drags).
  let downX = 0
  let downY = 0
  renderer.domElement.addEventListener('pointerdown', (e) => {
    downX = e.clientX
    downY = e.clientY
  })
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) return // was a drag
    if (!blockMeshes.length) return
    const rect = renderer.domElement.getBoundingClientRect()
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
    raycaster.setFromCamera(pointer, camera)
    raycaster.far = Infinity // occlusion pass shrinks .far; restore for picking
    const hits = raycaster.intersectObjects(blockMeshes, false)
    const hit = hits[0]
    if (hit && hit.instanceId != null) {
      const machines = hit.object.userData.machines as Machine[]
      onSelectCb(machines[hit.instanceId] ?? null)
    } else {
      onSelectCb(null)
    }
  })

  // Hide a machine's floating label when blocks sit between it and the camera. CSS2D labels
  // have no depth test, so we raycast from the camera to each label anchor and occlude.
  // ponytail: O(labels * instances) per moving frame; throttled to ~15Hz, fine for <=400 labels.
  let lastOcc = 0
  let needsOcclusion = true
  controls.addEventListener('change', () => (needsOcclusion = true))
  function updateOcclusion() {
    for (const p of placed) {
      _dir.subVectors(p.label.position, camera.position)
      const dist = _dir.length()
      raycaster.set(camera.position, _dir.normalize())
      raycaster.far = Math.max(0.1, dist - 0.6) // only blocks in front of the anchor
      p.label.visible = raycaster.intersectObjects(blockMeshes, false).length === 0
    }
  }

  let raf = 0
  function animate() {
    raf = requestAnimationFrame(animate)
    controls.update()
    const now = performance.now()
    if (needsOcclusion && now - lastOcc > 66) {
      updateOcclusion()
      needsOcclusion = false
      lastOcc = now
    }
    renderer.render(scene, camera)
    labelRenderer.render(scene, camera)
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = container
    if (!w || !h) return
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
    labelRenderer.setSize(w, h)
  }
  const ro = new ResizeObserver(resize)
  ro.observe(container)
  resize()
  animate()

  return {
    setMachines,
    setSelected,
    focusSelected,
    setElevation,
    resetView,
    dispose() {
      cancelAnimationFrame(raf)
      ro.disconnect()
      clearMeshes()
      controls.dispose()
      triggerGeo.dispose()
      triggerMat.dispose()
      plainBox.dispose()
      for (const m of coloredMats.values()) m.dispose()
      assets?.dispose()
      renderer.dispose()
      renderer.domElement.remove()
      labelRenderer.domElement.remove()
    },
  }
}
