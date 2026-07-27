// Single-machine animated scene for the /generator page. Plays back a Machine's real move and
// piston-extension timelines (Machine.moves/extensions - tick-stamped keyframes read straight out
// of the .simlog by stream_to_visualizer.py's build_animation_record, never synthesized) by
// snapping each moved block (or extending piston head) from one real keyframe to the next over a
// short fixed window right at the keyframe's own tick, then holding position until the next real
// event - NOT a continuous glide stretched across however long the gap between events happens to
// be (that was the original bug: a block with events at tick 1 and tick 13 would drift for 12
// ticks straight, which reads as "slow" and desyncs relative timing between blocks that don't
// share the same keyframe spacing). The only thing ever interpolated is the short snap transition
// itself; the keyframe positions/states themselves always come straight from the simulator's own
// event log.
//
// Deliberately NOT the shared multi-machine scene.ts: that one packs every visible machine's
// blocks into one InstancedMesh per block type across the whole grid, with per-instance matrices
// baked once at build time. Animating a single instance's position every frame is much simpler to
// reason about when there's only one machine's blocks in play, so this keeps its own small scene
// instead of reaching into scene.ts's shared packing.
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { BLOCK_TYPES, decodeState } from './blocks'
import { loadBlockAssets, frontAxis, type BlockAssets } from './textures'
import type { Machine, MachineEvent } from './data'

// Playback speed only - a display choice, not simulated timing. Real Minecraft ticks aren't
// evenly spaced in wall-clock terms here either; this just picks a watchable pace. User-adjustable
// via AnimatedSceneHandle.setSpeed - starts as a plain const's value but becomes a `let` below.
const DEFAULT_MS_PER_TICK = 150
// How long (in tick units) a move/extension takes to visually complete, ending exactly AT the
// keyframe's own tick - not a glide spanning the whole gap since the previous keyframe. Used by
// auto (looping) playback only.
const SNAP_TICKS = 0.4
// How long (wall-clock ms) a manual forward step's snap-into-place transition takes. Backward
// steps skip this entirely (instant snap) - see setEventIndex.
const MANUAL_STEP_MS = 200
const PISTON_HEAD_ID = 34 // BLOCK_PISTON_HEAD (blocks.py) - already has a texture/geometry entry
const TRIGGER_COLOR = 0xb14aff // matches scene.ts's trigger glow exactly
const BLOCKED_COLOR = 0xff9d9d // light red - a blocked piston push
const DROPPED_COLOR = 0xff3b3b // solid red - reserved for a scheduledTickDropped event only

export interface AnimatedSceneHandle {
  loadMachine(machine: Machine): void
  dispose(): void
  getEvents(): MachineEvent[]
  getEventIndex(): number
  stepSubtick(delta: number): void
  stepTick(direction: 1 | -1): void
  jumpToTick(tick: number): void
  play(): void
  pause(): void
  isPlaying(): boolean
  setSpeed(msPerTick: number): void
  getSpeed(): number
}

interface PosKeyframe {
  tick: number
  order: number
  pos: THREE.Vector3
}
interface ExtKeyframe {
  tick: number
  order: number
  extended: boolean
}

interface AnimatedBlock {
  blockIndex: number
  mesh: THREE.InstancedMesh
  instanceIndex: number
  quat: THREE.Quaternion
  keyframes: PosKeyframe[] // keyframes[0] is always the real tick-0 (initial) position
  renderPos: THREE.Vector3 // manual mode only: the currently-displayed position (persists across frames)
  transFrom: THREE.Vector3 | null // manual mode only: mid-transition source (forward steps animate)
  transTo: THREE.Vector3 | null
}

interface AnimatedHead {
  blockIndex: number
  mesh: THREE.InstancedMesh
  instanceIndex: number
  quat: THREE.Quaternion
  bodyPos: THREE.Vector3 // the piston's static placement - fallback if it's never itself pushed
  facing: THREE.Vector3 // unit vector the head slides along when extending
  keyframes: ExtKeyframe[]
  renderBlend: number // manual mode only: the currently-displayed 0..1 extension blend
  transFrom: number | null
  transTo: number | null
}

interface EffectOverlay {
  mesh: THREE.InstancedMesh
  instanceIndex: number
  blockIndex: number
  staticPos: THREE.Vector3 // fallback for a piston/observer that never itself moves
  quat: THREE.Quaternion // facing rotation to render at - identity for non-directional overlays
}

// Holds at `cur.pos` for the whole gap since the previous keyframe, then animates into `next.pos`
// only during the last SNAP_TICKS of that gap, arriving exactly at `next.tick`. Auto (looping)
// playback only - manual stepping uses lastKeyframeAt below instead.
function positionAt(keyframes: PosKeyframe[], t: number): THREE.Vector3 {
  if (t <= keyframes[0].tick) return keyframes[0].pos
  for (let i = 0; i < keyframes.length - 1; i++) {
    const cur = keyframes[i]
    const next = keyframes[i + 1]
    if (t < next.tick) {
      const windowStart = next.tick - SNAP_TICKS
      if (t < windowStart) return cur.pos
      const frac = Math.min(1, Math.max(0, (t - windowStart) / SNAP_TICKS))
      return cur.pos.clone().lerp(next.pos, frac)
    }
  }
  return keyframes[keyframes.length - 1].pos
}

// Same hold-then-snap timing as positionAt, but for a boolean extended/retracted state - returns
// a 0..1 blend factor (0 = fully retracted, 1 = fully extended) for lerping the head's offset.
function extensionBlendAt(keyframes: ExtKeyframe[], t: number): number {
  const first = keyframes[0].extended ? 1 : 0
  if (t <= keyframes[0].tick) return first
  for (let i = 0; i < keyframes.length - 1; i++) {
    const cur = keyframes[i]
    const next = keyframes[i + 1]
    const from = cur.extended ? 1 : 0
    const to = next.extended ? 1 : 0
    if (t < next.tick) {
      const windowStart = next.tick - SNAP_TICKS
      if (t < windowStart) return from
      const frac = Math.min(1, Math.max(0, (t - windowStart) / SNAP_TICKS))
      return from + (to - from) * frac
    }
  }
  const last = keyframes[keyframes.length - 1]
  return last.extended ? 1 : 0
}

// Manual-mode state resolution: the LAST keyframe whose (tick, order) is <= the target - an exact
// snapshot of what the simlog says was true at that point, no interpolation. (tick=0, order=-1)
// as a target correctly resolves to keyframes[0], the real initial state, since every real event
// has order >= 0.
function lastKeyframeAt<T extends { tick: number; order: number }>(keyframes: T[], tick: number, order: number): T {
  let result = keyframes[0]
  for (const k of keyframes) {
    if (k.tick < tick || (k.tick === tick && k.order <= order)) result = k
    else break
  }
  return result
}

export function createAnimatedScene(container: HTMLElement): AnimatedSceneHandle {
  const scene = new THREE.Scene()
  scene.background = new THREE.Color('#0f111a')

  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 2000)
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  container.appendChild(renderer.domElement)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.15))
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.1)
  dirLight.position.set(1, 2, 1.5)
  scene.add(dirLight)

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

  // Overlay geometry/materials are created once and reused across loadMachine() calls (matches
  // the existing plainBox/coloredMats pattern below) - only the per-machine InstancedMesh built
  // from them is torn down and rebuilt each load.
  const overlayGeo = new THREE.BoxGeometry(1.02, 1.02, 1.02)
  const overlayMat = (color: number, opacity: number) =>
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending })
  const triggerMat = overlayMat(TRIGGER_COLOR, 0.5)
  const blockedMat = overlayMat(BLOCKED_COLOR, 0.6)
  const droppedMat = overlayMat(DROPPED_COLOR, 0.6)
  const IDENTITY_QUAT = new THREE.Quaternion()

  let meshes: THREE.InstancedMesh[] = []
  let animated: AnimatedBlock[] = []
  let animatedByIndex = new Map<number, AnimatedBlock>()
  let heads: AnimatedHead[] = []
  let triggerOverlay: EffectOverlay | null = null
  let blockedOverlays: EffectOverlay[] = []
  let observerOverlays: EffectOverlay[] = []
  let droppedOverlays: EffectOverlay[] = []
  // Tick lists (not single events) a piston is "blocked" / a scheduledTickDropped fired at - see
  // isActiveAtTick: both are one-shot instants, held visible for exactly the tick they land in.
  let blockedTicksByIndex = new Map<number, number[]>()
  let droppedTicksByIndex = new Map<number, number[]>()
  // An observer's real on/off interval, paired from its own alternating observerFired/observerOff
  // stream - see isObserverActiveAtTick. Replaces the old fixed-duration OBSERVER_ON_TICKS guess
  // now that the simulator actually logs the off transition.
  let observerIntervalsByIndex = new Map<number, { on: number; off: number }[]>()
  let events: MachineEvent[] = []
  let eventIndex = -1 // -1 = the real t=0 initial state, before any logged event
  let mode: 'auto' | 'manual' = 'auto'
  let transitionStart = 0
  let terminationTick = 0
  let playbackStart = performance.now()
  let msPerTick = DEFAULT_MS_PER_TICK
  let loadToken = 0
  const dummy = new THREE.Object3D()

  function clearMachine() {
    for (const m of meshes) {
      scene.remove(m)
      m.dispose()
    }
    meshes = []
    animated = []
    animatedByIndex = new Map()
    heads = []
    triggerOverlay = null
    blockedOverlays = []
    observerOverlays = []
    droppedOverlays = []
    blockedTicksByIndex = new Map()
    droppedTicksByIndex = new Map()
    observerIntervalsByIndex = new Map()
    events = []
    eventIndex = -1
    mode = 'auto'
  }

  // Builds one small InstancedMesh (per-instance position updated every frame by
  // updateEffectOverlays, since a piston/observer can itself be a block that gets pushed around by
  // something else - it must flash at wherever it currently is, not the spot it started at) shared
  // by a whole overlay type - trigger glow, blocked-push flash, observer-fire flash.
  function buildOverlay(
    geometry: THREE.BufferGeometry,
    material: THREE.Material,
    entries: { blockIndex: number; pos: THREE.Vector3; quat?: THREE.Quaternion }[],
  ): EffectOverlay[] {
    if (!entries.length) return []
    const mesh = new THREE.InstancedMesh(geometry, material, entries.length)
    mesh.frustumCulled = false
    mesh.renderOrder = 2
    entries.forEach((e, i) => {
      dummy.position.copy(e.pos)
      dummy.quaternion.copy(e.quat ?? IDENTITY_QUAT)
      dummy.scale.set(1, 1, 1)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    })
    mesh.instanceMatrix.needsUpdate = true
    scene.add(mesh)
    meshes.push(mesh)
    return entries.map((e, i) => ({
      mesh, instanceIndex: i, blockIndex: e.blockIndex, staticPos: e.pos, quat: (e.quat ?? IDENTITY_QUAT).clone(),
    }))
  }

  // The world position a block should render/flash at RIGHT NOW - its own live animated position
  // if it's ever pushed by something else (a piston/observer body can itself be moved by another
  // piston), otherwise its fixed placement. Shared by the overlay effects AND the piston head,
  // which both otherwise baked in a stale tick-0 position forever.
  function livePositionNow(blockIndex: number, staticPos: THREE.Vector3): THREE.Vector3 {
    const ab = animatedByIndex.get(blockIndex)
    if (!ab) return staticPos
    return mode === 'manual' ? ab.renderPos : positionAt(ab.keyframes, currentTick())
  }

  // A blocked push / scheduledTickDropped is a single, complete, real event (one failed attempt,
  // one dropped reschedule) - held visible for exactly the tick it landed in, not a truncated
  // state.
  function isActiveAtTick(ticksByIndex: Map<number, number[]>, blockIndex: number, t: number, durationTicks: number): boolean {
    const ticks = ticksByIndex.get(blockIndex)
    return ticks !== undefined && ticks.some((tick) => t >= tick && t < tick + durationTicks)
  }

  // An observer's real on/off interval, now that the simlog actually logs both transitions (see
  // stream_to_visualizer.py's observerFired/observerOff split) - lit exactly between its own fire
  // and the matching off, half-open on the low end / open on the high end.
  function isObserverActiveAtTick(blockIndex: number, t: number): boolean {
    const intervals = observerIntervalsByIndex.get(blockIndex)
    return intervals !== undefined && intervals.some((iv) => t >= iv.on && t < iv.off)
  }

  function setOverlayVisible(ov: EffectOverlay, pos: THREE.Vector3, visible: boolean) {
    dummy.position.copy(pos)
    dummy.quaternion.copy(ov.quat)
    const s = visible ? 1 : 0
    dummy.scale.set(s, s, s)
    dummy.updateMatrix()
    ov.mesh.setMatrixAt(ov.instanceIndex, dummy.matrix)
    ov.mesh.instanceMatrix.needsUpdate = true
  }

  function updateEffectOverlays() {
    const t = mode === 'manual' ? (eventIndex < 0 ? 0 : events[eventIndex].tick) : currentTick()
    if (triggerOverlay) {
      // "at tick zero and subtick zero" = paused, before any event has run yet.
      setOverlayVisible(triggerOverlay, triggerOverlay.staticPos, mode === 'manual' && eventIndex < 0)
    }
    for (const ov of blockedOverlays) {
      const pos = livePositionNow(ov.blockIndex, ov.staticPos)
      setOverlayVisible(ov, pos, isActiveAtTick(blockedTicksByIndex, ov.blockIndex, t, 1))
    }
    for (const ov of observerOverlays) {
      const pos = livePositionNow(ov.blockIndex, ov.staticPos)
      setOverlayVisible(ov, pos, isObserverActiveAtTick(ov.blockIndex, t))
    }
    for (const ov of droppedOverlays) {
      const pos = livePositionNow(ov.blockIndex, ov.staticPos)
      setOverlayVisible(ov, pos, isActiveAtTick(droppedTicksByIndex, ov.blockIndex, t, 1))
    }
  }

  async function loadMachine(machine: Machine) {
    const token = ++loadToken
    await assetsReady
    if (token !== loadToken) return
    clearMachine()

    const blocks = machine.candidate.blocks
    const xs = blocks.map((b) => b.x)
    const ys = blocks.map((b) => b.y)
    const zs = blocks.map((b) => b.z)
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2
    const cy = Math.min(...ys)
    const cz = (Math.min(...zs) + Math.max(...zs)) / 2
    // Same MC -> three transform scene.ts uses: y up, negate z for right-handed, center in view.
    const toWorld = (x: number, y: number, z: number) => new THREE.Vector3(x - cx, y - cy + 0.5, -(z - cz))

    const byId = new Map<number, { blockIndex: number; matrix: THREE.Matrix4; quat: THREE.Quaternion }[]>()
    const quatByIndex = new Map<number, THREE.Quaternion>()
    blocks.forEach((blk, i) => {
      const d = decodeState(blk.state)
      const quat = new THREE.Quaternion()
      const axis = frontAxis(d.blockId)
      if (axis) quat.setFromUnitVectors(axis, new THREE.Vector3(...d.facingVec))
      quatByIndex.set(i, quat)
      dummy.position.copy(toWorld(blk.x, blk.y, blk.z))
      dummy.quaternion.copy(quat)
      dummy.scale.set(1, 1, 1)
      dummy.updateMatrix()
      const list = byId.get(d.blockId) ?? []
      list.push({ blockIndex: i, matrix: dummy.matrix.clone(), quat })
      byId.set(d.blockId, list)
    })

    const indexInMesh = new Map<number, { mesh: THREE.InstancedMesh; instanceIndex: number; quat: THREE.Quaternion }>()
    for (const [id, list] of byId) {
      const geo = assets ? assets.geo(id) : plainBox
      const mat = assets ? assets.material(id) : coloredMat(id)
      const mesh = new THREE.InstancedMesh(geo, mat, list.length)
      list.forEach((e, i) => {
        mesh.setMatrixAt(i, e.matrix)
        indexInMesh.set(e.blockIndex, { mesh, instanceIndex: i, quat: e.quat })
      })
      mesh.instanceMatrix.needsUpdate = true
      mesh.frustumCulled = false
      scene.add(mesh)
      meshes.push(mesh)
    }

    animated = (machine.moves ?? []).flatMap((mv) => {
      const start = indexInMesh.get(mv.blockIndex)
      const initial = blocks[mv.blockIndex]
      if (!start || !initial) return []
      // order -1 for the real t=0 state - sorts before every logged event (all have order >= 0).
      const keyframes: PosKeyframe[] = [{ tick: 0, order: -1, pos: toWorld(initial.x, initial.y, initial.z) }]
      for (const step of mv.steps) keyframes.push({ tick: step.tick, order: step.order, pos: toWorld(step.x, step.y, step.z) })
      return [{
        blockIndex: mv.blockIndex, mesh: start.mesh, instanceIndex: start.instanceIndex, quat: start.quat, keyframes,
        renderPos: keyframes[0].pos.clone(), transFrom: null, transTo: null,
      }]
    })
    animatedByIndex = new Map(animated.map((a) => [a.blockIndex, a]))

    // Piston head extension - one small InstancedMesh sized to the piston count, sliding along
    // each piston's own facing between its body position (retracted) and body + facing (extended).
    const extEntries = (machine.extensions ?? []).flatMap((ext) => {
      const initial = blocks[ext.blockIndex]
      const quat = quatByIndex.get(ext.blockIndex)
      if (!initial || !quat || ext.steps.length === 0) return []
      const d = decodeState(initial.state)
      const facing = new THREE.Vector3(...d.facingVec)
      const bodyPos = toWorld(initial.x, initial.y, initial.z)
      return [{ blockIndex: ext.blockIndex, bodyPos, facing, quat, keyframes: ext.steps as ExtKeyframe[] }]
    })
    if (extEntries.length) {
      const geo = assets ? assets.geo(PISTON_HEAD_ID) : plainBox
      const mat = assets ? assets.material(PISTON_HEAD_ID) : coloredMat(PISTON_HEAD_ID)
      const headMesh = new THREE.InstancedMesh(geo, mat, extEntries.length)
      headMesh.frustumCulled = false
      extEntries.forEach((e, i) => {
        dummy.position.copy(e.bodyPos)
        dummy.quaternion.copy(e.quat)
        dummy.scale.set(1, 1, 1)
        dummy.updateMatrix()
        headMesh.setMatrixAt(i, dummy.matrix)
      })
      headMesh.instanceMatrix.needsUpdate = true
      scene.add(headMesh)
      meshes.push(headMesh)
      heads = extEntries.map((e, i) => ({
        blockIndex: e.blockIndex, mesh: headMesh, instanceIndex: i, quat: e.quat, bodyPos: e.bodyPos, facing: e.facing,
        keyframes: e.keyframes, renderBlend: e.keyframes[0].extended ? 1 : 0,
        transFrom: null, transTo: null,
      }))
    }

    // Trigger glow (purple, matches scene.ts's multi-machine view) - only while parked at the real
    // t=0 state (tick 0, before any event has happened - see updateEffectOverlays), since the
    // trigger itself is a one-shot "this is where it all starts" marker, not a thing that's ever
    // "on" once the machine is actually running. Blocked-push (light red) / a dropped reschedule
    // (solid red, reserved for this event only) overlays: lit for the tick(s)/interval that block
    // has a matching event in - see isActiveAtTick. The observer's "on" state uses a real lit
    // texture quad instead of a colored overlay - see observerLitGeo/observerLitMat.
    triggerOverlay = buildOverlay(overlayGeo, triggerMat, [
      { blockIndex: -1, pos: toWorld(machine.candidate.trigger.x, machine.candidate.trigger.y, machine.candidate.trigger.z) },
    ])[0] ?? null
    const pistonEntries: { blockIndex: number; pos: THREE.Vector3 }[] = []
    const observerEntries: { blockIndex: number; pos: THREE.Vector3; quat: THREE.Quaternion }[] = []
    blocks.forEach((blk, i) => {
      const id = decodeState(blk.state).blockId
      if (id === 29 || id === 33) pistonEntries.push({ blockIndex: i, pos: toWorld(blk.x, blk.y, blk.z) })
      else if (id === 218) {
        observerEntries.push({ blockIndex: i, pos: toWorld(blk.x, blk.y, blk.z), quat: quatByIndex.get(i) ?? new THREE.Quaternion() })
      }
    })
    blockedOverlays = buildOverlay(overlayGeo, blockedMat, pistonEntries)
    observerOverlays = assets ? buildOverlay(assets.observerOnGeo, assets.observerOnMat, observerEntries) : []

    events = machine.events ?? []
    terminationTick = machine.terminationTick ?? 0

    // scheduledTickDropped's subject isn't necessarily a piston/observer - it's whatever occupies
    // the position at drop time - so its overlay entries can't be pre-enumerated by block type;
    // only build one for a blockIndex that actually appears in such an event.
    const droppedIndices = new Set(events.filter((e) => e.kind === 'scheduledTickDropped').map((e) => e.blockIndex))
    const droppedEntries = [...droppedIndices].flatMap((idx) => {
      const blk = blocks[idx]
      return blk ? [{ blockIndex: idx, pos: toWorld(blk.x, blk.y, blk.z) }] : []
    })
    droppedOverlays = buildOverlay(overlayGeo, droppedMat, droppedEntries)

    // Observer on/off intervals: pair each blockIndex's own alternating observerFired/observerOff
    // ticks by array position (events is already sorted by (tick, order), and a single observer's
    // own fire/off stream strictly alternates, so onTicks[i]/offTicks[i] is a safe pairing). A fire
    // with no matching off yet (log ends mid-pulse) stays lit through the rest of playback instead
    // of never lighting or throwing.
    const onTicksByIndex = new Map<number, number[]>()
    const offTicksByIndex = new Map<number, number[]>()
    for (const e of events) {
      if (e.kind === 'pistonBlocked') {
        const arr = blockedTicksByIndex.get(e.blockIndex) ?? []
        arr.push(e.tick)
        blockedTicksByIndex.set(e.blockIndex, arr)
      } else if (e.kind === 'scheduledTickDropped') {
        const arr = droppedTicksByIndex.get(e.blockIndex) ?? []
        arr.push(e.tick)
        droppedTicksByIndex.set(e.blockIndex, arr)
      } else if (e.kind === 'observerFired') {
        const arr = onTicksByIndex.get(e.blockIndex) ?? []
        arr.push(e.tick)
        onTicksByIndex.set(e.blockIndex, arr)
      } else if (e.kind === 'observerOff') {
        const arr = offTicksByIndex.get(e.blockIndex) ?? []
        arr.push(e.tick)
        offTicksByIndex.set(e.blockIndex, arr)
      }
    }
    for (const [blockIndex, onTicks] of onTicksByIndex) {
      const offTicks = offTicksByIndex.get(blockIndex) ?? []
      const intervals = onTicks.map((on, i) => ({ on, off: offTicks[i] ?? terminationTick + 1 }))
      observerIntervalsByIndex.set(blockIndex, intervals)
    }

    eventIndex = -1
    mode = 'auto'

    const footprint = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...zs) - Math.min(...zs), 4)
    camera.position.set(footprint * 0.9, footprint * 1.1 + 4, footprint * 1.3)
    controls.target.set(0, 1, 0)
    controls.update()

    playbackStart = performance.now()
  }

  function currentTick(): number {
    if (terminationTick <= 0) return 0
    const loopLength = terminationTick + 1 // +1 tick pause at rest before looping back to start
    return ((performance.now() - playbackStart) / msPerTick) % loopLength
  }

  function renderHead(head: AnimatedHead, blend: number, bodyPos: THREE.Vector3) {
    dummy.position.copy(bodyPos).addScaledVector(head.facing, blend)
    dummy.quaternion.copy(head.quat)
    // Fully retracted (blend 0) puts the head at the exact same spot as the piston body - scale
    // it to nothing instead of leaving two opaque unit cubes coincident, which z-fights/flickers
    // on every non-extended piston (most of them, most of the time).
    const s = blend > 0.001 ? 1 : 0
    dummy.scale.set(s, s, s)
    dummy.updateMatrix()
    head.mesh.setMatrixAt(head.instanceIndex, dummy.matrix)
    head.mesh.instanceMatrix.needsUpdate = true
  }

  function renderBlock(block: AnimatedBlock, pos: THREE.Vector3) {
    dummy.position.copy(pos)
    dummy.quaternion.copy(block.quat)
    dummy.scale.set(1, 1, 1)
    dummy.updateMatrix()
    block.mesh.setMatrixAt(block.instanceIndex, dummy.matrix)
    block.mesh.instanceMatrix.needsUpdate = true
  }

  // Auto (looping) playback - unchanged wall-clock-driven hold-then-snap behavior.
  function stepAuto() {
    const t = currentTick()
    for (const block of animated) renderBlock(block, positionAt(block.keyframes, t))
    for (const head of heads) {
      renderHead(head, extensionBlendAt(head.keyframes, t), livePositionNow(head.blockIndex, head.bodyPos))
    }
  }

  // Manual (pointer-driven) playback - state resolved exactly from the event pointer, with a
  // short transition ONLY when the pointer just moved forward (see animateToIndex/snapToIndex).
  function stepManual() {
    const now = performance.now()
    const frac = Math.min(1, (now - transitionStart) / MANUAL_STEP_MS)
    for (const block of animated) {
      if (block.transTo !== null) {
        block.renderPos = frac >= 1 ? block.transTo : block.transFrom!.clone().lerp(block.transTo, frac)
        if (frac >= 1) { block.transFrom = null; block.transTo = null }
      }
      renderBlock(block, block.renderPos)
    }
    for (const head of heads) {
      if (head.transTo !== null) {
        head.renderBlend = frac >= 1 ? head.transTo : head.transFrom! + (head.transTo - head.transFrom!) * frac
        if (frac >= 1) { head.transFrom = null; head.transTo = null }
      }
      renderHead(head, head.renderBlend, livePositionNow(head.blockIndex, head.bodyPos))
    }
  }

  function targetOf(index: number): { tick: number; order: number } {
    if (index < 0) return { tick: 0, order: -1 }
    const e = events[index]
    return { tick: e.tick, order: e.order }
  }

  function clampIndex(i: number): number {
    return Math.max(-1, Math.min(events.length - 1, i))
  }

  function snapToIndex(idx: number) {
    eventIndex = idx
    mode = 'manual'
    const { tick, order } = targetOf(idx)
    for (const block of animated) {
      block.renderPos = lastKeyframeAt(block.keyframes, tick, order).pos
      block.transFrom = null
      block.transTo = null
    }
    for (const head of heads) {
      head.renderBlend = lastKeyframeAt(head.keyframes, tick, order).extended ? 1 : 0
      head.transFrom = null
      head.transTo = null
    }
  }

  function animateToIndex(idx: number) {
    eventIndex = idx
    mode = 'manual'
    transitionStart = performance.now()
    const { tick, order } = targetOf(idx)
    for (const block of animated) {
      block.transFrom = block.renderPos.clone()
      block.transTo = lastKeyframeAt(block.keyframes, tick, order).pos
    }
    for (const head of heads) {
      head.transFrom = head.renderBlend
      head.transTo = lastKeyframeAt(head.keyframes, tick, order).extended ? 1 : 0
    }
  }

  // Backward -> instant snap; forward -> short animated transition. Matches the request exactly:
  // "snap backwards in time if going back and animate forward if going forward."
  function setEventIndex(newIndex: number) {
    const clamped = clampIndex(newIndex)
    if (mode === 'manual' && clamped === eventIndex) return
    if (mode === 'manual' && clamped < eventIndex) snapToIndex(clamped)
    else animateToIndex(clamped)
  }

  function nearestEventIndexAtOrBefore(t: number): number {
    let idx = -1
    for (let i = 0; i < events.length; i++) {
      if (events[i].tick <= t) idx = i
      else break
    }
    return idx
  }

  function pause() {
    if (mode === 'manual') return
    // Snap render state to auto's current (possibly mid-glide) position first so entering manual
    // mode doesn't visibly jump, then resolve the exact event index at-or-before that tick.
    const t = currentTick()
    for (const block of animated) block.renderPos = positionAt(block.keyframes, t)
    for (const head of heads) head.renderBlend = extensionBlendAt(head.keyframes, t)
    snapToIndex(nearestEventIndexAtOrBefore(t))
  }

  function play() {
    const t = eventIndex < 0 ? 0 : events[eventIndex].tick
    playbackStart = performance.now() - t * msPerTick
    mode = 'auto'
  }

  // Keeps the currently-displayed tick continuous across a speed change - otherwise retiming
  // playbackStart's fixed reference point against a new msPerTick would jump the visible tick.
  function setSpeed(newMsPerTick: number) {
    const t = currentTick()
    msPerTick = Math.max(1, newMsPerTick)
    playbackStart = performance.now() - t * msPerTick
  }

  function stepSubtick(delta: number) {
    if (mode !== 'manual') pause()
    setEventIndex(eventIndex + delta)
  }

  // Jumps to the first event of the previous/next DISTINCT tick that actually has events -
  // walking by "current tick +/- 1" (the old approach) breaks whenever the log skips ticks (e.g.
  // events at 1, 3, 7 - asking for tick 6 would land right back on tick 7's own first event,
  // i.e. a no-op, which is exactly the "backward button does nothing" bug this replaces).
  function stepTick(direction: 1 | -1) {
    if (mode !== 'manual') pause()
    const curTick = eventIndex < 0 ? 0 : events[eventIndex].tick
    if (direction < 0) {
      let prevTick: number | null = null
      for (let i = events.length - 1; i >= 0; i--) {
        if (events[i].tick < curTick) {
          prevTick = events[i].tick
          break
        }
      }
      setEventIndex(prevTick === null ? -1 : events.findIndex((e) => e.tick === prevTick))
      return
    }
    const idx = events.findIndex((e) => e.tick > curTick)
    setEventIndex(idx === -1 ? events.length - 1 : idx)
  }

  function jumpToTick(tick: number) {
    if (mode !== 'manual') pause()
    if (tick <= 0) {
      setEventIndex(-1)
      return
    }
    let idx = events.findIndex((e) => e.tick >= tick)
    if (idx === -1) idx = events.length - 1
    setEventIndex(idx)
  }

  let raf = 0
  function animate() {
    raf = requestAnimationFrame(animate)
    controls.update()
    if (mode === 'auto') stepAuto()
    else stepManual()
    updateEffectOverlays()
    renderer.render(scene, camera)
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = container
    if (!w || !h) return
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }
  const ro = new ResizeObserver(resize)
  ro.observe(container)
  resize()
  animate()

  return {
    loadMachine,
    getEvents: () => events,
    getEventIndex: () => eventIndex,
    stepSubtick,
    stepTick,
    jumpToTick,
    play,
    pause,
    isPlaying: () => mode === 'auto',
    setSpeed,
    getSpeed: () => msPerTick,
    dispose() {
      cancelAnimationFrame(raf)
      ro.disconnect()
      clearMachine()
      controls.dispose()
      plainBox.dispose()
      for (const m of coloredMats.values()) m.dispose()
      overlayGeo.dispose()
      triggerMat.dispose()
      blockedMat.dispose()
      droppedMat.dispose()
      assets?.dispose()
      renderer.dispose()
      renderer.domElement.remove()
    },
  }
}
