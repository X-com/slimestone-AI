<script lang="ts">
  import { onMount } from 'svelte'
  import {
    applyTraining,
    machineFromAnimation,
    parseCompactData,
    type AnimationFrame,
    type Machine,
    type RunStats,
    type Training,
    type TrainingFrame,
  } from './lib/data'
  import { createScene, type SceneHandle } from './lib/scene'
  import { createAnimatedScene, type AnimatedSceneHandle } from './lib/animatedScene'
  import MachineDetailPanel from './lib/MachineDetailPanel.svelte'

  const PAGE = 25
  const MAX_HISTORY_PAGES = 100 // drop oldest machines beyond this so a long session doesn't grow forever

  let topContainer: HTMLDivElement
  let bottomContainer: HTMLDivElement
  let topHandle = $state<SceneHandle | null>(null)
  let bottomHandle = $state<SceneHandle | null>(null)

  // Flat history in arrival order (oldest -> newest). Never re-sliced into the bottom scene
  // except on explicit navigation, so a new batch never disturbs the page you're viewing.
  //
  // DELIBERATELY NOT $state. Svelte 5's $state deep-proxies everything written into it, and at
  // the 2,500-machine cap that is ~40,000 proxied block objects - which setMachines then reads
  // hard (three blocks.map() passes plus a for-of per machine), paying a proxy trap on every
  // property access, on every rebuild, on every batch. Measured on that exact read pattern:
  // 2,000 rebuilds of 25 machines cost 39 ms plain and 1,283 ms proxied, a 33x difference.
  // Nothing here needs deep reactivity:
  // showPage() slices explicitly and scene.ts is imperative three.js outside Svelte entirely.
  // The only reactive consumer is the length, which machineCount carries.
  let machines: Machine[] = []
  let machineCount = $state(0)
  let latestBatch = $state<Machine[]>([]) // last decoded batch, shown up top (render <=100)
  let bottomVisible = $state<Machine[]>([]) // current history page's machines, for the stepper
  let page = $state(0) // bottom page index, anchored from the OLDEST machine
  let batches = $state(0) // batches received this session (also the per-batch hash namespace)
  let selected = $state<Machine | null>(null)
  let run = $state<Partial<RunStats>>({}) // the loop's own round row, refreshed once per round
  // Metadata frames can in principle arrive before the geometry they describe (they are two
  // sends, and only ordered per-connection). Anything unmatched is held here and re-applied on
  // the next batch rather than dropped, so a race shows up as a delay, never as lost data.
  let orphanMeta: Record<string, Training> = {}

  // --- the player -------------------------------------------------------------------------
  // The still-life view answers "what was found"; this answers "does it actually fly". The
  // motion is not simulated in the browser: the run re-simulates the machine with logging on
  // and sends back its real per-tick event record (rlgym/animation.py), which animatedScene.ts
  // already knows how to play. Asked for one machine at a time, so a run nobody watches pays
  // nothing.
  let animating = $state<Machine | null>(null)
  let awaitingId = $state<number | null>(null) // the candidate id we asked about, if any
  let animationError = $state('')
  let playerContainer = $state<HTMLDivElement>()
  let playerHandle: AnimatedSceneHandle | null = null
  let playing = $state(false)
  let speed = $state(180)

  function requestAnimation(m: Machine) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      animationError = 'not connected to a training run'
      return
    }
    animationError = ''
    awaitingId = m.candidate.id
    ws.send(JSON.stringify({ want: 'animation', id: m.candidate.id }))
  }

  function onAnimation(frame: AnimationFrame) {
    // Ignore an answer to a request we have moved on from - clicking two machines quickly would
    // otherwise open whichever simulation happened to finish last.
    if (frame.id !== awaitingId) return
    awaitingId = null
    if (!frame.animation) {
      animationError = frame.error ?? 'the run could not simulate that machine'
      return
    }
    closePlayer()
    animating = machineFromAnimation(frame.animation, selected?.label ?? `#${frame.id}`)
  }

  function closePlayer() {
    playerHandle?.dispose()
    playerHandle = null
    animating = null
    playing = false
  }

  // The scene is imperative three.js, so it is built after the container exists and torn down
  // by closePlayer - not by an effect cleanup, which would also fire on every speed change.
  $effect(() => {
    if (!animating || !playerContainer) return
    playerHandle ??= createAnimatedScene(playerContainer)
    playerHandle.loadMachine(animating)
    playerHandle.setSpeed(speed)
    playerHandle.play()
    playing = true
  })

  function togglePlay() {
    if (!playerHandle) return
    if (playerHandle.isPlaying()) playerHandle.pause()
    else playerHandle.play()
    playing = playerHandle.isPlaying()
  }

  let rootEl: HTMLDivElement
  let isFullscreen = $state(false)
  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen()
    else rootEl.requestFullscreen()
  }

  // Draggable split between the latest-batch viewport and the history viewport, as a flex-grow
  // ratio (not a literal percentage) so the divider's own height doesn't need subtracting out.
  let splitContainer: HTMLDivElement
  let splitPct = $state(34)
  let splitDragging = false
  function onSplitDown(e: PointerEvent) {
    splitDragging = true
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }
  function onSplitMove(e: PointerEvent) {
    if (!splitDragging) return
    const rect = splitContainer.getBoundingClientRect()
    const pct = ((e.clientY - rect.top) / rect.height) * 100
    splitPct = Math.min(85, Math.max(15, pct))
  }
  function onSplitUp() {
    splitDragging = false
  }

  // Plain ws:// - rlgym/stream.py serves without TLS, because the page and the training run on
  // the same machine. A remote deployment puts a terminating proxy in front (see the
  // integration doc) and this box takes the wss:// URL that hands you.
  let url = $state('ws://localhost:8765')
  let status = $state<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle')
  let ws: WebSocket | null = null

  // Auto-reconnect: if a live connection drops, retry until it comes back.
  //
  // Two cadences, because the two situations are not alike. Before the first successful
  // connection we are racing the training process's own startup - train.bat launches this
  // viewer and the loop together, and the loop has a checkpoint to load before it binds - so
  // retrying once a second turns that race into a barely-visible pause. After a connection has
  // worked once, a drop means the run ended or the machine went away, and hammering it every
  // second for hours would be pointless.
  const RETRY_MS = 30_000
  const STARTUP_RETRY_MS = 1_000
  let everConnected = $state(false) // read in the retry label, so it has to be reactive
  const retryDelay = () => (everConnected ? RETRY_MS : STARTUP_RETRY_MS)
  // On by default: this page exists to watch a run, and the overwhelmingly common case is that
  // it was opened BY the thing it wants to watch.
  let reconnect = $state(true)
  let retryPending = $state(false)
  let manualClose = false // user-initiated Disconnect must not trigger a retry
  let retryTimer: ReturnType<typeof setTimeout> | null = null

  function clearRetry() {
    if (retryTimer) clearTimeout(retryTimer)
    retryTimer = null
    retryPending = false
  }
  function scheduleRetry() {
    if (retryTimer) return
    retryPending = true
    retryTimer = setTimeout(() => {
      retryTimer = null
      retryPending = false
      connect()
    }, retryDelay())
  }

  const totalPages = $derived(Math.max(1, Math.ceil(machineCount / PAGE)))
  const connected = $derived(status === 'connecting' || status === 'connected')

  // Shared by both viewports so selecting in one clears/syncs the other (whichever handle
  // doesn't contain the selected hash just shows no highlight — no extra bookkeeping needed).
  function onSelect(m: Machine | null) {
    selected = m
    topHandle?.setSelected(m?.hash ?? null)
    bottomHandle?.setSelected(m?.hash ?? null)
  }

  // Per-viewport machine stepper (arrows + typed index), shared by the top and bottom controls.
  function stepMachine(list: Machine[], handle: SceneHandle | null, delta: number) {
    if (!list.length) return
    const i = selected ? list.findIndex((m) => m.hash === selected!.hash) : -1
    const n = list.length
    const next = i < 0 ? (delta > 0 ? 0 : n - 1) : (i + delta + n) % n
    onSelect(list[next])
    handle?.focusSelected(list[next].hash)
  }
  function jumpMachine(list: Machine[], handle: SceneHandle | null, oneIndexed: number) {
    if (!list.length) return
    const idx = Math.min(list.length, Math.max(1, Math.round(oneIndexed) || 1)) - 1
    onSelect(list[idx])
    handle?.focusSelected(list[idx].hash)
  }
  const topIndex = $derived(selected ? latestBatch.findIndex((m) => m.hash === selected!.hash) + 1 : 0)
  const bottomIndex = $derived(
    selected ? bottomVisible.findIndex((m) => m.hash === selected!.hash) + 1 : 0,
  )

  // Render one page into the bottom scene. Called ONLY on navigation / first data.
  function showPage(p: number) {
    const last = Math.max(0, Math.ceil(machines.length / PAGE) - 1)
    page = Math.max(0, Math.min(p, last))
    bottomVisible = machines.slice(page * PAGE, page * PAGE + PAGE)
    bottomHandle?.setMachines(bottomVisible, onSelect, true) // hold camera across pages
  }

  // Drop whole pages off the oldest end once history exceeds the cap, keeping `page` pointing at
  // the same logical page (shifted down) so the current view doesn't jump.
  function trimHistory() {
    const cap = MAX_HISTORY_PAGES * PAGE
    if (machines.length <= cap) return
    const dropPages = Math.floor((machines.length - cap) / PAGE)
    if (dropPages < 1) return
    machines.splice(0, dropPages * PAGE)
    machineCount = machines.length
    showPage(page - dropPages)
  }

  // Per spec: "scroll up / arrow left -> towards latest". Latest = highest index (from oldest),
  // so newer = page + 1, older = page - 1.
  const newer = () => showPage(page + 1)
  const older = () => showPage(page - 1)
  const resetView = () => {
    topHandle?.resetView()
    bottomHandle?.resetView()
  }

  function onBatch(buf: ArrayBuffer) {
    let batch: Machine[]
    try {
      batch = parseCompactData('live#' + batches, buf) // unique hash namespace per batch
    } catch (e) {
      console.warn('dropped malformed batch:', e)
      return
    }
    batches += 1
    if (!batch.length) return
    // Apply any metadata that arrived ahead of this geometry.
    if (Object.keys(orphanMeta).length) applyTraining(batch, orphanMeta)
    const firstEver = machines.length === 0
    // LAG FIX 2: a loop, not `push(...batch)`. Spreading into arguments throws
    // "RangeError: Maximum call stack size exceeded" on a large enough batch - measured in V8:
    // fine at 100,000, throws at 200,000 - and the backlog frame a reconnect receives is
    // exactly the one big enough to hit it. trimHistory() runs AFTER the append, so the 2,500
    // cap never protected this. The loop form did 200,000 in 3.3 ms.
    for (const m of batch) machines.push(m)
    machineCount = machines.length
    trimHistory()
    latestBatch = batch.slice(-PAGE)
    topHandle?.setMachines(latestBatch, onSelect, true) // hold camera across batches
    if (firstEver) showPage(Math.ceil(machines.length / PAGE) - 1) // start bottom at newest page
  }

  // The second frame kind: training metadata and run stats, as JSON text. Matched onto machines
  // by the candidate id the hub stamped into each binary record.
  function onMeta(text: string) {
    let frame: TrainingFrame & Partial<AnimationFrame>
    try {
      frame = JSON.parse(text) as TrainingFrame & Partial<AnimationFrame>
    } catch (e) {
      console.warn('dropped malformed metadata frame:', e)
      return
    }
    // Two text frames share this channel. An animation reply carries the key even when the
    // answer is a failure, so `in` is the discriminator rather than truthiness.
    if ('animation' in frame) {
      onAnimation(frame as AnimationFrame)
      return
    }
    if (frame.run) run = frame.run
    const meta = frame.machines ?? {}
    if (!Object.keys(meta).length) return
    // Apply to the whole history, not just the last batch: on a backfill the metadata frame
    // covers everything that just arrived, and after a page turn the older machines are the
    // ones on screen.
    applyTraining(machines, meta)
    applyTraining(latestBatch, meta)
    orphanMeta = { ...orphanMeta, ...meta }
    // Labels are baked into the scene at build time, so a metadata frame that renamed anything
    // needs a rebuild to show it. Cheap: <=25 machines, camera held.
    topHandle?.setMachines(latestBatch, onSelect, true)
    bottomHandle?.setMachines(bottomVisible, onSelect, true)
  }

  function connect() {
    if (ws) return
    clearRetry()
    manualClose = false
    status = 'connecting'
    try {
      ws = new WebSocket(url)
    } catch (e) {
      status = 'error'
      ws = null
      if (reconnect) scheduleRetry()
      return
    }
    ws.binaryType = 'arraybuffer'
    // Reset on open (not on drop): a fresh session's backlog fully replaces the view, and a
    // failed reconnect leaves the last-seen machines on screen instead of blanking them.
    ws.onopen = () => {
      status = 'connected'
      everConnected = true
      machines = []
      machineCount = 0
      latestBatch = []
      bottomVisible = []
      orphanMeta = {}
      run = {}
      page = 0
      batches = 0
    }
    // One socket, two frame kinds: binary geometry and JSON metadata. `e.data` is a string for
    // text frames and an ArrayBuffer for binary ones, which is the whole discriminator needed.
    ws.onmessage = (e) =>
      typeof e.data === 'string' ? onMeta(e.data) : onBatch(e.data as ArrayBuffer)
    ws.onerror = () => (status = 'error')
    ws.onclose = () => {
      status = 'closed'
      ws = null
      if (reconnect && !manualClose) scheduleRetry()
    }
  }

  function disconnect() {
    manualClose = true
    clearRetry()
    ws?.close()
    ws = null
    status = 'idle'
  }

  // Toggling the box on while already dropped should start trying immediately.
  function onReconnectToggle() {
    if (reconnect && !ws && (status === 'closed' || status === 'error')) scheduleRetry()
    else if (!reconnect) clearRetry()
  }

  // Shift+scroll pages history (and must beat OrbitControls' wheel-zoom -> capture + stopPropagation).
  function onWheel(e: WheelEvent) {
    if (!e.shiftKey) return
    e.preventDefault()
    e.stopPropagation()
    if (e.deltaY < 0) newer()
    else if (e.deltaY > 0) older()
  }

  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return
    if (e.key === 'ArrowLeft') newer()
    else if (e.key === 'ArrowRight') older()
    else return
    e.preventDefault()
  }

  onMount(() => {
    try {
      topHandle = createScene(topContainer)
      bottomHandle = createScene(bottomContainer)
    } catch (e) {
      status = 'error'
      return
    }
    bottomContainer.addEventListener('wheel', onWheel, { capture: true, passive: false })
    window.addEventListener('keydown', onKey)
    // Connect without being asked. Nothing on this page is useful disconnected, and the flow
    // that opened it (train.bat -> viewer + training) has no way to press a button. A failure
    // here is not an error state to sit in: `reconnect` defaults on, so it keeps trying at the
    // startup cadence until the loop finishes binding.
    connect()
    const onFullscreenChange = () => (isFullscreen = document.fullscreenElement === rootEl)
    document.addEventListener('fullscreenchange', onFullscreenChange)
    return () => {
      window.removeEventListener('keydown', onKey)
      bottomContainer.removeEventListener('wheel', onWheel, { capture: true })
      document.removeEventListener('fullscreenchange', onFullscreenChange)
      clearRetry()
      manualClose = true
      ws?.close()
      ws = null
      topHandle?.dispose()
      bottomHandle?.dispose()
    }
  })

  const statusColor = $derived(
    status === 'connected'
      ? 'bg-emerald-400 text-slate-900'
      : status === 'connecting'
        ? 'bg-amber-400 text-slate-900'
        : status === 'error'
          ? 'bg-red-500 text-white'
          : 'bg-slate-700 text-slate-300',
  )

  // A browser never shows the "accept this certificate" prompt for a wss:// WebSocket the way it
  // does for a plain https:// page load - a self-signed dev cert has to be trusted by visiting
  // this https:// equivalent directly at least once first, or every wss:// connect just fails
  // silently (no detail in the error event).
  const httpsEquivalent = $derived(url.replace(/^wss:\/\//, 'https://'))

  // Chrome/Edge/Brave/Opera 147+ gate any request from a public page to a private-network
  // address (localhost/127.0.0.1/::1/RFC1918) behind a "wants to connect to devices on your
  // local network" permission prompt - remembered per-origin, but silently blocks every attempt
  // until granted (or re-granted, if a prior attempt was dismissed/blocked). This only fires
  // when *this* page's own origin is itself public, so it never affects a page served from
  // localhost.
  const targetsPrivateHost = $derived(/^wss?:\/\/(localhost|127\.0\.0\.1|\[?::1\]?)(:|\/|$)/i.test(url))
  const pageIsPublicOrigin =
    typeof location !== 'undefined' && !/^(localhost|127\.0\.0\.1|\[?::1\]?)$/i.test(location.hostname)
</script>

<div bind:this={rootEl} class="absolute inset-0 flex flex-col bg-slate-950 text-slate-200">
  <!-- Control bar -->
  <div
    class="flex shrink-0 flex-wrap items-center gap-3 border-b border-slate-800 bg-slate-900/80 px-4 py-2 text-xs"
  >
    {#if connected}
      <button
        class="rounded bg-slate-700 px-3 py-1 font-medium text-slate-100 hover:bg-slate-600"
        onclick={disconnect}>Disconnect</button
      >
    {:else}
      <button
        class="rounded bg-cyan-400 px-3 py-1 font-medium text-slate-900 hover:bg-cyan-300"
        onclick={connect}>Connect</button
      >
    {/if}

    <input
      class="w-56 rounded border border-slate-700 bg-slate-800 px-2 py-1 font-mono text-slate-200 disabled:opacity-50"
      bind:value={url}
      disabled={connected}
      spellcheck="false"
      aria-label="WebSocket URL"
    />

    <span class="rounded px-2 py-0.5 font-medium {statusColor}">{status}</span>

    <label
      class="flex cursor-pointer items-center gap-1 text-slate-400"
      title="Keep trying until the training run appears, then every 30s if it drops"
    >
      <input type="checkbox" bind:checked={reconnect} onchange={onReconnectToggle} />
      auto-reconnect{#if retryPending}<span class="text-amber-400">
          · retrying{everConnected ? ' in 30s' : '…'}</span
        >{/if}
    </label>

    <span class="text-slate-400">
      {batches} batch{batches === 1 ? '' : 'es'} · {machineCount.toLocaleString()} machines
    </span>

    <!-- History pager -->
    <div class="ml-auto flex items-center gap-2">
      <button
        class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
        title="Toggle fullscreen"
        onclick={toggleFullscreen}>{isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}</button
      >
      <span class="text-slate-700">|</span>
      <button
        class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
        title="Re-frame both views"
        onclick={resetView}>Reset view</button
      >
      <span class="text-slate-700">|</span>
      <button
        class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700 disabled:opacity-40"
        title="Towards latest (← / shift-scroll up)"
        disabled={page >= totalPages - 1}
        onclick={newer}>◀ Newer</button
      >
      <span class="flex items-center gap-1 tabular-nums text-slate-400">
        page
        <input
          type="number"
          min="1"
          max={totalPages}
          class="w-12 rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-center text-slate-200"
          value={page + 1}
          onchange={(e) => showPage(Number((e.currentTarget as HTMLInputElement).value) - 1)}
          aria-label="Jump to page"
        />
        / {totalPages}
      </span>
      <button
        class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700 disabled:opacity-40"
        title="Towards oldest (→ / shift-scroll down)"
        disabled={page <= 0}
        onclick={older}>Older ▶</button
      >
      <span class="text-[10px] text-slate-500">shift+scroll / ←→</span>
    </div>
  </div>

  <!-- The loop's own round row, on the page instead of only in the console. Hidden until the
       first round reports, so a fresh connection does not show a strip full of zeros. -->
  {#if run.round}
    <div
      class="flex shrink-0 flex-wrap items-center gap-x-5 gap-y-1 border-b border-slate-800 bg-slate-900/50 px-4 py-1.5 text-[11px] text-slate-400"
    >
      {#snippet stat(label: string, value: string | number, tone = 'text-slate-200')}
        <span class="whitespace-nowrap"
          >{label} <span class="font-mono tabular-nums {tone}">{value}</span></span
        >
      {/snippet}
      {@render stat('round', `${run.round} / ${run.rounds ?? '?'}`)}
      {@render stat('functional /1k', (run.model_per_1k ?? 0).toFixed(2), 'text-emerald-300')}
      {@render stat('control /1k', (run.control_per_1k ?? 0).toFixed(2))}
      {@render stat('stripped', run.stripped ?? 0, 'text-amber-300')}
      {@render stat('library', run.library ?? 0)}
      {@render stat('attempts', (run.attempts ?? 0).toLocaleString())}
      {@render stat('replay', (run.replay ?? 0).toLocaleString())}
      {#if run.stopped}
        {@render stat('stopped', run.stopped)}
        {@render stat('depth', (run.mean_depth ?? 0).toFixed(2))}
      {/if}
      <span class="ml-auto whitespace-nowrap text-slate-600">round took {run.seconds ?? 0}s</span>
    </div>
  {/if}

  {#snippet stepper(list: Machine[], handle: SceneHandle | null, index: number)}
    {#if list.length}
      <div
        class="absolute bottom-2 left-2 flex items-center gap-1 rounded-lg bg-slate-900/80 p-1 text-[11px] text-slate-300"
      >
        <button class="rounded px-1.5 py-0.5 hover:bg-slate-700" onclick={() => stepMachine(list, handle, -1)}
          >◀</button
        >
        <input
          type="number"
          min="1"
          max={list.length}
          placeholder="#"
          class="w-9 rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-center text-slate-200"
          value={index || ''}
          onchange={(e) => jumpMachine(list, handle, Number((e.currentTarget as HTMLInputElement).value))}
          aria-label="Jump to machine"
        />
        <span class="text-slate-500">/ {list.length}</span>
        <button class="rounded px-1.5 py-0.5 hover:bg-slate-700" onclick={() => stepMachine(list, handle, 1)}
          >▶</button
        >
      </div>
    {/if}
  {/snippet}

  <!-- Resizable split: latest streamed batch (top) / paged history (bottom) -->
  <div bind:this={splitContainer} class="relative flex min-h-0 flex-1 flex-col">
    <div class="relative min-h-0 overflow-hidden" style="flex-grow: {splitPct}; flex-basis: 0;">
      <div bind:this={topContainer} class="absolute inset-0"></div>
      <span
        class="pointer-events-none absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] text-emerald-300"
      >
        Latest batch · {latestBatch.length} machine{latestBatch.length === 1 ? '' : 's'}
      </span>
      {@render stepper(latestBatch, topHandle, topIndex)}
      {#if status === 'idle' || (status !== 'connected' && !machineCount)}
        <div
          class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 text-center text-sm text-slate-500"
        >
          {#if status === 'error'}
            <span>Connection failed — is the stream server running?</span>
            {#if url.startsWith('wss://')}
              <span class="max-w-md text-xs">
                Self-signed dev cert? Open
                <a
                  class="pointer-events-auto text-cyan-400 underline"
                  href={httpsEquivalent}
                  target="_blank"
                  rel="noopener">{httpsEquivalent}</a
                >
                once, accept the warning, then Connect again.
              </span>
            {/if}
            {#if targetsPrivateHost && pageIsPublicOrigin}
              <span class="max-w-md text-xs">
                Viewing a hosted page? Chrome/Edge 147+ block a public page from reaching your
                local network unless you approve it — look for a "wants to connect to devices on
                your local network" prompt, or the address bar's lock icon → Site settings → Local
                network access.
              </span>
            {/if}
          {:else if connected || retryPending}
            <span>Waiting for the training run at <span class="font-mono">{url}</span>…</span>
            <span class="text-xs">It appears here as soon as the loop starts streaming.</span>
          {:else}
            <span>Connect to a training stream to see live machines.</span>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Drag to resize the split between the two viewports -->
    <div
      class="relative h-1.5 shrink-0 cursor-row-resize bg-slate-800 hover:bg-cyan-500/60"
      role="separator"
      aria-orientation="horizontal"
      aria-label="Resize latest batch / history split"
      onpointerdown={onSplitDown}
      onpointermove={onSplitMove}
      onpointerup={onSplitUp}
      onpointercancel={onSplitUp}
    ></div>

    <div
      class="relative min-h-0 overflow-hidden border-t border-slate-800"
      style="flex-grow: {100 - splitPct}; flex-basis: 0;"
    >
      <div bind:this={bottomContainer} class="absolute inset-0"></div>
      <span
        class="pointer-events-none absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] text-cyan-300"
      >
        History · page {page + 1} / {totalPages}
      </span>
      {@render stepper(bottomVisible, bottomHandle, bottomIndex)}
    </div>
  </div>

  <MachineDetailPanel
    machine={selected}
    onClose={() => onSelect(null)}
    onSimulate={requestAnimation}
    simulating={awaitingId !== null}
    error={animationError}
  />

  <!-- The player. An overlay rather than a third viewport: it is one machine at a time, asked
       for deliberately, and it should not permanently cost the grid any room. -->
  {#if animating}
    <div class="absolute inset-0 z-20 flex flex-col bg-slate-950/95">
      <div
        class="flex shrink-0 items-center gap-3 border-b border-slate-800 px-4 py-2 text-xs text-slate-300"
      >
        <span class="font-semibold text-slate-100">{animating.label}</span>
        <span class="text-slate-500">{animating.terminationTick ?? 0} ticks</span>
        <button
          class="rounded bg-cyan-400 px-3 py-1 font-medium text-slate-900 hover:bg-cyan-300"
          onclick={togglePlay}>{playing ? '❚❚ Pause' : '▶ Play'}</button
        >
        <button
          class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
          onclick={() => {
            playerHandle?.stepTick(-1)
            playing = false
          }}>◀ tick</button
        >
        <button
          class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700"
          onclick={() => {
            playerHandle?.stepTick(1)
            playing = false
          }}>tick ▶</button
        >
        <label class="flex items-center gap-1 text-slate-400">
          speed
          <input
            type="range"
            min="40"
            max="600"
            step="20"
            bind:value={speed}
            oninput={() => playerHandle?.setSpeed(speed)}
            class="w-28"
          />
          <span class="w-12 font-mono">{speed}ms</span>
        </label>
        <button
          class="ml-auto rounded px-2 py-1 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
          onclick={closePlayer}>✕ Close</button
        >
      </div>
      <div bind:this={playerContainer} class="relative min-h-0 flex-1"></div>
    </div>
  {/if}
</div>
