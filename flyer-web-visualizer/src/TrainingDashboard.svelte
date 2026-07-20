<script lang="ts">
  import { onMount } from 'svelte'
  import { parseCompactData, type Machine } from './lib/data'
  import { createScene, type SceneHandle } from './lib/scene'
  import MachineDetailPanel from './lib/MachineDetailPanel.svelte'

  const PAGE = 25
  const MAX_HISTORY_PAGES = 100 // drop oldest machines beyond this so a long session doesn't grow forever

  let topContainer: HTMLDivElement
  let bottomContainer: HTMLDivElement
  let topHandle = $state<SceneHandle | null>(null)
  let bottomHandle = $state<SceneHandle | null>(null)

  // Flat history in arrival order (oldest -> newest). Never re-sliced into the bottom scene
  // except on explicit navigation, so a new batch never disturbs the page you're viewing.
  let machines = $state<Machine[]>([])
  let latestBatch = $state<Machine[]>([]) // last decoded batch, shown up top (render <=100)
  let bottomVisible = $state<Machine[]>([]) // current history page's machines, for the stepper
  let page = $state(0) // bottom page index, anchored from the OLDEST machine
  let batches = $state(0) // batches received this session (also the per-batch hash namespace)
  let selected = $state<Machine | null>(null)

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

  let url = $state('wss://localhost:8765')
  let status = $state<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle')
  let ws: WebSocket | null = null

  // Auto-reconnect: if a live connection drops, retry every RETRY_MS until it comes back.
  const RETRY_MS = 30_000
  let reconnect = $state(false)
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
    }, RETRY_MS)
  }

  const totalPages = $derived(Math.max(1, Math.ceil(machines.length / PAGE)))
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
    const firstEver = machines.length === 0
    machines.push(...batch) // in-place: Svelte 5 $state array stays reactive without an O(n) copy
    trimHistory()
    latestBatch = batch.slice(-PAGE)
    topHandle?.setMachines(latestBatch, onSelect, true) // hold camera across batches
    if (firstEver) showPage(Math.ceil(machines.length / PAGE) - 1) // start bottom at newest page
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
      machines = []
      latestBatch = []
      bottomVisible = []
      page = 0
      batches = 0
    }
    ws.onmessage = (e) => onBatch(e.data as ArrayBuffer)
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

    <label class="flex cursor-pointer items-center gap-1 text-slate-400" title="Retry every 30s if the connection drops">
      <input type="checkbox" bind:checked={reconnect} onchange={onReconnectToggle} />
      auto-reconnect{#if retryPending}<span class="text-amber-400"> · retrying in 30s</span>{/if}
    </label>

    <span class="text-slate-400">
      {batches} batch{batches === 1 ? '' : 'es'} · {machines.length.toLocaleString()} machines
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
      {#if status === 'idle' || (status !== 'connected' && !machines.length)}
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

  <MachineDetailPanel machine={selected} onClose={() => onSelect(null)} />
</div>
