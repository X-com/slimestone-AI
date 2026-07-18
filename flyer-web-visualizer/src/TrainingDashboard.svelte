<script lang="ts">
  import { onMount } from 'svelte'
  import { parseCompactData, type Machine } from './lib/data'
  import { createScene, type SceneHandle } from './lib/scene'

  const PAGE = 100

  let topContainer: HTMLDivElement
  let bottomContainer: HTMLDivElement
  let topHandle: SceneHandle | null = null
  let bottomHandle: SceneHandle | null = null

  // Flat history in arrival order (oldest -> newest). Never re-sliced into the bottom scene
  // except on explicit navigation, so a new batch never disturbs the page you're viewing.
  let machines = $state<Machine[]>([])
  let latestBatch = $state<Machine[]>([]) // last decoded batch, shown up top (render <=100)
  let page = $state(0) // bottom page index, anchored from the OLDEST machine
  let batches = $state(0) // batches received this session (also the per-batch hash namespace)

  let url = $state('ws://localhost:8765')
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

  const selectOn = (h: SceneHandle | null) => (m: Machine | null) =>
    h?.setSelected(m?.hash ?? null)

  // Render one page into the bottom scene. Called ONLY on navigation / first data.
  function showPage(p: number) {
    const last = Math.max(0, Math.ceil(machines.length / PAGE) - 1)
    page = Math.max(0, Math.min(p, last))
    const slice = machines.slice(page * PAGE, page * PAGE + PAGE)
    bottomHandle?.setMachines(slice, selectOn(bottomHandle), true) // hold camera across pages
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
    machines = [...machines, ...batch]
    latestBatch = batch.slice(-PAGE)
    topHandle?.setMachines(latestBatch, selectOn(topHandle), true) // hold camera across batches
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
    return () => {
      window.removeEventListener('keydown', onKey)
      bottomContainer.removeEventListener('wheel', onWheel, { capture: true })
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
</script>

<div class="absolute inset-0 flex flex-col text-slate-200">
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
      <span class="tabular-nums text-slate-400">page {page + 1} / {totalPages}</span>
      <button
        class="rounded bg-slate-800 px-2 py-1 hover:bg-slate-700 disabled:opacity-40"
        title="Towards oldest (→ / shift-scroll down)"
        disabled={page <= 0}
        onclick={older}>Older ▶</button
      >
      <span class="text-[10px] text-slate-500">shift+scroll / ←→</span>
    </div>
  </div>

  <!-- Top third: latest streamed batch -->
  <div class="relative min-h-0 flex-1 overflow-hidden">
    <div bind:this={topContainer} class="absolute inset-0"></div>
    <span
      class="pointer-events-none absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] text-emerald-300"
    >
      Latest batch · {latestBatch.length} machine{latestBatch.length === 1 ? '' : 's'}
    </span>
    {#if status === 'idle' || (status !== 'connected' && !machines.length)}
      <div
        class="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-500"
      >
        {status === 'error'
          ? 'Connection failed — is the stream server running?'
          : 'Connect to a training stream to see live machines.'}
      </div>
    {/if}
  </div>

  <!-- Bottom two-thirds: paged history -->
  <div class="relative min-h-0 flex-[2] overflow-hidden border-t border-slate-800">
    <div bind:this={bottomContainer} class="absolute inset-0"></div>
    <span
      class="pointer-events-none absolute left-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] text-cyan-300"
    >
      History · page {page + 1} / {totalPages}
    </span>
  </div>
</div>
