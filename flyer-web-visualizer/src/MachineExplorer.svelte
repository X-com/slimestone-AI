<script lang="ts">
  import { onMount } from 'svelte'
  import { loadMachines, loadComplexMachines, parseCompactData, type Machine } from './lib/data'
  import { createScene, type SceneHandle } from './lib/scene'
  import MachineDetailPanel from './lib/MachineDetailPanel.svelte'

  const PAGE = 25
  const UPLOAD_CAP = 25 // cap rendered uploads: one DOM label per machine gets heavy past this

  let container: HTMLDivElement
  let handle: SceneHandle | null = null

  let all = $state<Machine[]>([])
  let uploaded = $state<Machine[]>([])
  let complex = $state<Machine[]>([]) // large hand-authored examples (public/machines/)
  let complexIdx = $state(0)
  let visible = $state<Machine[]>([])
  let mode = $state<'first' | 'last' | 'random' | 'uploaded' | 'complex'>('first')
  let selected = $state<Machine | null>(null)
  let error = $state<string | null>(null)
  let loading = $state(true)

  // Top-level tab: "Complex Examples" vs everything archive/upload ("Examples").
  const tab = $derived(mode === 'complex' ? 'complex' : 'examples')

  function shuffle100(list: Machine[]): Machine[] {
    const a = [...list]
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a.slice(0, PAGE)
  }

  function pick(next: 'first' | 'last' | 'random') {
    mode = next
    selected = null
    if (next === 'first') visible = all.slice(0, PAGE)
    else if (next === 'last') visible = all.slice(-PAGE)
    else visible = shuffle100(all)
  }

  function showUploaded() {
    mode = 'uploaded'
    selected = null
    visible = uploaded.slice(0, UPLOAD_CAP)
  }

  // Complex tab: enter it, or (if already there) cycle to the next example. One at a time.
  function complexTab() {
    if (!complex.length) return
    complexIdx = mode === 'complex' ? (complexIdx + 1) % complex.length : complexIdx
    mode = 'complex'
    visible = [complex[complexIdx]]
    selected = complex[complexIdx] // "select one" — open its detail panel by default
  }

  function examplesTab() {
    if (mode === 'complex') pick('first')
  }

  async function onUpload(e: Event) {
    const input = e.target as HTMLInputElement
    const files = [...(input.files ?? [])]
    input.value = '' // let the same file be re-picked later
    if (!files.length) return
    try {
      const parsed: Machine[] = []
      for (const f of files) parsed.push(...parseCompactData(f.name, await f.arrayBuffer()))
      if (!parsed.length) throw new Error('no machines found in file(s)')
      uploaded = [...uploaded, ...parsed]
      error = null
      showUploaded()
    } catch (err) {
      error = String(err)
    }
  }

  function clearUploads() {
    uploaded = []
    pick('first')
  }

  // Camera elevation snaps (keeping the current azimuth).
  const ISO_EL = 35 // ~true isometric elevation
  const TOP_EL = 89
  let viewEl = $state(ISO_EL)

  function isoView() {
    viewEl = ISO_EL
    handle?.setElevation(ISO_EL)
  }
  function topView() {
    viewEl = TOP_EL
    handle?.setElevation(TOP_EL)
  }

  // Left/right arrows step through the visible machines, centering + framing each.
  function selectDelta(step: number) {
    if (!visible.length) return
    const i = selected ? visible.findIndex((m) => m.hash === selected!.hash) : -1
    const n = visible.length
    const next = i < 0 ? (step > 0 ? 0 : n - 1) : (i + step + n) % n
    selected = visible[next]
    handle?.focusSelected(visible[next].hash)
  }

  // Type a 1-based index (up to PAGE) to jump straight to that machine.
  function selectIndex(oneIndexed: number) {
    if (!visible.length) return
    const idx = Math.min(visible.length, Math.max(1, Math.round(oneIndexed) || 1)) - 1
    selected = visible[idx]
    handle?.focusSelected(visible[idx].hash)
  }
  const selectedIndex = $derived(selected ? visible.findIndex((m) => m.hash === selected!.hash) + 1 : 0)

  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return
    if (e.key === 'ArrowLeft') selectDelta(-1)
    else if (e.key === 'ArrowRight') selectDelta(1)
    else if (e.key === 'ArrowUp') topView()
    else if (e.key === 'ArrowDown') isoView()
    else return
    e.preventDefault()
  }

  onMount(() => {
    try {
      handle = createScene(container)
    } catch (e) {
      error = `WebGL unavailable: ${e}`
      loading = false
      return
    }
    window.addEventListener('keydown', onKey)
    Promise.all([
      loadMachines().then((m) => (all = m)),
      loadComplexMachines()
        .then((c) => (complex = c))
        .catch(() => (complex = [])), // missing examples shouldn't block the archive
    ])
      .then(() => {
        if (complex.length) complexTab() // default: show/select the first complex example
        else pick('first')
      })
      .catch((e) => (error = String(e)))
      .finally(() => (loading = false))
    return () => {
      window.removeEventListener('keydown', onKey)
      handle?.dispose()
    }
  })

  // Push the visible set into the scene whenever it changes.
  $effect(() => {
    if (handle && visible.length) handle.setMachines(visible, (m) => (selected = m))
  })
  // Sync selection highlight.
  $effect(() => {
    handle?.setSelected(selected?.hash ?? null)
  })
</script>

<div bind:this={container} class="absolute inset-0"></div>

<!-- Header -->
<header
  class="absolute left-0 top-0 flex items-center gap-4 px-4 py-3 text-slate-200"
>
  <h1 class="text-sm font-semibold tracking-wide">Flyer Machines</h1>

  <!-- Top tabs: Complex Examples (cycle) vs Examples (archive/upload) -->
  <div class="flex gap-1">
    <button
      class="rounded px-2.5 py-1 text-xs font-medium transition
        {tab === 'complex'
        ? 'bg-fuchsia-400 text-slate-900'
        : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'}"
      title="Large hand-authored machines — click to cycle through them"
      disabled={!complex.length}
      onclick={complexTab}
    >
      Complex Examples{#if mode === 'complex' && complex.length > 1}
        &nbsp;{complexIdx + 1}/{complex.length} ↻{/if}
    </button>
    <button
      class="rounded px-2.5 py-1 text-xs font-medium transition
        {tab === 'examples'
        ? 'bg-cyan-400 text-slate-900'
        : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'}"
      title="Archive machines from ga_archive.jsonl (+ .data uploads)"
      onclick={examplesTab}
    >
      Examples
    </button>
  </div>

  {#if tab === 'examples'}
  <div class="flex gap-1">
    {#each [['first', `First ${PAGE}`], ['last', `Last ${PAGE}`], ['random', `Random ${PAGE}`]] as [m, label] (m)}
      <button
        class="rounded px-2.5 py-1 text-xs font-medium transition
          {mode === m
          ? 'bg-cyan-400 text-slate-900'
          : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'}"
        onclick={() => pick(m as 'first' | 'last' | 'random')}
      >
        {label}{mode === m && m === 'random' ? ' ↻' : ''}
      </button>
    {/each}

    <!-- Upload .data (compact binary candidates) -->
    <label
      class="cursor-pointer rounded bg-slate-800/80 px-2.5 py-1 text-xs font-medium text-slate-300 transition hover:bg-slate-700"
      title="Load machines from .data files (compact binary format)"
    >
      Upload .data
      <input type="file" accept=".data" multiple class="hidden" onchange={onUpload} />
    </label>

    {#if uploaded.length}
      <button
        class="rounded px-2.5 py-1 text-xs font-medium transition
          {mode === 'uploaded'
          ? 'bg-cyan-400 text-slate-900'
          : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'}"
        onclick={showUploaded}
      >
        Uploaded ({uploaded.length})
      </button>
      <button
        class="rounded px-2 py-1 text-xs text-slate-400 hover:bg-slate-800 hover:text-slate-100"
        title="Clear uploaded machines"
        onclick={clearUploads}>✕</button
      >
    {/if}
  </div>
  {/if}
  {#if !loading && !error}
    <span class="text-xs text-slate-400">
      {#if mode === 'complex'}
        {complex[complexIdx]?.candidate.name ?? complex[complexIdx]?.source} · {complex[
          complexIdx
        ]?.block_count.toLocaleString()} blocks
      {:else if mode === 'uploaded'}
        showing {visible.length} of {uploaded.length} uploaded{uploaded.length >
        UPLOAD_CAP
          ? ` (cap ${UPLOAD_CAP})`
          : ''}
      {:else}
        showing {visible.length} of {all.length}
      {/if}
    </span>
  {/if}
  {#if loading}<span class="text-xs text-slate-400">loading…</span>{/if}
  {#if error}<span class="text-xs text-red-400">{error}</span>{/if}
</header>

<!-- Camera controls (bottom-left, clear of the right-side detail panel) -->
<div
  class="absolute bottom-3 left-3 flex items-center gap-1 rounded-lg bg-slate-900/80 p-1.5 text-xs text-slate-300"
>
  <button
    class="rounded px-2 py-1 hover:bg-slate-700"
    title="Previous machine (←)"
    onclick={() => selectDelta(-1)}>◀</button
  >
  <button
    class="rounded px-2 py-1 {viewEl === ISO_EL
      ? 'bg-cyan-400 text-slate-900'
      : 'hover:bg-slate-700'}"
    title="Isometric view (↓)"
    onclick={isoView}>Iso</button
  >
  <button
    class="rounded px-2 py-1 {viewEl === TOP_EL
      ? 'bg-cyan-400 text-slate-900'
      : 'hover:bg-slate-700'}"
    title="Top-down view (↑)"
    onclick={topView}>Top</button
  >
  <button
    class="rounded px-2 py-1 hover:bg-slate-700"
    title="Next machine (→)"
    onclick={() => selectDelta(1)}>▶</button
  >
  {#if visible.length}
    <input
      type="number"
      min="1"
      max={visible.length}
      placeholder="#"
      class="w-10 rounded border border-slate-700 bg-slate-800 px-1 py-0.5 text-center text-slate-200"
      value={selectedIndex || ''}
      onchange={(e) => selectIndex(Number((e.currentTarget as HTMLInputElement).value))}
      aria-label="Jump to machine"
    />
    <span class="text-slate-500">/ {visible.length}</span>
  {/if}
  <span class="px-1 text-[10px] text-slate-500">◀ ▶ machines · type # · drag = orbit</span>
</div>

<!-- Detail panel -->
<MachineDetailPanel machine={selected} onClose={() => (selected = null)} />
