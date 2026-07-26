<script lang="ts">
  import { onMount } from 'svelte'
  import { parseGeneratorRecords, type Machine, type MachineEvent } from './lib/data'
  import { createAnimatedScene, type AnimatedSceneHandle } from './lib/animatedScene'

  // Cap so a long-running preview session doesn't grow the list forever - drop the oldest once
  // past this.
  const MAX_MACHINES = 500

  let animContainer: HTMLDivElement
  let animHandle: AnimatedSceneHandle | null = null

  let machines = $state<Machine[]>([])
  let selectedHash = $state<string | null>(null)
  let received = $state(0)

  // Tick/subtick stepper state - polled each frame from animHandle (which runs its own rAF loop
  // independent of Svelte's reactivity) rather than duplicated as the source of truth.
  let eventIndexUI = $state(-1)
  let eventsUI = $state<MachineEvent[]>([])
  let playingUI = $state(true)
  let tickInputUI = $state(0)
  let speedInputUI = $state(150)
  const currentEventUI = $derived(eventIndexUI >= 0 ? (eventsUI[eventIndexUI] ?? null) : null)
  const currentTickUI = $derived(currentEventUI ? currentEventUI.tick : 0)
  $effect(() => {
    tickInputUI = currentTickUI
  })

  let url = $state('ws://localhost:8766')
  let status = $state<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle')
  let ws: WebSocket | null = null
  const connected = $derived(status === 'connecting' || status === 'connected')
  const selected = $derived(machines.find((m) => m.hash === selectedHash) ?? null)

  // The one viewport shows exactly the fixture selected in the list below it - no separate
  // "browsing" scene showing something else while a different machine animates elsewhere.
  function select(m: Machine) {
    selectedHash = m.hash
    animHandle?.loadMachine(m)
  }

  function onFrame(text: string) {
    let batch: Machine[]
    try {
      batch = parseGeneratorRecords(text)
    } catch (e) {
      console.warn('dropped malformed frame:', e)
      return
    }
    if (!batch.length) return
    received += batch.length
    const firstEver = machines.length === 0
    machines.push(...batch) // Svelte 5 $state array stays reactive in place, no O(n) copy needed
    if (machines.length > MAX_MACHINES) machines.splice(0, machines.length - MAX_MACHINES)
    if (firstEver) select(batch[0])
  }

  function connect() {
    if (ws) return
    status = 'connecting'
    try {
      ws = new WebSocket(url)
    } catch (e) {
      status = 'error'
      ws = null
      return
    }
    ws.onopen = () => {
      status = 'connected'
      machines = []
      received = 0
      selectedHash = null
    }
    ws.onmessage = (e) => onFrame(e.data as string)
    ws.onerror = () => (status = 'error')
    ws.onclose = () => {
      status = 'closed'
      ws = null
    }
  }

  function disconnect() {
    ws?.close()
    ws = null
    status = 'idle'
  }

  function stepSubtick(delta: number) {
    animHandle?.stepSubtick(delta)
  }
  function jumpToTickInput() {
    animHandle?.jumpToTick(tickInputUI)
  }
  function tickPrev() {
    animHandle?.stepTick(-1)
  }
  function tickNext() {
    animHandle?.stepTick(1)
  }
  function togglePlay() {
    if (playingUI) animHandle?.pause()
    else animHandle?.play()
  }
  function applySpeed() {
    animHandle?.setSpeed(speedInputUI)
  }

  // Scroll wheel over the tick/subtick controls scrubs time: down = backward, up = forward -
  // preventDefault so the page/list underneath doesn't scroll while doing it.
  function onTickWheel(e: WheelEvent) {
    e.preventDefault()
    if (e.deltaY > 0) tickPrev()
    else if (e.deltaY < 0) tickNext()
  }
  function onSubtickWheel(e: WheelEvent) {
    e.preventDefault()
    if (e.deltaY > 0) stepSubtick(-1)
    else if (e.deltaY < 0) stepSubtick(1)
  }

  let pollRaf = 0
  function pollStepper() {
    pollRaf = requestAnimationFrame(pollStepper)
    if (!animHandle) return
    eventIndexUI = animHandle.getEventIndex()
    eventsUI = animHandle.getEvents()
    playingUI = animHandle.isPlaying()
  }

  onMount(() => {
    animHandle = createAnimatedScene(animContainer)
    speedInputUI = animHandle.getSpeed()
    pollStepper()
    return () => {
      ws?.close()
      ws = null
      cancelAnimationFrame(pollRaf)
      animHandle?.dispose()
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

<div class="absolute inset-0 flex flex-col bg-slate-950 text-slate-200">
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
      class="w-64 rounded border border-slate-700 bg-slate-800 px-2 py-1 font-mono text-slate-200 disabled:opacity-50"
      bind:value={url}
      disabled={connected}
      spellcheck="false"
      aria-label="WebSocket URL"
    />

    <span class="rounded px-2 py-0.5 font-medium {statusColor}">{status}</span>
    <span class="text-slate-400">{received.toLocaleString()} machine{received === 1 ? '' : 's'} received</span>
  </div>

  <div class="relative flex min-h-0 flex-1">
    <div class="relative min-h-0 flex-1">
      <div bind:this={animContainer} class="absolute inset-0"></div>
      <span
        class="pointer-events-none absolute right-2 top-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] text-cyan-300"
      >
        {#if selected}
          Viewing {selected.label ?? selected.hash}{playingUI ? ' (looping)' : ' (paused)'}
        {:else}
          Select a machine from the list to animate its piston logic
        {/if}
      </span>

      <!-- Tick/subtick stepper - lets you walk the machine's exact logged event order (tick +
      order/"subtick") one step at a time, to verify the simulator's event sequencing rather than
      just watching it loop. Backward snaps instantly; forward animates into place (see
      animatedScene.ts's setEventIndex). High z-index + opaque-ish background + border so it can
      never be visually lost against the 3D viewport behind it. -->
      <div
        class="absolute left-2 top-2 z-50 flex flex-col gap-1.5 rounded-lg border border-slate-600 bg-slate-900 p-2.5 text-[11px] text-slate-200 shadow-lg"
      >
        <div class="flex items-center gap-1" onwheel={onTickWheel} title="scroll to step ticks">
          <span class="w-14 text-slate-400">tick</span>
          <button class="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600" onclick={tickPrev} title="previous tick">◀</button>
          <input
            type="number"
            class="w-16 rounded border border-slate-700 bg-slate-800 px-1 py-1 text-center font-mono text-slate-200"
            bind:value={tickInputUI}
            onkeydown={(e) => e.key === 'Enter' && jumpToTickInput()}
            onblur={jumpToTickInput}
            aria-label="jump to tick"
          />
          <button class="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600" onclick={tickNext} title="next tick">▶</button>
        </div>
        <div class="flex items-center gap-1" onwheel={onSubtickWheel} title="scroll to step subticks">
          <span class="w-14 text-slate-400">subtick</span>
          <button
            class="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600"
            onclick={() => stepSubtick(-1)}
            title="previous event"
          >
            ◀
          </button>
          <span class="w-16 text-center font-mono text-slate-300">
            {eventIndexUI + 1}/{eventsUI.length}
          </span>
          <button class="rounded bg-slate-700 px-2 py-1 hover:bg-slate-600" onclick={() => stepSubtick(1)} title="next event">
            ▶
          </button>
        </div>
        <div class="flex items-center gap-1">
          <span class="w-14 text-slate-400">speed</span>
          <input
            type="number"
            min="10"
            step="10"
            class="w-16 rounded border border-slate-700 bg-slate-800 px-1 py-1 text-center font-mono text-slate-200"
            bind:value={speedInputUI}
            onkeydown={(e) => e.key === 'Enter' && applySpeed()}
            onblur={applySpeed}
            aria-label="ms per tick"
          />
          <span class="text-slate-500">ms/tick</span>
          <button
            class="ml-auto rounded bg-cyan-400 px-2 py-1 font-medium text-slate-900 hover:bg-cyan-300"
            onclick={togglePlay}
          >
            {playingUI ? 'Pause' : 'Restart'}
          </button>
        </div>
        <div class="whitespace-nowrap font-mono">
          {#if currentEventUI}
            tick {currentEventUI.tick} &middot; order {currentEventUI.order} &middot;
            <span
              class:text-red-400={currentEventUI.kind === 'pistonBlocked'}
              class:text-rose-300={currentEventUI.kind === 'observerFired'}
              class:text-cyan-300={currentEventUI.kind !== 'pistonBlocked' && currentEventUI.kind !== 'observerFired'}
              >{currentEventUI.kind}</span
            >
            &middot; block #{currentEventUI.blockIndex}
          {:else}
            <span class="text-slate-500">tick 0 &middot; initial state ({eventsUI.length} events)</span>
          {/if}
        </div>
      </div>
      {#if status === 'idle' || (status === 'error' && !machines.length)}
        <div
          class="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-1 text-center text-sm text-slate-500"
        >
          {#if status === 'error'}
            <span>Connection failed - is a generator/stream_*.py script running?</span>
          {:else}
            <span>Connect to a generator stream to see machines.</span>
          {/if}
        </div>
      {/if}
    </div>

    <div class="w-56 shrink-0 overflow-y-auto border-l border-slate-800 bg-slate-900/60">
      {#each machines as m (m.hash)}
        <button
          class="block w-full truncate px-3 py-1.5 text-left text-xs hover:bg-slate-800
            {m.hash === selectedHash ? 'bg-cyan-400/20 text-cyan-300' : 'text-slate-300'}"
          onclick={() => select(m)}
          title={m.label ?? m.hash}
        >
          {m.label ?? m.hash}
        </button>
      {:else}
        <div class="p-3 text-xs text-slate-600">no machines yet</div>
      {/each}
    </div>
  </div>
</div>
