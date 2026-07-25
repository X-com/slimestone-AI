<script lang="ts">
  import { decodeState } from './blocks'
  import { formatEvent, type SimEvent } from './simlog'

  let {
    x,
    y,
    z,
    state,
    events,
    clientX,
    clientY,
  }: {
    x: number
    y: number
    z: number
    state: number
    events: SimEvent[] | null
    clientX: number
    clientY: number
  } = $props()

  const decoded = $derived(decodeState(state))
  // Keep the tooltip on-screen near the cursor without covering it.
  const left = $derived(Math.min(clientX + 14, (typeof window !== 'undefined' ? window.innerWidth : 2000) - 340))
  const top = $derived(Math.min(clientY + 14, (typeof window !== 'undefined' ? window.innerHeight : 2000) - 320))
</script>

<div
  class="pointer-events-none absolute z-20 w-80 max-w-[85vw] rounded-lg border border-slate-700 bg-slate-900/95 p-3 text-xs text-slate-200 shadow-xl"
  style="left: {left}px; top: {top}px;"
>
  <div class="mb-1 flex items-center justify-between gap-2">
    <span class="font-mono text-cyan-300">({x}, {y}, {z})</span>
    <span class="font-semibold">{decoded.type.name}<span class="text-slate-500"> meta={decoded.meta}</span></span>
  </div>
  {#if events === null}
    <p class="text-slate-500">No simulation_data log loaded for this machine.</p>
  {:else if events.length === 0}
    <p class="text-slate-500">No events for this block (never activated/pushed).</p>
  {:else}
    <p class="mb-1 text-slate-500">{events.length} event(s), in order:</p>
    <ol class="max-h-64 space-y-0.5 overflow-y-auto font-mono text-[11px] leading-tight">
      {#each events as ev, i (i)}
        <li class="border-t border-slate-800 pt-0.5 first:border-0 first:pt-0">{formatEvent(ev)}</li>
      {/each}
    </ol>
  {/if}
</div>
