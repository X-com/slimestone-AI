<script lang="ts">
  import type { Machine } from './data'

  let { machine, onClose }: { machine: Machine | null; onClose: () => void } = $props()
</script>

{#if machine}
  {@const c = machine.candidate}
  {@const r = machine.result}
  <aside
    class="absolute right-0 top-0 h-full w-80 max-w-[85vw] overflow-y-auto bg-slate-900/95 p-4 text-sm text-slate-200 shadow-xl"
  >
    <div class="mb-3 flex items-start justify-between gap-2">
      <h2 class="font-semibold">Machine</h2>
      <button
        class="rounded px-2 py-0.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
        onclick={onClose}>✕</button
      >
    </div>
    <dl class="space-y-2">
      {#if machine.source}
        <div>
          <dt class="text-xs uppercase text-slate-500">source</dt>
          <dd class="break-all font-mono text-xs text-cyan-300">{machine.source}</dd>
        </div>
      {:else}
        <div>
          <dt class="text-xs uppercase text-slate-500">id</dt>
          <dd class="break-all font-mono text-xs text-cyan-300">{machine.hash}</dd>
        </div>
      {/if}
      {#each [['index', c.id], ['blocks', machine.block_count], ['trigger', `x${c.trigger.x} y${c.trigger.y} z${c.trigger.z}`]] as [k, v] (k)}
        <div class="flex justify-between gap-3">
          <dt class="text-xs uppercase text-slate-500">{k}</dt>
          <dd class="text-right font-mono text-xs">{v}</dd>
        </div>
      {/each}
      {#if r}
        {#each [['generation', machine.generation], ['origin', machine.origin], ['ticks', r.ticks], ['period', `${r.period} ticks`], ['shift (flight)', `x${r.shift.x} y${r.shift.y} z${r.shift.z}`], ['found', machine.found_at.slice(0, 19).replace('T', ' ')]] as [k, v] (k)}
          <div class="flex justify-between gap-3">
            <dt class="text-xs uppercase text-slate-500">{k}</dt>
            <dd class="text-right font-mono text-xs">{v}</dd>
          </div>
        {/each}
      {:else}
        <p class="pt-1 text-xs text-slate-500">No simulation metadata available.</p>
      {/if}
    </dl>
    <button
      class="mt-4 w-full cursor-not-allowed rounded bg-slate-800 px-3 py-1.5 text-xs font-medium text-slate-500"
      disabled
      title="Coming soon: run this machine's simulation"
    >
      Simulate
    </button>
  </aside>
{/if}
