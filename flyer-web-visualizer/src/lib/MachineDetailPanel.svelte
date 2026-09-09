<script lang="ts">
  import type { Machine } from './data'

  let {
    machine,
    onClose,
    onSimulate,
    simulating = false,
    error = '',
  }: {
    machine: Machine | null
    onClose: () => void
    // Absent wherever there is no training run behind the page (the Explorer, uploaded files):
    // the button is then not shown at all, rather than shown and broken.
    onSimulate?: (m: Machine) => void
    simulating?: boolean
    error?: string
  } = $props()

  // Which share of the search budget paid for this discovery. The uninformed 5% is the control
  // group, so a machine found by it means something different from one the model chose - worth
  // colouring rather than burying in a row of identical grey text.
  const sourceTone: Record<string, string> = {
    top: 'text-emerald-300',
    sampled: 'text-cyan-300',
    uninformed: 'text-amber-300',
  }
</script>

{#snippet row(k: string, v: string | number, tone = '')}
  <div class="flex justify-between gap-3">
    <dt class="text-xs uppercase text-slate-500">{k}</dt>
    <dd class="text-right font-mono text-xs {tone}">{v}</dd>
  </div>
{/snippet}

{#snippet heading(text: string)}
  <h3 class="mt-4 border-b border-slate-800 pb-1 text-[10px] uppercase tracking-wider text-slate-500">
    {text}
  </h3>
{/snippet}

{#if machine}
  {@const c = machine.candidate}
  {@const r = machine.result}
  {@const t = machine.training}
  <aside
    class="absolute right-0 top-0 h-full w-80 max-w-[85vw] overflow-y-auto bg-slate-900/95 p-4 text-sm text-slate-200 shadow-xl"
  >
    <div class="mb-3 flex items-start justify-between gap-2">
      <h2 class="font-semibold">{t ? t.name : 'Machine'}</h2>
      <button
        class="rounded px-2 py-0.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100"
        onclick={onClose}>✕</button
      >
    </div>
    {#if onSimulate}
      <button
        class="mb-3 w-full rounded bg-cyan-400 px-3 py-1.5 text-sm font-medium text-slate-900
               hover:bg-cyan-300 disabled:cursor-wait disabled:opacity-60"
        disabled={simulating}
        onclick={() => onSimulate(machine)}
      >
        {simulating ? 'Simulating…' : '▶ Simulate'}
      </button>
      {#if error}
        <p class="mb-3 rounded bg-rose-950/60 px-2 py-1 text-xs text-rose-300">{error}</p>
      {/if}
    {/if}
    <dl class="space-y-2">
      {#if machine.source && !t}
        <div>
          <dt class="text-xs uppercase text-slate-500">source</dt>
          <dd class="break-all font-mono text-xs text-cyan-300">{machine.source}</dd>
        </div>
      {:else}
        <div>
          <dt class="text-xs uppercase text-slate-500">id</dt>
          <dd class="break-all font-mono text-xs text-cyan-300">{t ? t.digest : machine.hash}</dd>
        </div>
      {/if}
      {@render row('index', c.id)}
      {@render row('blocks', machine.block_count)}
      {@render row('trigger', `x${c.trigger.x} y${c.trigger.y} z${c.trigger.z}`)}

      <!-- Streamed from the RL loop (rlgym/stream.py). The compact .data format carries geometry
           only, so all of this rides a second JSON frame and is joined on the record's id. -->
      {#if t}
        {@render heading('Flight')}
        {@render row('period', `${t.period} ticks`)}
        {@render row('shift', `x${t.shift[0]} y${t.shift[1]} z${t.shift[2]}`)}

        {@render heading('Discovery')}
        {@render row('round found', t.round_found)}
        {@render row('generation', t.generation)}
        {@render row('parent', t.parent)}
        {@render row('found by', t.source, sourceTone[t.source] ?? '')}

        {@render heading('Redundancy')}
        <!-- A block is redundant when removing it leaves every piston firing on the same tick in
             the same order AND the machine still flying. Those are stripped before a machine is
             admitted, so this counts what was thrown away, not what remains. -->
        {@render row(
          'blocks stripped',
          t.redundant_removed,
          t.redundant_removed ? 'text-amber-300' : '',
        )}
        {@render row(
          'added block kept',
          t.added_is_load_bearing ? 'yes' : 'no - all redundant',
          t.added_is_load_bearing ? 'text-emerald-300' : 'text-slate-500',
        )}
        {#if !t.added_is_load_bearing}
          <p class="pt-1 text-[11px] leading-snug text-slate-500">
            Every block this episode added was removable without disturbing the pistons, so what
            survived trimming is the parent machine. Nothing new was found here.
          </p>
        {/if}
      {:else if r}
        {@render heading('Flight')}
        {@render row('generation', machine.generation)}
        {@render row('origin', machine.origin)}
        {@render row('ticks', r.ticks)}
        {@render row('period', `${r.period} ticks`)}
        {@render row('shift (flight)', `x${r.shift.x} y${r.shift.y} z${r.shift.z}`)}
        {@render row('found', machine.found_at.slice(0, 19).replace('T', ' '))}
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
