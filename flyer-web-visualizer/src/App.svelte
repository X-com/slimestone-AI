<script lang="ts">
  import Router, { link, router } from 'svelte-spa-router'
  import MachineExplorer from './MachineExplorer.svelte'
  import TrainingDashboard from './TrainingDashboard.svelte'

  const routes = {
    '/': MachineExplorer,
    '/live': TrainingDashboard,
  }

  const navLinks = [
    ['/', 'Explorer'],
    ['/live', 'Live Training'],
  ] as const
</script>

<div class="flex h-full flex-col bg-[#0f111a]">
  <nav
    class="flex shrink-0 items-center gap-4 border-b border-slate-800 bg-slate-900 px-4 py-2 text-slate-200"
  >
    <span class="text-sm font-semibold tracking-wide">Flyer Visualizer</span>
    <div class="flex gap-1 text-xs">
      {#each navLinks as [path, label] (path)}
        <a
          href={path}
          use:link
          class="rounded px-2.5 py-1 font-medium transition
            {router.location === path
            ? 'bg-cyan-400 text-slate-900'
            : 'bg-slate-800/80 text-slate-300 hover:bg-slate-700'}"
        >
          {label}
        </a>
      {/each}
    </div>
  </nav>

  <main class="relative flex-1 overflow-hidden">
    <Router {routes} />
  </main>
</div>
