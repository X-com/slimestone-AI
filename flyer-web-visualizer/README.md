# Flyer Machine Visualizer

3D viewer for the genetic-algorithm flying machines

## Run

```bash
pnpm install
pnpm dev        # open the printed localhost URL
```

- Two top tabs: **Complex Examples** and **Examples**. The page opens on the first complex example, selected.
- **Complex Examples** — large hand-authored machines from `public/machines/*.json` (listed in `manifest.json`, regenerate with `pnpm sync-machines`). Shown one at a time; click the tab again to cycle to the next (`n/N ↻`).
- **Examples** (archive) — **First 100 / Last 100 / Random 100** pick which 100 archive machines (of 800) to show. Random re-rolls each click.
- **Upload .data** — load machines from `.data` files (the compact binary candidate format from `../genetic-ml`). Pick one or many files; records are parsed client-side and shown as "Uploaded (N)". Rendering is capped at 400 at a time (one DOM label per machine); the header says when it's capped. `✕` clears uploads.
- The **trigger block** (the GA-specified simulation start point) is drawn at its position as a translucent purple glowing cube (0.1 alpha, additive).
- Each machine floats a label — archive: first 8 hash chars (full hash on hover); complex/uploaded: name/`#id` (source on hover).
- **Click** a machine or its label → detail panel, and the camera recenters/pivots on it. `id` is the structural hash; `index` is the GA machine index (`candidate.id`). Archive also shows generation, origin, blocks, ticks, period, flight `shift`, trigger; uploaded shows source, index, blocks, trigger (bare `.data` candidates carry no hash or simulation metadata).
- Blocks are **textured** (see below). Pistons / sticky pistons / observers are rotated so their directional face points the right way. Floating labels are depth-tested — they hide behind blocks in front of them.

## Camera

- **Orbit / zoom / pan** all pivot around the currently selected machine. Left-drag orbit · right-drag (or two-finger) pan · wheel zoom.
- **Controls** (bottom-left, or keyboard): `←`/`→` step through the visible machines, recentering + framing each (also changes the selection); `↑` top-down, `↓` isometric. Elevation snaps keep the current azimuth and zoom distance.

## Textures

Block faces use a texture atlas built at load from `public/textures/*.png`. These are
collected by `scripts/sync-textures.sh` (run via `pnpm sync-textures`), which prefers the
**DABB** pack (`../DABB`, has directional arrows on pistons/observers) and falls back to
**vanilla** (`../textures`) per texture — e.g. pistons/observers come from DABB, stone /
glass / slime / redstone from vanilla. The resolved PNGs are committed under
`public/textures/`, so the app runs without re-syncing. Glass is alpha-tested (opaque
frame, clear pane); slime is translucent; if the atlas fails to load, blocks fall back to
flat colors. To retexture, drop a new pack in and re-run `pnpm sync-textures`.

```bash
pnpm sync-data
```
