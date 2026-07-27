#!/usr/bin/env bash
# Collect the block textures the visualizer needs into public/textures/.
# Preference: DABB resource pack (has directional indicators) -> vanilla fallback.
# Run from the app root:  bash scripts/sync-textures.sh
set -euo pipefail

DABB="../ignore/DABB/assets/minecraft/textures/block"
VAN="../ignore/textures/blocks"
DEST="public/textures"
mkdir -p "$DEST"

# dest_name  source1  [source2 ...]  (first existing wins)
pick() {
  local dest="$1"; shift
  for src in "$@"; do
    if [ -f "$src" ]; then cp "$src" "$DEST/$dest"; return; fi
  done
  echo "WARN: no source found for $dest" >&2
}

pick stone.png              "$VAN/stone.png"
pick glass.png              "$VAN/glass.png"
pick slime.png              "$VAN/slime.png"
pick redstone_block.png     "$VAN/redstone_block.png"

pick piston_top.png         "$DABB/piston_top.png"          "$VAN/piston_top_normal.png"
pick piston_top_sticky.png  "$DABB/piston_top_sticky.png"   "$VAN/piston_top_sticky.png"
pick piston_side.png        "$DABB/piston_side.png"         "$VAN/piston_side.png"
pick piston_side_sticky.png "$DABB/piston_side_sticky.png"  "$VAN/piston_side.png"
pick piston_bottom.png      "$DABB/piston_bottom.png"       "$VAN/piston_bottom.png"
pick piston_bottom_sticky.png "$DABB/piston_bottom_sticky.png" "$VAN/piston_bottom.png"

pick observer_front.png     "$DABB/observer_front.png"      "$VAN/observer_front.png"
pick observer_back.png      "$DABB/observer_back.png"       "$VAN/observer_back.png"
pick observer_side.png      "$DABB/observer_side.png"       "$VAN/observer_side.png"
pick observer_top.png       "$DABB/observer_top.png"        "$VAN/observer_top.png"

# Observer "on" (powered) reskin - DABB only, all 4 distinct faces (front/side/top/back), not
# just the back arrow - see ignore/DABB/assets/minecraft/models/block/observer_powered.json.
# Deliberately no vanilla fallback for any of these (vanilla only has a back-face "lit" texture,
# a different look the user does not want used as a substitute).
pick observer_front_on.png  "$DABB/observer_front2.png"
pick observer_side_on.png   "$DABB/observer_side2.png"
pick observer_top_on.png    "$DABB/observer_top2.png"
pick observer_back_on.png   "$DABB/observer_back_on.png"

echo "curated textures -> $DEST ($(ls "$DEST" | wc -l) files)"

# Every other Minecraft block texture, so nothing renders as "missing" once a block type is
# wired up: DABB wins per-file, vanilla (the full ignore/textures/blocks set) fills in whatever
# DABB doesn't have. Skips anything the curated section above already placed.
all_names=$( { ls "$DABB" 2>/dev/null; ls "$VAN" 2>/dev/null; } | sort -u )
for name in $all_names; do
  case "$name" in *.png) ;; *) continue ;; esac
  [ -f "$DEST/$name" ] && continue
  if [ -f "$DABB/$name" ]; then cp "$DABB/$name" "$DEST/$name"; else cp "$VAN/$name" "$DEST/$name"; fi
done

echo "textures -> $DEST ($(ls "$DEST" | wc -l) files)"
