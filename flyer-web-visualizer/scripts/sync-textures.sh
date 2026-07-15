#!/usr/bin/env bash
# Collect the block textures the visualizer needs into public/textures/.
# Preference: DABB resource pack (has directional indicators) -> vanilla fallback.
# Run from the app root:  bash scripts/sync-textures.sh
set -euo pipefail

DABB="../DABB/assets/minecraft/textures/block"
VAN="../textures/blocks"
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

echo "textures -> $DEST ($(ls "$DEST" | wc -l) files)"