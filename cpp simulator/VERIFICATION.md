# Verification Plan

The Java extract remains the behavior oracle until the C++ simulator matches it.

## Reference Java Commands

Generate per-schematic stream JSON files:

```powershell
cd '..\extract'
& 'D:\ProgramFiles\java8\bin\java.exe' -cp 'out\codex-classes' kernel.main.Mcp1122SchematicToStreamJsonMain
```

Run Java stream fixture results:

```powershell
& 'D:\ProgramFiles\java8\bin\java.exe' -cp 'out\codex-classes' kernel.main.Mcp1122FlyingMachineStreamFixtureMain
```

Run Java with trace enabled for one candidate or fixture set:

```powershell
& 'D:\ProgramFiles\java8\bin\java.exe' -cp 'out\codex-classes' -Dmcp1122main.trace=true kernel.main.Mcp1122FlyingMachineStreamFixtureMain
```

Java writes trace data through `McpUpdateTrace` using this shape:

```text
worldTime tag x,y,z optionalA,optionalB
```

The C++ `Trace` class uses the same line shape.

## C++ Milestone Checks

## Current Status

The C++ simulator now has an integrated tick loop, scheduled observer ticks,
block-event processing, piston structure helper logic, moving piston settlement,
piston-head delegation, basic redstone power, and translation-invariant cycle
detection.

Current fixture result against `../extract/outlog/stream-input/*.json`:

```text
lines=17
ok=17
working=9
errors=0
```

Known matching working fixtures:

```text
complex_machine4
movingInstantwire
simple_caterpillar
simple_machine1
simple_machine2
simple_machine3
simple_no_sticky_loop
simple_observer_engine
simple_upwards_engine
```

Known remaining misses:

```text
1 wide trencher
1-wide with rails
1797 onepointfive
24 onepointfive
complex_machine
complex_machine3
movableRCA
submarine
```

The remaining misses should be debugged with Java/C++ trace diffs. The most likely
remaining gaps are full block power semantics, rail/fence/support block break behavior,
and additional piston edge cases involving extended pistons and moving piston blocks.

### Milestone 1: Stream Boundary

Expected now:

- C++ parses every `../extract/outlog/stream-input/*.json`
- C++ loads all non-air block states into packed world storage
- C++ outputs JSON result lines
- C++ explicitly reports `native_tick_loop_not_implemented`

This verifies that the Python/Java/native stream boundary is stable before porting behavior.

### Milestone 2: Initial State Key

Add a C++ initial-state key and compare it with a Java initial-state dump.

Compare:

- block count
- normalized packed block entries
- trigger position
- state ids

### Milestone 3: Tick Queues And Block Events

Add native scheduled tick and block-event queues.

Compare per tick against Java trace:

- `h.tick`
- `w.beq`
- block event fire order
- scheduled tick drain order

### Milestone 4: Observer Behavior

Port observer state transitions and scheduled updates.

Compare:

- observer schedule tick
- powered/unpowered state id changes
- neighbor update order

### Milestone 5: Piston Structure Helper

Port piston move/destroy list calculation using fixed arrays.

Compare:

- move count
- destroy count
- move positions in order
- destroy positions in order

Current C++ diagnostic:

```powershell
Get-ChildItem '..\extract\outlog\stream-input' -Filter '*.json' |
  Sort-Object Name |
  Get-Content |
  .\build\mcp1122_cpp_stream.exe --debug-piston-helper
```

The diagnostic reports `trigger_not_piston` for candidates whose stream trigger points
at an observer or another non-piston block.

### Milestone 6: Piston Movement

Port piston block event handling and movement.

Compare:

- piston event receive order
- block removal/set order
- moving block creation
- neighbor update order
- moving block settlement

Current C++ diagnostic:

```powershell
Get-ChildItem '..\extract\outlog\stream-input' -Filter '*.json' |
  Sort-Object Name |
  Get-Content |
  .\build\mcp1122_cpp_stream.exe --debug-piston-move
```

This currently validates immediate move/destroy ordering only. It is not yet a
replacement for Java `doMove`, because Java creates moving piston tile entities and
settles them through the moving-block tick path.

### Milestone 7: Cycle Detection

Port exact cycle detection first, then light detection.

Compare final stream results:

- `ok`
- `working`
- `ticks`
- `start`
- `end`
- `period`
- `shift`

Only after this should performance tuning begin.
