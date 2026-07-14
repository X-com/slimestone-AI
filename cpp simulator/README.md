# MCP1122 C++ Extract

This folder is the incremental C++ rewrite of `../extract`.

The port is intentionally built around the stream boundary first:

```text
JSON candidate line -> packed native world -> simulator -> JSON result line
```

The current milestone is a correctness scaffold:

- parse the same JSON stream format used by `Mcp1122FlyingMachineStreamMain`
- load candidates into packed native world storage
- write trace lines with the same shape as `McpUpdateTrace`
- return an explicit `native_tick_loop_not_implemented` result until piston/tick semantics are ported

Do not treat this executable as a finished simulator yet. The Java extract remains the oracle.

## Build

From this folder:

With the MSYS2 UCRT compiler:

```powershell
.\build-msys2.ps1
```

Or with CMake:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Run

```powershell
Get-Content '..\extract\outlog\stream-input\simple_machine1.json' |
  .\build\Release\mcp1122_cpp_stream.exe
```

Optional trace:

```powershell
Get-Content '..\extract\outlog\stream-input\simple_machine1.json' |
  .\build\Release\mcp1122_cpp_stream.exe --trace outlog\cpp-update-trace.log
```

Piston helper diagnostic:

```powershell
Get-Content '..\extract\outlog\stream-input\simple_machine1.json' |
  .\build\mcp1122_cpp_stream.exe --debug-piston-helper
```

Immediate piston move diagnostic:

```powershell
Get-Content '..\extract\outlog\stream-input\simple_machine1.json' |
  .\build\mcp1122_cpp_stream.exe --debug-piston-move
```

`--debug-piston-move` intentionally settles moved blocks immediately. Java first creates
moving piston tile entities, so this mode is for validating move order and destination
logic, not final tick-accurate flying-machine simulation.

## Porting Milestones

1. Keep Java stream results and `McpUpdateTrace` logs as the oracle.
2. Match JSON parsing and initial packed world state.
3. Port state-key/cycle detector.
4. Port scheduled tick and block-event queues.
5. Port observer behavior.
6. Port piston structure helper.
7. Port piston block-event handling and `doMove`.
8. Port moving block settlement.
9. Run differential tests against Java fixtures and traces.
10. Replace JSON with a binary batch protocol only after correctness is stable.
