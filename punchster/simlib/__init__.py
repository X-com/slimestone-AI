"""Self-contained copy of the genetic_ml Python<->C++ simulator bridge.

These modules were copied out of `genetic algorithm/genetic_ml/` so that punchster/ can run the
simulator without depending on that package being importable. They keep the same module names and
public API; only the internal imports were rewritten to be package-relative (`.config` instead of
`genetic_ml.config`).
"""