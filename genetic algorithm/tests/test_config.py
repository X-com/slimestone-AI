from __future__ import annotations

from genetic_ml import config


def test_simulator_exe_name_adds_exe_suffix_only_on_windows(monkeypatch):
    monkeypatch.setattr(config.platform, "system", lambda: "Windows")
    assert config.simulator_exe_name() == "cpp_simulator_stream.exe"

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    assert config.simulator_exe_name() == "cpp_simulator_stream"

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    assert config.simulator_exe_name() == "cpp_simulator_stream"
