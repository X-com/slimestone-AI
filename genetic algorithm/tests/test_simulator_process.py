from __future__ import annotations

from genetic_ml.simulator_process import _prepend_matching_runtime_dir


def test_prepend_matching_runtime_dir_puts_dir_first_when_it_exists(tmp_path):
    env = {"PATH": r"C:\some\other\bin"}

    _prepend_matching_runtime_dir(env, runtime_dir=tmp_path)

    assert env["PATH"] == f"{tmp_path};C:\\some\\other\\bin"


def test_prepend_matching_runtime_dir_is_a_noop_when_dir_is_missing(tmp_path):
    missing = tmp_path / "does-not-exist"
    env = {"PATH": r"C:\some\other\bin"}

    _prepend_matching_runtime_dir(env, runtime_dir=missing)

    assert env["PATH"] == r"C:\some\other\bin"


def test_prepend_matching_runtime_dir_handles_missing_path_key(tmp_path):
    env: dict[str, str] = {}

    _prepend_matching_runtime_dir(env, runtime_dir=tmp_path)

    assert env["PATH"] == f"{tmp_path};"
