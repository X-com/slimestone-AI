from __future__ import annotations

import ssl
import subprocess

import pytest

from genetic_ml import dev_tls


def test_reuses_existing_cert_files_without_touching_any_tool(tmp_path, monkeypatch):
    certfile, keyfile = tmp_path / "localhost.pem", tmp_path / "localhost-key.pem"
    certfile.write_text("existing-cert")
    keyfile.write_text("existing-key")
    monkeypatch.setattr(dev_tls.shutil, "which", lambda _name: (_ for _ in ()).throw(AssertionError("should not probe for tools")))

    result_cert, result_key = dev_tls.ensure_dev_cert(tmp_path)

    assert (result_cert, result_key) == (certfile, keyfile)
    assert certfile.read_text() == "existing-cert"  # untouched, not regenerated


def test_falls_back_to_openssl_when_mkcert_is_absent_and_produces_a_loadable_cert(tmp_path, monkeypatch):
    real_which = dev_tls.shutil.which
    monkeypatch.setattr(
        dev_tls.shutil, "which", lambda name: None if name == "mkcert" else real_which(name)
    )
    if real_which("openssl") is None:
        pytest.skip("openssl not installed on this machine")

    certfile, keyfile = dev_tls.ensure_dev_cert(tmp_path)

    assert certfile.exists() and keyfile.exists()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)  # raises if the generated cert/key don't match


def test_raises_a_clear_error_when_no_tls_tooling_is_available(tmp_path, monkeypatch):
    monkeypatch.setattr(dev_tls.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="mkcert"):
        dev_tls.ensure_dev_cert(tmp_path)


def test_mkcert_is_tried_first_with_the_expected_arguments(tmp_path, monkeypatch):
    certfile, keyfile = tmp_path / "localhost.pem", tmp_path / "localhost-key.pem"
    monkeypatch.setattr(dev_tls.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "mkcert" else None)

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[1] == "-cert-file":
            certfile.write_text("mkcert-cert")
            keyfile.write_text("mkcert-key")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(dev_tls.subprocess, "run", fake_run)

    result_cert, result_key = dev_tls.ensure_dev_cert(tmp_path)

    assert (result_cert, result_key) == (certfile, keyfile)
    assert calls[0] == ["mkcert", "-install"]
    assert calls[1] == ["mkcert", "-cert-file", str(certfile), "-key-file", str(keyfile), "localhost", "127.0.0.1", "::1"]
