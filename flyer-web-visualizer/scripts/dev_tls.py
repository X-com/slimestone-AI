"""Auto-provisions a TLS certificate for stream_flyers.py's wss:// demo server. Self-contained
(no dependency on the backend's genetic_ml package) so this script/directory can be copied
between repos on its own and still work - see docs/training-integration.md.
"""
from __future__ import annotations

import shutil
import ssl
import subprocess
from pathlib import Path

_DEFAULT_CERT_DIR = Path(__file__).resolve().parent.parent / ".certs"
_SUBPROCESS_TIMEOUT_SECONDS = 30


def ensure_dev_cert(cert_dir: str | Path = _DEFAULT_CERT_DIR) -> tuple[Path, Path]:
    """Returns (certfile, keyfile), generating them on first use and reusing them after that.

    Tries mkcert first, if it's on PATH: a certificate every local browser trusts with zero
    warnings (github.com/FiloSottile/mkcert). Falls back to a self-signed certificate via
    `openssl req` - present on virtually every Linux/macOS box, and on Windows if Git for
    Windows is installed - which needs no extra installs, but a browser will refuse it until
    you visit it directly once (see tls_hint()). Raises RuntimeError if neither tool is found."""
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    certfile, keyfile = cert_dir / "localhost.pem", cert_dir / "localhost-key.pem"
    if certfile.exists() and keyfile.exists():
        return certfile, keyfile

    if shutil.which("mkcert") is not None and _try_mkcert(certfile, keyfile):
        return certfile, keyfile
    if shutil.which("openssl") is not None and _try_openssl(certfile, keyfile):
        return certfile, keyfile
    raise RuntimeError(
        "No TLS tooling found - install mkcert (https://github.com/FiloSottile/mkcert, then run "
        "`mkcert -install`) or openssl, then re-run. Both are free and take under a minute."
    )


def _try_mkcert(certfile: Path, keyfile: Path) -> bool:
    try:
        subprocess.run(
            ["mkcert", "-install"], check=True, capture_output=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS
        )
        subprocess.run(
            ["mkcert", "-cert-file", str(certfile), "-key-file", str(keyfile), "localhost", "127.0.0.1", "::1"],
            check=True, capture_output=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    if not (certfile.exists() and keyfile.exists()):
        return False
    print(f"[dev-tls] generated a locally-trusted cert via mkcert: {certfile}")
    return True


def _try_openssl(certfile: Path, keyfile: Path) -> bool:
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "825", "-nodes",
                "-keyout", str(keyfile), "-out", str(certfile),
                "-subj", "/CN=localhost",
                "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1",
            ],
            check=True, capture_output=True, timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    if not (certfile.exists() and keyfile.exists()):
        return False
    print(f"[dev-tls] generated a self-signed cert via openssl: {certfile}")
    return True


def build_ssl_context(cert_dir: str | Path = _DEFAULT_CERT_DIR) -> ssl.SSLContext:
    """The ssl.SSLContext to pass as serve(..., ssl=...) for a wss:// server."""
    certfile, keyfile = ensure_dev_cert(cert_dir)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile, keyfile)
    return context


def tls_hint(host: str, port: int) -> str:
    """One-line reminder to print alongside "serving wss://host:port" - unlike a normal https://
    page load, a browser never shows its "accept this certificate" prompt for a wss:// WebSocket
    connection, so an untrusted (self-signed openssl-fallback) cert just makes every wss://
    connect attempt fail silently until the browser is told to trust it by visiting the https://
    origin directly at least once."""
    return (
        f"[dev-tls] if the browser can't connect: open https://{host}:{port}/ once and accept "
        "any certificate warning there, then reload the visualizer page and Connect."
    )
