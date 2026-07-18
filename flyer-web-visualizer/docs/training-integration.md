# Streaming live machines to the `/live` dashboard

The visualizer's **Live Training** page connects to a WebSocket and renders machines as they
arrive. To feed it from the real training loop you host a small WS server that:

1. **On each new connection** sends the current historical backlog (one binary frame).
2. **Then streams** new machines as the loop discovers them (one frame per batch).

The frontend resets its view on every connection, so the server **must** resend the full backlog
each time a client connects (this is what makes the frontend's *auto-reconnect* checkbox work).

## Wire format (already implemented on both ends)

A frame is just concatenated compact records — the same `.data` bytes on disk. Reuse the existing
encoder; no new format:

```python
from genetic_ml.compact_format import encode_candidate   # candidate -> bytes (20B header + 16B/block)

frame = b"".join(encode_candidate(c) for c in candidates)  # one WS binary frame = one "batch"
```

The browser decodes each frame with `parseCompactData` (`src/lib/data.ts`) as one batch. There is
**no outer length prefix** — records are self-delimiting, so a frame may hold 1 or 10,000 machines.

`scripts/stream_flyers.py` is an offline stand-in that does exactly this from a saved file; the steps
below wire the same behavior into the live loop.

## 1. Declare the server once, with the backlog (top of training)

Run the WS server in a background thread so it never blocks the (synchronous) GA/RL loop. Drop this
helper in as e.g. `genetic_ml/stream_hub.py`:

```python
# genetic_ml/stream_hub.py
import asyncio, threading
from websockets.asyncio.server import serve

class StreamHub:
    """Background-thread WebSocket hub: backfills the backlog on connect, fans out live frames."""
    def __init__(self, backlog: bytes = b"", host="localhost", port=8765):
        self._backlog = bytearray(backlog)   # every record sent so far, in wire format
        self._clients: set[asyncio.Queue] = set()
        self._host, self._port = host, port
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._serve())
        self._loop.run_forever()

    async def _serve(self):
        await serve(self._client, self._host, self._port, max_size=None)
        print(f"[stream] ws://{self._host}:{self._port}")

    async def _client(self, ws):
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._clients.add(q)
            backlog = bytes(self._backlog)
        try:
            if backlog:
                await ws.send(backlog)            # (1) historical backfill, one frame
            while True:
                await ws.send(await q.get())      # (2) live batches
        except Exception:
            pass
        finally:
            with self._lock:
                self._clients.discard(q)

    def publish(self, frame: bytes):
        """Thread-safe. Append to the backlog and fan a batch out to all connected clients."""
        if not frame:
            return
        with self._lock:
            self._backlog += frame
            clients = list(self._clients)
        for q in clients:
            self._loop.call_soon_threadsafe(q.put_nowait, frame)
```

Wire it in near the top of `main_ga.py` (the on-disk `flyers.data` is already in wire format, so the
backlog is just its bytes):

```python
from genetic_ml.stream_hub import StreamHub

COMPACT_FILE = COMPACT_DIR / "flyers.data"
backlog = COMPACT_FILE.read_bytes() if COMPACT_FILE.exists() else b""
hub = StreamHub(backlog=backlog)   # backfill included with the declaration
hub.start()
```

## 2. Stream new machines from the loop

Wherever the loop records a working discovery (in `ga_loop.run_ga`, alongside
`resolved_working_writer.save(child)`), publish the same candidates as a batch:

```python
# after a generation's working machines are found:
new_working = [child for child, res in offspring if res.working and _is_new(child)]
if new_working:
    hub.publish(b"".join(encode_candidate(c) for c in new_working))
```

Batch granularity is up to you — publish per generation, or accumulate and publish every N
discoveries (mirroring `CompactWorkingWriter`'s `flush_every`). Each `publish` becomes one batch the
dashboard shows at the top; the bottom pager keeps the full history.

## Notes

- **Backlog grows in memory.** `StreamHub` keeps every published frame so late joiners get full
  history. For very long runs, cap it (e.g. keep the last K records) or backfill from disk on connect
  instead of an in-memory buffer.
- **One hub, many clients.** Multiple browser tabs can watch the same run; each gets the backlog then
  the same live batches.
- **Thread safety.** The loop only ever calls `hub.publish(...)` — it's the single synchronous entry
  point; everything async happens on the hub's own thread.
- **RL vs GA.** The loop here is a GA, but the contract is identical for an RL trainer: encode each
  emitted machine's candidate and `publish` it.

## Serving over HTTPS (`wss://`)

The frontend connects to `wss://localhost:8765` by default, so `StreamHub` needs to terminate TLS.
This is fully automatic — `genetic_ml/dev_tls.py`'s `build_ssl_context()` generates a local
certificate on first use and reuses it after that; nobody has to hand-run mkcert/openssl commands:

```python
from genetic_ml.dev_tls import build_ssl_context

ssl_context = build_ssl_context()  # generates data/dev-certs/ on first call, reuses it after
hub = StreamHub(backlog=backlog, ssl_context=ssl_context)
```

`StreamHub` just forwards `ssl_context` to `serve(..., ssl=ssl_context)` and prints `wss://` instead
of `ws://` once it's serving. `main_rl.py`/`main_ga.py` both do this behind a `STREAM_TLS = True`
constant (flip it to `False`, and change the frontend's URL box back to `ws://`, to skip TLS).
`scripts/stream_flyers.py` does the same by default (`--no-tls` to opt out), but through its own
**self-contained** `scripts/dev_tls.py` (same mkcert→openssl logic, cert in `.certs/` next to the
script) rather than importing `genetic_ml` — `scripts/` gets copied between repos on its own (see
"Keeping the two repos in sync" below) where a `genetic algorithm` sibling may not exist or may be
a stale mirror, so this script can't assume it's reachable.

**Why `wss://`, not `ws://`, is required at all — mixed content.** This is *not* CORS (WebSockets
don't do a CORS preflight); it's simpler than that: a page served over `https://` is not allowed to
open an insecure `ws://` socket — the browser blocks it outright, no warning to click through, just
a hard block. Once the frontend itself is served over HTTPS (any real deploy, including Vercel), the
stream endpoint must be `wss://` too. Serving over plain `http://` locally never hits this, which is
exactly why a local `ws://` setup can *look* like it works before this matters.

**How the cert gets made** (`ensure_dev_cert`, tried in this order, first success wins):

1. **[mkcert](https://github.com/FiloSottile/mkcert)**, if it's on `PATH` — runs `mkcert -install`
   (idempotent, safe to call every time) then generates a `localhost`/`127.0.0.1`/`::1` cert signed
   by mkcert's local CA. Every browser on the machine trusts it immediately, zero warnings.
2. **`openssl`**, if mkcert isn't available — one `openssl req -x509 -newkey rsa:2048 ...` call
   generates a self-signed cert with the same SAN entries. Works with no extra installs (present on
   effectively every Linux/macOS box, and on Windows if Git for Windows is installed), but needs the
   one-time manual trust step below.
3. **Neither found** → raises `RuntimeError` with the install instructions above, instead of
   silently falling back to plaintext `ws://`.

The generated cert/key live in `<repo root>/data/dev-certs/` (gitignored — the key must never be
committed) and are reused across runs and across GA/RL/`stream_flyers.py`, so cert generation only
ever happens once per machine.

### The one manual step a self-signed cert still needs

A browser shows its "accept this certificate" interstitial only for a top-level page **navigation**
(typing/opening a URL) — never for a `wss://` `WebSocket` connection made from a script. So with a
self-signed cert (the openssl fallback path), `new WebSocket('wss://localhost:8765')` just fails
with a bare, detail-free error in devtools (`WebSocket connection to 'wss://...' failed:`) and
nothing reaches the server, even though the server is up and listening — there is no exception to
click through from the WebSocket call itself.

Fix it once per browser/profile: open `https://localhost:8765/` directly in a new tab, click through
the "connection is not private" warning, then reload the visualizer page and hit Connect — the
`wss://` connection now succeeds, since the browser remembers the trust exception for that origin.
Both `StreamHub` and `stream_flyers.py` print this reminder (`dev_tls.tls_hint()`) right after they
start serving over TLS, and the dashboard's own "Connection failed" message links to the same URL
when it detects a `wss://` target. Installing mkcert avoids this step entirely (its cert is already
trusted, so there's no warning to click through in the first place).

Once trust is granted, visiting `https://localhost:8765/` directly will show a plain-text error like
`Failed to open a WebSocket connection: invalid Connection header ... You cannot access a WebSocket
server directly with a browser.` — **that response is expected and is actually the success signal**:
it means the TLS handshake completed and the browser now trusts the cert (a browser sending a normal
page-load GET, with no `Upgrade: websocket` header, always gets this from a WebSocket-only server —
it isn't a bug to fix). If instead the tab shows a certificate error, trust wasn't granted yet.

### The hosted-site-specific blocker: Chrome/Edge 147+ Local Network Access

A page connecting fine from `http://localhost:5173` (local dev) but failing from the **deployed**
site is not necessarily just a stale build. Chromium browsers (Chrome, Edge, Brave, Opera) shipped a
**Local Network Access** permission gate in Chrome 142 (Oct 2025) for `fetch`/XHR, and **extended it
to WebSocket/WebTransport in Chrome 147** (Apr 2026): a page loaded from a **public** origin (e.g.
`https://your-app.vercel.app`) opening a socket to a **private** address (`localhost`, `127.0.0.1`,
`::1`, or any RFC1918 address) now requires the user to grant a one-time permission — the browser
shows *"`<site>` wants to connect to devices on your local network"* the first time it happens.

- A page served from `localhost` connecting to `wss://localhost:8765` is **private → private** and
  is never gated — this is exactly why local dev "just works" and gives no signal about the hosted
  path.
- The prompt requires the *requesting* page to itself be a secure context (served over `https://`,
  which every real deploy is) — otherwise Chrome fails the connection **silently, with no prompt and
  no way to allow it**.
- The grant/deny decision is remembered per-origin. If it was ever dismissed or denied, every later
  attempt from that origin fails until it's reset via the address bar's lock/info icon → **Site
  settings → Local network access → Allow**, then reload.
- Firefox and Safari don't enforce this (yet) — useful for isolating whether this is the blocker.

This is a deliberate browser security boundary (stops arbitrary public sites from silently probing a
visitor's home network), not a bug in this app, and there is no server-side response that grants it
automatically — TLS being correct is necessary but not sufficient for a *hosted* page. The dashboard
itself surfaces a hint for this case (`targetsPrivateHost && pageIsPublicOrigin` in
`TrainingDashboard.svelte`) alongside the self-signed-cert hint.

If the actual goal is letting a hosted page watch a *remote* training run (not the viewer's own
machine), don't point it at `localhost` at all — that's what the tunnel/reverse-proxy option below is
for; a real public `wss://` hostname isn't a "local network address" and isn't gated by this at all.

### Alternative: skip the app-level cert entirely with a reverse proxy / tunnel

No code change needed: put a TLS terminator in front of the plain `ws://localhost:8765` server. e.g.
[Caddy](https://caddyserver.com/) (`reverse_proxy localhost:8765`, automatic HTTPS) run locally, or a
tunnel like `cloudflared`/`ngrok` that hands you a **public** `wss://…` URL to paste into the
dashboard's URL box. This is also the right answer for genuinely remote training (a real host+tunnel
address instead of `localhost`), and it sidesteps Local Network Access entirely, since the target is
no longer a private address from the browser's point of view.

- **CSP.** Only relevant if whatever hosts the built frontend sets a `Content-Security-Policy` —
  then `connect-src` must include the `wss://` endpoint. The app itself sets none.
- **Origin.** `websockets.serve` accepts all origins by default, so this isn't a blocker unless
  someone adds origin validation later.

## Keeping the two repos in sync

The frontend (`flyer-web-visualizer/`) is developed in two checkouts: this monorepo (where it's
wired to the real GA/RL backend for local integration work) and a standalone repo that Vercel
actually deploys from. Editing only this copy never reaches production. When changing anything under
`flyer-web-visualizer/` (`src/`, `scripts/`, `docs/`), mirror the same files into the other checkout
and commit/push there too — that's the repo with the Vercel hook. `scripts/stream_flyers.py` is
written to be copy-safe (self-contained `scripts/dev_tls.py`, no assumption about which name or
layout the backend sibling directory uses), specifically so this sync doesn't silently break it.
