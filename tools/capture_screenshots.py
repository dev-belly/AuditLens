"""Capture screenshots of the AuditLens dashboard for the README.

Why this exists
---------------
The README should show what the dashboard actually looks like, and a screenshot is
the only honest way to do that. Streamlit renders client-side over a websocket, so a
plain ``chrome --screenshot`` captures the loading skeleton and nothing else - the
skeleton is what the browser has before the first websocket frame arrives.

This tool drives Chrome over the DevTools Protocol instead, so it can *wait* for the
app to finish rendering before capturing. It needs no browser-automation package:
``websockets`` and ``httpx`` are already in the project's environment as Streamlit
dependencies.

Navigation is done by clicking the sidebar, not by deep-linking to ``/explorer`` and
friends. A direct request to a sub-path reaches the server before the app has run
``st.navigation``, so Streamlit answers with a "Page not found" dialog over an
otherwise correct page. Clicking the real link is both the authentic user flow and
the only way to get a clean capture.

Usage
-----
Start the dashboard first, then run this against it::

    streamlit run dashboard/app.py --server.port 8501 &
    python tools/capture_screenshots.py --port 8501

Both the Chrome path and the output directory can be overridden; see ``--help``.
The browser is launched with a temporary profile and is always closed on exit,
including on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "screenshots"

#: ``(sidebar label, page heading, output filename)``.
#:
#: The two labels are not always the same - the sidebar says "Benford Analysis" and
#: the page heading says "Benford's Law Analysis" - so both are carried explicitly.
#: The sidebar label is what gets clicked; the heading is what proves the new page has
#: actually painted, rather than the previous page still being on screen.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("Executive Overview", "Executive Overview", "01_executive_overview"),
    ("Transaction Explorer", "Transaction Explorer", "02_transaction_explorer"),
    ("Audit Rules", "Audit Rules", "03_audit_rules"),
    ("Benford Analysis", "Benford's Law Analysis", "04_benford_analysis"),
    ("Machine Learning", "Machine Learning", "05_machine_learning"),
    ("Vendor Risk", "Vendor Risk", "06_vendor_risk"),
)

#: Candidate browser locations, most specific first. Playwright's bundled Chromium
#: is preferred because it is a known-good headless build.
BROWSER_CANDIDATES: tuple[str, ...] = (
    "~/Library/Caches/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-mac-arm64/chrome-headless-shell",
    "~/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)

#: Streamlit paints a skeleton first, then swaps in real content once the websocket
#: session is established. ``stStatusWidget`` is the "Running..." indicator. Requiring
#: a decent amount of text and no skeleton is what separates a real capture from the
#: loading state that a naive screenshot returns.
#:
#: ``__TITLE__`` is replaced with a *JSON-encoded* string, not a raw one: the headings
#: contain apostrophes ("Benford's Law Analysis"), and interpolating one inside single
#: quotes would produce a syntax error rather than a failed match.
READY_JS = """
(() => {
  const app = document.querySelector('[data-testid="stAppViewContainer"]');
  if (!app) return 'no-app';
  if (document.querySelector('[data-testid="stSkeleton"]')) return 'skeleton';
  if (document.querySelector('[data-testid="stStatusWidget"]')) return 'running';
  const heads = [...document.querySelectorAll('h1,h2,h3')]
    .map(h => (h.innerText || '').trim());
  if (!heads.some(h => h.includes(__TITLE__))) return 'wrong-page';
  const text = (app.innerText || '').trim();
  if (text.length < 400) return 'thin:' + text.length;
  return 'ready:' + text.length;
})()
"""

#: Click the sidebar entry for a page, the way a reviewer would.
CLICK_JS = """
(() => {
  const links = [...document.querySelectorAll('[data-testid="stSidebarNavLink"]')];
  const target = links.find(a => (a.innerText || '').trim().includes(__TITLE__));
  if (!target) return 'not-found';
  target.click();
  return 'clicked';
})()
"""


def find_browser(explicit: str | None) -> str:
    """Return the first usable browser executable, or raise."""
    candidates = [explicit] if explicit else []
    candidates += [str(Path(p).expanduser()) for p in BROWSER_CANDIDATES]
    for candidate in candidates:
        if candidate and Path(candidate).exists() and os.access(candidate, os.X_OK):
            return candidate
    raise SystemExit(
        "No Chromium build found. Pass --browser /path/to/chrome.\n"
        "Checked:\n  " + "\n  ".join(str(c) for c in candidates if c)
    )


class DevTools:
    """A minimal Chrome DevTools Protocol client.

    Only the handful of methods this tool needs are wrapped. Events are consumed and
    discarded while waiting for the reply to the request we sent, which is sufficient
    here: nothing in this flow depends on an event stream.
    """

    def __init__(self, ws: websockets.WebSocketClientProtocol) -> None:
        self._ws = ws
        self._next_id = 0

    async def send(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        await self._ws.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(await self._ws.recv())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})

    async def evaluate(self, expression: str) -> str:
        result = await self.send(
            "Runtime.evaluate", {"expression": expression, "returnByValue": True}
        )
        return str(result.get("result", {}).get("value", ""))

    async def wait_for_render(self, timeout: float, heading: str) -> str:
        """Poll until Streamlit has painted the page headed *heading*, or give up."""
        script = READY_JS.replace("__TITLE__", json.dumps(heading))
        deadline = time.monotonic() + timeout
        last = "unknown"
        while time.monotonic() < deadline:
            try:
                last = await self.evaluate(script)
            except Exception:  # noqa: BLE001 - a navigation in flight is not fatal
                last = "error"
            if last.startswith("ready:"):
                return last
            await asyncio.sleep(0.5)
        return f"timeout({last})"


async def _page_target(port: int, attempts: int = 40) -> str:
    """Wait for the browser's debug endpoint and return the page target's WS URL."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        for _ in range(attempts):
            try:
                response = await client.get(f"http://127.0.0.1:{port}/json/list")
                targets = [t for t in response.json() if t.get("type") == "page"]
                if targets:
                    return targets[0]["webSocketDebuggerUrl"]
            except Exception:  # noqa: BLE001 - the port is simply not open yet
                pass
            await asyncio.sleep(0.5)
    raise SystemExit("Chrome's debugging port never became available.")


async def capture(
    *,
    browser: str,
    base_url: str,
    output_dir: Path,
    width: int,
    height: int,
    scale: float,
    settle: float,
    timeout: float,
    debug_port: int,
) -> int:
    """Capture every page. Returns a process exit code."""
    profile = tempfile.mkdtemp(prefix="auditlens-shot-")
    proc = subprocess.Popen(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--no-first-run",
            "--disable-extensions",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={debug_port}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    failures = 0
    try:
        ws_url = await _page_target(debug_port)
        async with websockets.connect(ws_url, max_size=None) as ws:
            devtools = DevTools(ws)
            await devtools.send("Page.enable")
            await devtools.send("Runtime.enable")
            await devtools.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": width,
                    "height": height,
                    "deviceScaleFactor": scale,
                    "mobile": False,
                },
            )

            # Land on the app root first. A direct request to a sub-path reaches the
            # server before ``st.navigation`` has run, so Streamlit answers with a
            # "Page not found" dialog drawn over an otherwise correct page. Loading
            # the root and then clicking the sidebar avoids that entirely.
            await devtools.send("Page.navigate", {"url": f"{base_url.rstrip('/')}/"})

            for index, (label, heading, name) in enumerate(PAGES):
                print(f"  {name:26s} {label}", flush=True)

                if index:
                    clicked = await devtools.evaluate(CLICK_JS.replace("__TITLE__", json.dumps(label)))
                    if clicked != "clicked":
                        print(f"    ! sidebar link not found ({clicked})", flush=True)
                        failures += 1
                        continue

                status = await devtools.wait_for_render(timeout, heading)
                if not status.startswith("ready:"):
                    print(f"    ! did not reach a ready state: {status}", flush=True)
                    failures += 1

                # Let Plotly finish its transition; the DOM is ready before the
                # canvas is, and a capture taken here would show empty chart frames.
                await asyncio.sleep(settle)

                shot = await devtools.send(
                    "Page.captureScreenshot",
                    {"format": "png", "captureBeyondViewport": False},
                )
                data = base64.b64decode(shot["data"])

                destination = output_dir / f"{name}.png"
                destination.write_bytes(data)
                print(f"    wrote {destination.name} ({len(data) / 1024:.0f} KB)", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8501, help="Streamlit's port (default 8501)")
    parser.add_argument("--host", default="127.0.0.1", help="Streamlit's host")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--browser", default=None, help="Explicit browser executable path")
    parser.add_argument("--width", type=int, default=1440, help="Viewport width in CSS px")
    parser.add_argument("--height", type=int, default=1000, help="Viewport height in CSS px")
    parser.add_argument("--scale", type=float, default=2.0, help="Device scale factor")
    parser.add_argument("--settle", type=float, default=2.5, help="Seconds to wait after render")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-page render timeout")
    parser.add_argument("--debug-port", type=int, default=9333, help="Chrome debugging port")
    args = parser.parse_args(argv)

    browser = find_browser(args.browser)
    if not args.output.exists():
        args.output.mkdir(parents=True)

    print(f"Browser : {browser}")
    print(f"Output  : {args.output}")
    print(f"Viewport: {args.width}x{args.height} @ {args.scale}x")
    print()

    return asyncio.run(
        capture(
            browser=browser,
            base_url=f"http://{args.host}:{args.port}",
            output_dir=args.output,
            width=args.width,
            height=args.height,
            scale=args.scale,
            settle=args.settle,
            timeout=args.timeout,
            debug_port=args.debug_port,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
