"""Desktop launcher — starts Flask in background, opens pywebview native window."""
from __future__ import annotations

import os
import sys
import time
import socket
import threading
import urllib.request
import subprocess
from pathlib import Path

# Force CREATE_NO_WINDOW on all subprocess calls in Windows
# This prevents random CMD flashes from internal libraries (e.g. audio-separator's ffmpeg calls)
if os.name == "nt":
    _orig_popen = subprocess.Popen
    class _PatchedPopen(_orig_popen):
        def __init__(self, *args, **kwargs):
            if "creationflags" not in kwargs:
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)
    subprocess.Popen = _PatchedPopen

# Ensure src is on path for both source and frozen modes
if getattr(sys, "frozen", False):
    _base = Path(sys._MEIPASS)
else:
    _base = Path(__file__).resolve().parent

_src = _base / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# Override audio_separator path at startup in frozen mode.
# This prevents importlib.resources from looking inside the PYZ zip archive
# where models-scores.json and other metadata JSONs do not exist.
if getattr(sys, "frozen", False):
    try:
        import audio_separator
        audio_separator.__file__ = os.path.join(sys._MEIPASS, "audio_separator", "__init__.py")
        audio_separator.__path__ = [os.path.join(sys._MEIPASS, "audio_separator")]
    except Exception as e:
        print(f"[!] Path override failed: {e}", file=sys.stderr)

os.environ["FLASK_DEBUG"] = "0"

import webview  # noqa: E402
from app import app, start_cleanup_sweeper, shutdown_event  # noqa: E402


def get_free_port() -> int:
    """Return a free TCP port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_server(url: str, timeout: float = 90.0) -> bool:
    """Wait until Flask accepts HTTP connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


class DesktopApi:
    """Native desktop actions exposed to the web interface."""

    def __init__(self) -> None:
        self._window = None

    def browse_folder(self) -> str:
        """Open the native folder picker and return the selected path."""
        if self._window is None:
            return ""
        selected = self._window.create_file_dialog(webview.FOLDER_DIALOG)
        return selected[0] if selected else ""


if __name__ == "__main__":
    # Start Flask server in a dedicated background thread with multithreading (threaded=True).
    # This prevents pywebview's internal single-threaded WSGI adapter from deadlocking
    # when processing concurrent SSE streams (/api/logs) and POST requests (/api/remaster).
    start_cleanup_sweeper()
    port = get_free_port()
    url = f"http://127.0.0.1:{port}"

    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, threaded=True, debug=False, use_reloader=False),
        daemon=True,
    )
    server_thread.start()

    if not wait_for_server(url, timeout=90.0):
        webview.create_window(
            "AuraDSP — startup error",
            html="""
            <body style="font-family:Segoe UI,sans-serif;background:#FDEEF2;color:#2A1B18;padding:32px">
              <h2>backend belum kebuka</h2>
              <p>Flask server gagal start dalam 90 detik. Tutup app lalu buka ulang.</p>
            </body>
            """,
            width=720,
            height=420,
        )
        webview.start(debug=False)
        raise SystemExit(1)

    api = DesktopApi()
    window = webview.create_window(
        "AuraDSP",
        url,
        js_api=api,
        width=1200,
        height=800,
        min_size=(900, 600),
        confirm_close=True,
    )
    api._window = window

    # Set shutdown_event when pywebview window is closed
    window.events.closed += lambda: shutdown_event.set()

    webview.start(debug=False)
