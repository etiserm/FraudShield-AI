from __future__ import annotations

import socket
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

import uvicorn
import webview

from api import app


def choose_local_port() -> int:
    """Find an available loopback port for this app instance."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def wait_for_server(server: uvicorn.Server, worker: threading.Thread, url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not worker.is_alive():
            raise RuntimeError("The local FraudShield service stopped during startup.")
        try:
            with urlopen(url, timeout=0.5) as response:
                if response.status == 200:
                    return
        except (TimeoutError, URLError, OSError):
            time.sleep(0.1)
    server.should_exit = True
    raise RuntimeError("The local FraudShield service did not start in time.")


def main() -> None:
    port = choose_local_port()
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
        )
    )
    worker = threading.Thread(target=server.run, name="fraudshield-api", daemon=True)
    worker.start()

    try:
        wait_for_server(server, worker, f"{base_url}/api/health")
        webview.create_window(
            "FraudShield AI",
            url=base_url,
            width=1200,
            height=820,
            min_size=(760, 620),
            background_color="#101820",
        )
        webview.start(debug=False)
    finally:
        server.should_exit = True
        worker.join(timeout=10)


if __name__ == "__main__":
    main()
