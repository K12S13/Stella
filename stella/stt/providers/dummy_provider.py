from __future__ import annotations

import threading
import time

from stella.stt.providers.base import BaseSTTProvider, STTResult


class DummySTTProvider(BaseSTTProvider):
    name = "dummy"

    def listen_once(
        self,
        seconds: int = 30,
        stop_event: threading.Event | None = None,
    ) -> STTResult:
        for _ in range(10):
            if stop_event is not None and stop_event.is_set():
                break
            time.sleep(0.1)

        return STTResult(
            success=True,
            text="що таке кеш процесора",
            provider=self.name,
        )