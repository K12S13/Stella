from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class STTResult:
    success: bool
    text: str
    error: str | None = None
    provider: str = "unknown"


class BaseSTTProvider:
    name = "base"

    def listen_once(
        self,
        seconds: int = 30,
        stop_event: threading.Event | None = None,
    ) -> STTResult:
        raise NotImplementedError