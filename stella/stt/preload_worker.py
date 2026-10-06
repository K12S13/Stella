from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from stella.core.config import StellaConfig
from stella.stt.providers.whisper_provider import preload_whisper_model


class STTPreloadWorker(QObject):
    finished = Signal(bool, str)

    def __init__(self, config: StellaConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        if self.config.stt_provider != "whisper":
            self.finished.emit(True, "STT preload skipped: provider is not whisper.")
            return

        ok, message = preload_whisper_model(self.config)
        self.finished.emit(ok, message)