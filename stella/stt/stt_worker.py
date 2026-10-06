from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from stella.core.config import StellaConfig
from stella.stt.router import STTRouter
from stella.text.stt_text_cleanup import cleanup_stt_text


class STTWorker(QObject):
    finished = Signal(bool, str, str, str)
    # success, text, error, provider

    def __init__(self, config: StellaConfig) -> None:
        super().__init__()
        self.config = config
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        try:
            router = STTRouter(self.config)
            result = router.listen_once(stop_event=self.stop_event)

            cleaned_text = cleanup_stt_text(result.text) if result.text else ""

            self.finished.emit(
                result.success,
                cleaned_text,
                result.error or "",
                result.provider,
            )

        except Exception as error:
            self.finished.emit(False, "", str(error), "exception")