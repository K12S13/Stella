from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from stella.core.config import StellaConfig
from stella.tts.router import TTSRouter


class TTSWorker(QObject):
    finished = Signal(bool, str, str)

    def __init__(self, config: StellaConfig, text: str, source: str) -> None:
        super().__init__()
        self.config = config
        self.text = text
        self.source = source

    def run(self) -> None:
        try:
            router = TTSRouter(self.config)
            result = router.speak(self.text)
            self.finished.emit(result.success, result.message, self.source)
        except Exception as error:
            self.finished.emit(False, f"TTS exception: {error}", self.source)