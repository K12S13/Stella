from __future__ import annotations

from stella.core.config import StellaConfig
from stella.tts.piper_provider import PiperTTSProvider, TTSResult


class TTSRouter:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config
        self.piper = PiperTTSProvider(config)

    def speak(self, text: str) -> TTSResult:
        if not self.config.tts_enabled:
            return TTSResult(False, "TTS вимкнено.")

        if self.config.tts_provider != "piper":
            return TTSResult(False, f"TTS provider не підтримується: {self.config.tts_provider}")

        return self.piper.speak(text)