from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stella.core.config import StellaConfig


@dataclass
class TTSResult:
    success: bool
    message: str


class PiperTTSProvider:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def speak(self, text: str) -> TTSResult:
        if not self.config.tts_enabled:
            return TTSResult(False, "TTS вимкнено.")

        if not text.strip():
            return TTSResult(False, "Порожній текст для TTS.")

        model_path = Path(self.config.tts_piper_model)

        if not model_path.exists():
            return TTSResult(False, f"Piper model not found: {model_path}")

        piper_path = shutil.which("piper")

        if not piper_path:
            return TTSResult(False, "Команду piper не знайдено. Перевір: pip install piper-tts")

        player = shutil.which(self.config.tts_output_player)

        if not player:
            fallback_player = shutil.which("pw-play") or shutil.which("aplay")

            if not fallback_player:
                return TTSResult(False, "Не знайдено аудіоплеєр: pw-play або aplay.")

            player = fallback_player

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
                synth = subprocess.run(
                    [
                        piper_path,
                        "--model",
                        str(model_path),
                        "--output_file",
                        temp_file.name,
                    ],
                    input=text,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                if synth.returncode != 0:
                    return TTSResult(
                        False,
                        f"Piper error: {synth.stderr.strip() or synth.stdout.strip()}"
                    )

                play = subprocess.run(
                    [player, temp_file.name],
                    text=True,
                    capture_output=True,
                    check=False,
                )

                if play.returncode != 0:
                    return TTSResult(
                        False,
                        f"Audio player error: {play.stderr.strip() or play.stdout.strip()}"
                    )

            return TTSResult(True, "Озвучено.")

        except Exception as error:
            return TTSResult(False, f"TTS exception: {error}")