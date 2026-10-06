from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from stella.core.config import StellaConfig


@dataclass
class TTSResult:
    success: bool
    message: str
    wav_path: str | None = None


def normalize_for_piper(text: str) -> str:
    replacements = {
        "’": "'",
        "ʼ": "'",
        "`": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "—": "-",
        "–": "-",
        "…": "...",
    }

    normalized = text.strip()

    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized.lower()


class PiperTTSProvider:
    def __init__(self, config: StellaConfig) -> None:
        self.config = config

    def _piper_binary(self) -> str | None:
        executable = shutil.which("piper")

        if executable:
            return executable

        venv_binary = Path(sys.executable).with_name("piper")

        if venv_binary.exists():
            return str(venv_binary)

        return None

    def _player_binary(self) -> str | None:
        configured = self.config.tts_output_player.strip()

        if configured and shutil.which(configured):
            return configured

        for candidate in ["pw-play", "paplay", "aplay"]:
            if shutil.which(candidate):
                return candidate

        return None

    def synthesize_to_wav(
        self,
        text: str,
        model_path: str | None = None,
    ) -> TTSResult:
        clean_text = normalize_for_piper(text)

        if not clean_text:
            return TTSResult(False, "TTS text is empty.")

        piper_binary = self._piper_binary()

        if piper_binary is None:
            return TTSResult(False, "Piper binary not found.")

        selected_model = model_path or self.config.tts_piper_model
        model = Path(selected_model)

        if not model.exists():
            return TTSResult(False, f"Piper model not found: {model}")

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            prefix="stella_tts_",
            delete=False,
        ) as temp_file:
            wav_path = Path(temp_file.name)

        synth = subprocess.run(
            [
                piper_binary,
                "--model",
                str(model),
                "--output_file",
                str(wav_path),
            ],
            input=clean_text,
            text=True,
            capture_output=True,
            check=False,
        )

        if synth.returncode != 0:
            try:
                wav_path.unlink(missing_ok=True)
            except OSError:
                pass

            output = "\n".join(
                part.strip()
                for part in [synth.stdout, synth.stderr]
                if part.strip()
            )

            return TTSResult(False, output or "Piper synthesis failed.")

        return TTSResult(True, "Synthesis OK.", str(wav_path))

    def play_wav(self, wav_path: str) -> TTSResult:
        player = self._player_binary()

        if player is None:
            return TTSResult(
                False,
                "No audio player found. Expected pw-play, paplay or aplay.",
            )

        play = subprocess.run(
            [player, wav_path],
            text=True,
            capture_output=True,
            check=False,
        )

        if play.returncode != 0:
            output = "\n".join(
                part.strip()
                for part in [play.stdout, play.stderr]
                if part.strip()
            )

            return TTSResult(False, output or "Audio playback failed.")

        return TTSResult(True, "Playback OK.")

    def speak(
        self,
        text: str,
        model_path: str | None = None,
    ) -> TTSResult:
        synth = self.synthesize_to_wav(text, model_path=model_path)

        if not synth.success or synth.wav_path is None:
            return synth

        try:
            return self.play_wav(synth.wav_path)

        finally:
            try:
                Path(synth.wav_path).unlink(missing_ok=True)
            except OSError:
                pass